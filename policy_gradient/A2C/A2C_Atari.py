import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from utils.utils import get_parallel_env_wrapper, normalize_observations

max_train_steps = 100000
gamma = 0.99
lr = 1e-4
update_interval = 5
n_workers = 8
saved_path = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'saved_models', 'a2c_atari.pt')
log_dir = 'runs/atari'
pretrain = True
episode_score = []
test_interval = 100

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if os.path.isdir(log_dir):
    shutil.rmtree(log_dir, ignore_errors=True)


def weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight.data)


class ActorCriticNet(nn.Module):
    def __init__(self, n_channels=4, n_actions=6):
        super(ActorCriticNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_channels, 32, (8, 8), (4, 4)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (4, 4), (2, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), (1, 1)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU()
        )
        self.actor = nn.Linear(512, n_actions)
        self.critic = nn.Linear(512, 1)

    def forward(self, x):
        x = x / 255.0
        x = torch.from_numpy(x).float()
        x = normalize_observations(x).to(device)
        x = self.net(x)
        probs = F.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        return probs, value


def test(play=False):
    num_iter = 10
    if play:
        num_iter = 1
    game = get_parallel_env_wrapper(env_name, n_envs=1)
    mean_score = 0
    state = game.reset()

    for _ in range(num_iter):
        while True:
            proba, _ = model(state)
            dis = Categorical(proba)
            action = dis.sample()
            next_state, reward, finish, _ = game.step(action)
            state = next_state
            mean_score += reward
            if finish[0]:
                break
    return mean_score / num_iter


def compute_all_return(final_val, r_list, mask_list):
    v_target = []
    accumulate_r = final_val.cpu()
    for reward, mask in zip(r_list[::-1], mask_list[::-1]):
        rw, m = torch.from_numpy(reward), torch.from_numpy(mask)
        accumulate_r = rw + gamma * accumulate_r * m
        v_target.append(accumulate_r)
    return torch.stack(v_target[::-1]).float().to(device)


def add_to_writer():
    writer.add_scalar('Average Score', score, step_idx)
    writer.add_scalar('Loss', loss, step_idx)
    writer.add_scalar('Actor Loss', policy_loss, step_idx)
    writer.add_scalar('Critic Loss', value_loss, step_idx)
    writer.add_scalar('Entropy', entropy, step_idx)
    writer.flush()


if __name__ == '__main__':
    print(f'Using device: {device}')
    print(f'Using {n_workers} worker(s)')

    torch.autograd.set_detect_anomaly(True)
    writer = SummaryWriter(log_dir, comment='Atari')
    env_name = 'ALE/SpaceInvaders-v5'
    env = get_parallel_env_wrapper(env_name, n_envs=n_workers)
    step_idx = 0

    model = ActorCriticNet().to(device)
    optimizer = optim.RMSprop(model.parameters(), lr=lr, eps=1e-5)

    if pretrain:
        print('Loading checkpoint...')
        model.load_state_dict(torch.load(saved_path))
    else:
        model.apply(weights_init)

    print('Start training')
    s = env.reset()  # (8, 4, 84, 84)

    while step_idx < max_train_steps:
        p_lst, v_lst, r_lst, e_lst, mask_lst = [], [], [], [], []
        for _ in range(update_interval):
            p, v = model(s)  # (8,6)
            dist = Categorical(p)
            a = dist.sample().detach()
            s_prime, r, done, _ = env.step(a)

            v_lst.append(v)  # 5*(8,1)
            p_lst.append(p.gather(1, a.unsqueeze(1)))  # 5*(8,1)
            r_lst.append(r[:, np.newaxis])  # 5*(8,1)
            e_lst.append(dist.entropy().unsqueeze(1))  # 5*(8,1)
            mask_lst.append(1 - done[:, np.newaxis])  # 5*(8,1)

            s = s_prime
            step_idx += 1

        with torch.no_grad():
            _, v_final = model(s_prime)
            target_lst = compute_all_return(v_final.clone(), r_lst, mask_lst)  # (5,8,1)

        p_lst, v_lst = torch.stack(p_lst), torch.stack(v_lst)
        advantage = target_lst - v_lst
        policy_loss = -(torch.log(p_lst) * advantage.detach()).mean()
        value_loss = advantage.pow(2).mean()
        entropy = torch.stack(e_lst).mean()
        loss = policy_loss - 0.01 * entropy + 0.5 * value_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        if step_idx % test_interval == 0:
            score = test()
            print(f'Step {step_idx} Mean Score {score[0]:.2f}')
            torch.save(model.state_dict(), saved_path)
            add_to_writer()
