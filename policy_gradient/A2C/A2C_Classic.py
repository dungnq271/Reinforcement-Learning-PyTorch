import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
import gym
import shutil

max_train_steps = 20000
gamma = 0.999
lr = 1e-4
update_interval = 5
n_workers = 8
saved_path = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'saved_models', 'a2c_classic.pt')
log_dir = 'runs/classic'
pretrain = True
episode_score = []
test_interval = 100

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')
print(f'Using {n_workers} worker(s)')

if os.path.isdir(log_dir):
    shutil.rmtree(log_dir, ignore_errors=True)


class ActorCriticNet(nn.Module):
    def __init__(self, input_dim=4, n_actions=2):
        super(ActorCriticNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU()
        )
        self.actor = nn.Linear(512, n_actions)
        self.critic = nn.Linear(512, 1)

    def predict_probs(self, x):
        x = torch.from_numpy(x).float().to(device)
        x = self.net(x)
        x = self.actor(x)
        policy = F.softmax(x, dim=-1)
        return policy

    def predict_reward(self, x):
        # x = self.normalize_observations(x)
        x = x.float().to(device)
        x = self.net(x)
        value = self.critic(x)
        return value


def add_log():
    writer.add_scalar('Average Score', score, step_idx)
    writer.add_scalar('Loss', loss, step_idx)
    writer.add_scalar('Actor Loss', policy_loss, step_idx)
    writer.add_scalar('Critic Loss', value_loss, step_idx)
    writer.add_scalar('Entropy', entropy, step_idx)
    # writer.add_scalar('Learning Rate', scheduler.get_last_lr()[-1], step_idx)
    writer.flush()


def test(step_idx, model):
    env = gym.make('CartPole-v1')
    score = 0.0
    done = False
    num_test = 10

    for i in range(num_test):
        s = env.reset()
        while not done:
            p = model.predict_probs(s)  # p(8,6); v(8,1)
            dist = Categorical(p)
            a = dist.sample()
            s_prime, r, done, _ = env.step(a.cpu().numpy())
            score += r
            s = s_prime
        done = False
    mean_score = score / num_test
    print(f"Step:{step_idx}, avg score: {mean_score:.1f}")
    return mean_score


def compute_all_return(final_val, r_list, mask):
    v_target = []
    accumulate_r = final_val.cpu()
    for reward, mask in zip(r_list[::-1], mask[::-1]):
        rw, m = torch.tensor(reward), torch.tensor(mask)
        accumulate_r = rw + gamma * accumulate_r * m
        v_target.append(accumulate_r)
    return torch.stack(v_target[::-1]).float().to(device)


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    writer = SummaryWriter(log_dir)
    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    step_idx = 0

    model = ActorCriticNet(input_dim=env.observation_space.shape[0],
                           n_actions=env.action_space.n).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # scheduler = CosineAnnealingLR(optimizer, T_max=max_train_steps // update_interval)

    if pretrain:
        print('Loading checkpoint...')
        model.load_state_dict(torch.load(saved_path))

    print('Start training')
    s = env.reset()
    s_prime = s

    while step_idx < max_train_steps:
        s_lst, p_lst, v_lst, r_lst, e_lst, mask_lst = [], [], [], [], [], []
        for _ in range(update_interval):
            # env.render()
            p = model.predict_probs(s)
            dist = Categorical(p)
            a = dist.sample()
            s_prime, r, done, _ = env.step(a.cpu().numpy())

            s_lst.append(torch.from_numpy(s))
            p_lst.append(p[a])  # (5)
            r_lst.append(r/100.0)  #
            e_lst.append(dist.entropy())  # (5)
            mask_lst.append(1 - done)  # (5)

            s = s_prime
            step_idx += 1
            if done:
                s = env.reset()

        v_final = model.predict_reward(torch.from_numpy(s_prime).unsqueeze(0))
        # print("v_final, r_lst, mask_lst", v_final, r_lst, mask_lst)
        target_lst = compute_all_return(v_final.squeeze(0).detach().clone(), r_lst, mask_lst).detach()  # (5,1)

        p_lst, s_lst = torch.stack(p_lst), torch.stack(s_lst)  # (5), (5,4)
        v_lst = model.predict_reward(s_lst)  # (5,1)
        # print("target_lst, v_lst", target_lst, v_lst)
        advantage = target_lst - v_lst
        # print("advantage, p_lst", advantage, p_lst)
        policy_loss = -(torch.log(p_lst).unsqueeze(1) * advantage.detach()).mean()
        # print("policy_loss", torch.log(p_lst).unsqueeze(1) * advantage.detach())
        entropy = torch.stack(e_lst).mean()
        value_loss = F.smooth_l1_loss(v_lst, target_lst)
        loss = policy_loss - 0.01 * entropy + 0.5 * value_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        # scheduler.step()
        if step_idx % test_interval == 0:
            score = test(step_idx, model)
            torch.save(model.state_dict(), saved_path)
            add_log()
        # break
