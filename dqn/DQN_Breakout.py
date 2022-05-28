import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
from collections import deque
import math
import os
import shutil
from utils.utils import get_single_env_wrapper

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

episodes = 100000
max_buffer = 10000
batch_size = 32
gamma = 0.999
lr = 1e-4
eps_start = 1
eps_end = 0.1
eps_decay = 200
update_steps = 100
print_interval = 100
pretrain = True
saved_path = os.path.join(os.path.dirname(__file__), os.pardir, 'saved_models', 'dqn_breakout.pt')
log_dir = 'runs/dqn_breakout'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

if os.path.isdir(log_dir):
    shutil.rmtree(log_dir, ignore_errors=True)


class DQNAtari(nn.Module):
    def __init__(self, memory_capacity, n_channels=4, n_actions=4):
        super(DQNAtari, self).__init__()
        self.memory = deque([], maxlen=memory_capacity)
        self.net = nn.Sequential(
            nn.Conv2d(n_channels, 32, (8, 8), (4, 4)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (4, 4), (2, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), (1, 1)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        x = x.float().div(255)
        # x = normalize_observations(x)
        x = self.net(x.to(device))
        return x

    def push_to_memory(self, transition):
        self.memory.append(transition)

    def sample_from_memory(self):
        return random.sample(self.memory, batch_size)

    def select_action(self, s, t):
        eps_threshold = eps_end + (eps_start - eps_end) * \
                        math.exp(-1 * t / eps_decay)
        q_mean = 0
        if np.random.uniform(0, 1) < eps_threshold:
            a = [env.action_space.sample()]
        else:
            q = self(s).detach()
            a = torch.argmax(q, dim=-1).cpu()
            q_mean = q.mean().cpu()
        return a, q_mean

    def optimize(self, target):
        if len(self.memory) < batch_size:
            return
        transition_batch = self.sample_from_memory()
        transition_batch = list(zip(*transition_batch))

        states = torch.cat(transition_batch[0])  # 128,4,84,84
        actions = torch.tensor(transition_batch[1]).unsqueeze(1).to(device)  # 128,1
        assert actions.shape == (batch_size, 1), print(actions.shape)
        rewards = torch.tensor(np.array(transition_batch[2])).to(device)
        assert rewards.shape == (batch_size, 1), print(actions.shape)
        next_states = torch.cat(transition_batch[3])
        final_mask = torch.tensor(np.array(transition_batch[4])).to(device).float()
        assert final_mask.shape == (batch_size, 1), print(actions.shape)

        assert states.shape == next_states.shape == (batch_size, 4, 84, 84), print(states.shape)

        with torch.no_grad():
            q_prime = target(next_states)
            target = rewards + gamma * q_prime.max(-1)[0].unsqueeze(1) * (1 - final_mask)
        q = self(states)
        output = q.gather(1, actions)

        assert target.shape == output.shape, print(target.shape, output.shape)
        criterion = nn.SmoothL1Loss()
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 1)
        optimizer.step()
        return loss.item()


def test(play=False):
    num_iter = 10
    if play:
        num_iter = 1
    game = get_single_env_wrapper(env_name)
    mean_reward = 0
    s = game.reset()

    for _ in range(num_iter):
        while True:
            a, _ = policy_net.select_action(torch.from_numpy(s), episode)
            s_prime, r, finish, _ = game.step(a)
            s = s_prime
            mean_reward += reward
            if finish[0]:
                break
    return mean_reward / num_iter


def add_to_writer():
    writer.add_scalar('Average Reward', avg_reward, episode)
    writer.add_scalar('Loss', ls, episode)
    if len(q_avg_lst) != 0:
        writer.add_scalar('Average Q', np.mean(q_avg_lst), episode)
    else:
        writer.add_scalar('Average Q', 0, episode)
    writer.flush()


if __name__ == '__main__':
    writer = SummaryWriter(log_dir)
    env_name = 'ALE/Breakout-v5'
    env = get_single_env_wrapper(env_name)
    n_action = env.action_space.n

    policy_net = DQNAtari(max_buffer, n_actions=n_action).to(device)
    target_net = DQNAtari(max_buffer, n_actions=n_action).to(device)

    if pretrain:
        print('Loading checkpoint...')
        policy_net.load_state_dict(torch.load(saved_path))

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.RMSprop(policy_net.parameters(), lr=lr, momentum=0.95)

    state = torch.from_numpy(env.reset())  # 1,4,84,84
    for episode in range(episodes):
        done = [False]
        q_avg_lst = []

        while not done[0]:
            action, q_avg = policy_net.select_action(state, episode)
            next_state, reward, done, _ = env.step(action)
            next_state = torch.from_numpy(next_state)
            policy_net.push_to_memory((state, action[0], reward, next_state, done))
            ls = policy_net.optimize(target_net)
            state = next_state
            if q_avg != 0:
                q_avg_lst.append(q_avg)

        if episode % update_steps == 0 and episode != 0:
            target_net.load_state_dict(policy_net.state_dict())
            
        if episode % print_interval == 0 and episode != 0:
            avg_reward = test()
            print(f'Episode {episode} Average Reward {avg_reward[0]}')
            torch.save(policy_net.state_dict(), saved_path)
            add_to_writer()
