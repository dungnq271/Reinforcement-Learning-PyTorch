import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque
import math
import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# CartPole-v1: Solve after 500-600 episodes

EPISODES = 2000
MAX_BUFFER = 10000
BATCH_SIZE = 128
GAMMA = 0.999
LR = 1e-4
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
UPDATE_STEPS = 100

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')


class DQNAgent(nn.Module):
    def __init__(self, memory_capacity, input_dim, output_dim):
        super(DQNAgent, self).__init__()
        self.memory = deque([], maxlen=memory_capacity)
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def push_to_memory(self, transition):
        self.memory.append(transition)

    def sample_from_memory(self):
        return random.sample(self.memory, BATCH_SIZE)

    def select_action(self, s, t):
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1 * t / EPS_DECAY)
        if np.random.uniform(0, 1) < eps_threshold:
            action = env.action_space.sample()
        else:
            action = torch.argmax(self(s), dim=-1).detach().item()
        return action

    def optimize(self, target):
        if len(self.memory) < BATCH_SIZE:
            return
        transition_batch = self.sample_from_memory()
        transition_batch = list(zip(*transition_batch))
        assert len(transition_batch) == 5

        states = torch.cat(transition_batch[0])
        actions = torch.tensor(transition_batch[1], device=device).unsqueeze(dim=0)
        rewards = torch.tensor(transition_batch[2], device=device)
        next_states = torch.cat(transition_batch[3]).to(device)
        final_mask = torch.tensor(transition_batch[4], device=device).float()
        assert states.shape == next_states.shape == (BATCH_SIZE, 4)

        with torch.no_grad():
            q_prime = target(next_states)
            target = rewards + GAMMA * \
                     q_prime.max(-1)[0] * (1 - final_mask)
        q = self(states)
        output = q.gather(1, actions.T).squeeze()

        assert target.shape == output.shape
        criterion = nn.SmoothL1Loss()
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        for param in self.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()


if __name__ == '__main__':
    writer = SummaryWriter()
    env = gym.make('CartPole-v1')
    env.seed(23)
    state_dim = len(env.reset())
    n_action = env.action_space.n

    policy_net = DQNAgent(MAX_BUFFER, state_dim, n_action).to(device)
    target_net = DQNAgent(MAX_BUFFER, state_dim, n_action).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.RMSprop(policy_net.parameters(), lr=LR)

    episode_score = []
    mean_scores = []

    for episode in range(EPISODES):
        state = torch.from_numpy(env.reset()).unsqueeze(0)
        done = False
        score = 0.0

        if episode == 0:
            writer.add_graph(policy_net, input_to_model=state, verbose=False)

        while done is not True:
            env.render()
            if episode >= EPISODES-100:
                time.sleep(0.1)
            action = policy_net.select_action(state, episode)
            next_state, reward, done, _ = env.step(action)
            next_state = torch.from_numpy(next_state).unsqueeze(0)
            score += reward
            policy_net.push_to_memory((state, action, reward, next_state, done))
            policy_net.optimize(target_net)
            state = next_state

        episode_score.append(score)
        writer.add_scalar("Score", score)
        writer.flush()

        if episode % UPDATE_STEPS == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if episode % 100 == 99:
            mean_score = np.mean(episode_score)
            mean_scores.append(mean_score)
            episode_score = []
            print(f'Episode {episode+1} Score {mean_score}')
            if mean_score >= 195.0:
                print(f'Game solved after {episode+1} episodes!!!')

    env.close()
    writer.close()
    plt.plot(mean_scores)
