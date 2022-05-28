import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# CartPole-v1: Solve after 700 episodes

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

EPISODES = 10000
LR = 2e-4
GAMMA = 0.98
UPDATE_INTERVAL = 100
BASELINE = False


class PolicyNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, output_dim)
        self.optimizer = optim.Adam(self.parameters(), lr=LR)
        self.data = []

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), -1)
        return x

    def reset(self):
        self.data = []

    def push_data(self, item):
        self.data.append(item)

    def train_step(self):
        G = 0
        self.optimizer.zero_grad()
        for state, reward, proba in self.data[::-1]:
            G = reward + GAMMA * G
            loss = -torch.log(proba) * G
            loss.backward()
        self.optimizer.step()


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    env.seed(23)
    state_dim = len(env.reset())
    n_action = env.action_space.n

    policy = PolicyNet(state_dim, n_action).to(device)
    episode_score = []
    mean_scores = []

    for i in range(EPISODES):
        score = 0.0
        s = env.reset()
        done = False

        while done is not True:
            env.render()
            prob = policy(torch.from_numpy(s).float().to(device))
            m = Categorical(prob)
            a = m.sample()
            s_prime, r, done, info = env.step(a.item())
            policy.push_data((s, r, prob[a]))
            score += r
            s = s_prime
            # time.sleep(0.1) # for slow training

        episode_score.append(score)
        policy.train_step()
        policy.reset()

        if i % UPDATE_INTERVAL == 0 and i != 0:
            mean_score = np.mean(episode_score)
            mean_scores.append(mean_score)
            episode_score = []
            print(f'Episode {i} Score {mean_score}')
            if mean_score >= 195.0:
                print(f'Game solved after {i} episodes!!!')
                break

    env.close()
    plt.plot(mean_scores)
