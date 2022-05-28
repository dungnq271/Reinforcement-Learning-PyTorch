import gym
import numpy as np
import matplotlib.pyplot as plt
import time


EPISODES = 5000
ALPHA = 0.81
GAMMA = 0.96
RENDER = False
epsilon = 0.9
interval = 100
scores = []


def policy(env, Q, s, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        a = env.action_space.sample()
    else:
        a = np.argmax(Q[s, :])
    return a

def epsilon_greedy(env, n_action, Q, s, epsilon):
    if np.random.uniform(0, 1) < epsilon/n_action + 1 - epsilon:
        a = env.action_space.sample()
    else:
        a = np.argmax(Q[s, :])
    return a

if __name__ == '__main__':
    env = gym.make('FrozenLake-v1')
    env.seed(23)

    n_state = env.observation_space.n
    n_action = env.action_space.n

    Q = np.zeros((n_state, n_action))
    avg_rewards = []
    rewards = []

    for episode in range(EPISODES):
        s = env.reset()
        done = False

        while done is not True:
            if episode >= EPISODES-100:
                if RENDER:
                    env.render()
                    time.sleep(0.1)

            a = policy(env, Q, s, epsilon)
            # a = epsilon_greedy(env, n_action, Q, s, epsilon)
            s_prime, r, done, info = env.step(a)
            # update Q value
            Q[s, a] = Q[s, a] + ALPHA * (r + GAMMA * np.max(Q[s_prime, :]) - Q[s, a])
            s = s_prime

        rewards.append(r)
        epsilon -= 0.001

        if episode % 100 == 0 and episode != 0:
            avg_reward = np.mean(rewards)
            avg_rewards.append(avg_reward)
            rewards = []
            print(f'Episode {episode} Mean Reward {avg_reward}')

    env.close()

    plt.plot(avg_rewards)
    plt.xlabel('episodes (100\'s)')
    plt.ylabel('average reward')
    plt.show()

