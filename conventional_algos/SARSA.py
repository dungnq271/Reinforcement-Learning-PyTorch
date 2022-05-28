import gym
import numpy as np
import matplotlib.pyplot as plt
import time
from Q_learning import policy

EPISODES = 1500
ALPHA = 0.81
GAMMA = 0.98
RENDER = False
epsilon = 0.9
interval = 100
scores = []

if __name__ == '__main__':
    env = gym.make('FrozenLake-v1')
    env.seed(23)

    Q = np.zeros((env.observation_space.n, env.action_space.n))
    avg_rewards = []
    rewards = []

    for episode in range(EPISODES):
        s = env.reset()
        done = False

        while done is not True:
            if RENDER:
                env.render()
                time.sleep(0.01)

            a = policy(env, Q, s, epsilon)
            s_prime, r, done, info = env.step(a)
            a_prime = policy(env, Q, s_prime, epsilon)
            # update Q value
            Q[s, a] = Q[s, a] + ALPHA * (r + GAMMA * Q[s_prime, a_prime] - Q[s, a])
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
