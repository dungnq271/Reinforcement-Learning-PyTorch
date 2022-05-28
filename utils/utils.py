import gym
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import numpy as np
import matplotlib.pyplot as plt
import random


class ChannelFirst(gym.ObservationWrapper):
    def __init__(self, env_id):
        super().__init__(env_id)
        new_shape = np.roll(self.observation_space.shape, shift=1)  # shape: (H, W, C) -> (C, H, W)

        # Update because this is the last wrapper in the hierarchy, we'll be pooling the env for observation shape info
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=new_shape, dtype=np.uint8)

    def observation(self, observation):
        if len(observation.shape) > 3:
            return np.moveaxis(observation, 3, 1)
        else:
            return np.moveaxis(observation, 2, 0)


class LinearSchedule:
    def __init__(self, start_value, end_value, schedule_duration):
        self.start_value = start_value
        self.end_value = end_value
        self.schedule_duration = schedule_duration

    def __call__(self, num_steps):
        progress = np.clip(num_steps / self.schedule_duration, a_min=None, a_max=1)
        return self.start_value + (self.end_value - self.start_value) * progress


def get_parallel_env_wrapper(env_id, n_envs, render_mode=None, num_frame_stack=4):
    env_wrapped = make_atari_env(env_id, n_envs=n_envs, env_kwargs={'render_mode': render_mode})
    env_wrapped = VecFrameStack(env_wrapped, n_stack=num_frame_stack)
    env_wrapped = ChannelFirst(env_wrapped)
    return env_wrapped


def get_single_env_wrapper(env_id, render_mode=None):
    env_wrapped = ChannelFirst(AtariWrapper(gym.make(env_id, render_mode=render_mode)))
    return env_wrapped


def visualize_state(state):
    stacked_frames = np.hstack([np.repeat((img * 255).astype(np.uint8)[:, :, np.newaxis], 3, axis=2) for img in
                                state[0]])  # (C, H, W) -> (H, C*W, 3)
    plt.imshow(stacked_frames)
    plt.show()


def normalize_observations(x):
    dim = 1
    x = (x - x.mean(dim=dim, keepdims=True)) / \
        (x.std(dim=dim, unbiased=True, keepdims=True) + 1e-10)
    return x


def set_random_seeds(env, seed):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        env.action_space.seed(seed)
        env.seed(seed)

        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


# test utils
if __name__ == '__main__':
    # import matplotlib.pyplot as plt
    #
    # schedule = LinearSchedule(start_value=1., end_value=0.1, schedule_duration=50)
    # schedule_values = []
    # for i in range(100):
    #     schedule_values.append(schedule(i))
    #
    # plt.plot(schedule_values)
    # plt.show()

    # env = 'ALE/SpaceInvaders-v5'
    env = 'ALE/Breakout-v5'
    env = get_single_env_wrapper(env)
    # env = get_parallel_env_wrapper(env, n_envs=1)
    print(env.observation_space.shape)

    episode_reward = 0
    screen = env.reset()
    print(screen.shape)
    while True:
        # action = np.array([env.action_space.sample() for _ in range(8)])
        action = env.action_space.sample()
        screen, reward, done, _ = env.step(action)
        print(screen.shape)
        # visualize_state(screen)
        episode_reward += reward
        if done:
            print('Reward: %s' % episode_reward)
            break
