from utils.video_utils import preprocess, create_gif
from utils.utils import get_parallel_env_wrapper
from policy_gradient.A2C.A2C_Atari import ActorCriticNet
import torch
from torch.distributions import Categorical
import os

pretrained_path = os.path.join(os.path.dirname(__file__), 'saved_models', 'a2c_atari.pt')
gif_path = os.path.join(os.path.dirname(__file__), 'data', 'SpaceInvaders_A2C1.gif')
device = 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == '__main__':
    env = 'ALE/SpaceInvaders-v5'
    env = get_parallel_env_wrapper(env, render_mode='human', n_envs=1)

    model = ActorCriticNet().to(device)
    model.load_state_dict(torch.load(pretrained_path))
    model.eval()

    frames = []
    saved_frames = []
    max_reward = 0
    episode_reward = 0
    state = env.reset()

    for _ in range(10):
        while True:
            frames.append(preprocess(state))
            proba, _ = model(state)
            dis = Categorical(proba)
            action = dis.sample()
            next_state, reward, finish, _ = env.step(action)
            state = next_state
            episode_reward += reward
            if finish[0]:
                print(episode_reward)
                episode_reward = 0
                break

    create_gif(gif_path, frames)
