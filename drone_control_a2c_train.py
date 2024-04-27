import gymnasium as gym
import time
import rl_env
import pygame
import numpy as np
import threading
from MIPolicy_v2 import *

from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

#env = gym.make('airsim_drone-v0')
env = make_vec_env('airsim_drone-v0', n_envs=2, monitor_dir='files/')

policy_kwargs = dict(
    net_arch=[64,32,16],
    features_extractor_class=MultiFE,
    features_extractor_kwargs={
              "features_dim":128,
              "nb_features": 4
            }
)

# model = PPO(
#     policy="MultiInputPolicy",
#     env = env,
#     policy_kwargs=policy_kwargs,
#     n_steps=128,
#     verbose=1,
#     ent_coef=0.1,
#     stats_window_size=2
#     )

model = A2C(
    policy="MultiInputPolicy",
    env = env,
    policy_kwargs=policy_kwargs,
    n_steps=128,
    verbose=1,
    ent_coef=0.1,
    stats_window_size=2
    )


model.learn(total_timesteps=4000, log_interval=1, progress_bar=True)
model.save('drone_a2c')


# episodes = 10
# for episode in range(1, episodes+1):
#     obs = env.reset()
#     terminated = False
#     total_reward = 0

#     while not terminated:
#         action = env.action_space.sample()
#         print(f"action : {action}")
#         obs, reward, terminated, _ , info  = env.step(action)
#         print(obs)

