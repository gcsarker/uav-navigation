import gymnasium as gym
import time
import rl_env
import pygame
import numpy as np
import threading
from MIPolicy_v2 import *

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

env = gym.make('airsim_drone-v0')
env = Monitor(env=env, filename='files/logfile.csv')


policy_kwargs = dict(
    net_arch=[64,32,16],
    features_extractor_class=MultiFE,
    features_extractor_kwargs={
              "features_dim":128,
              "nb_features": 4
            }
)

config = {
    'batch_size': 32,
    'buffer_size': 10000,
    'exploration_final_eps': 0.07,
    'exploration_fraction': 0.5,
    'gamma': 0.98,
    'gradient_steps': 8, # don't do a single gradient update, but 8
    'learning_rate': 0.004,
    'learning_starts': 1000,
    'target_update_interval': 600, # see below, the target network gets overwritten with the main network every 600 steps
    'train_freq': 16, # don't train after every step in the environment, but after 16 steps
}

model = DQN(
    policy="MultiInputPolicy",
    env = env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    **config
)

model.learn(total_timesteps=2000, log_interval=1, progress_bar=True)
model.save('drone_dqn')

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

