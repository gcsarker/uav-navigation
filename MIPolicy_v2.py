import gymnasium as gym
import torch as th
from torch import nn
import numpy as np
from typing import Dict, Any

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class MultiFE(BaseFeaturesExtractor):
    def __init__(
            self, 
            observation_space: gym.spaces.Dict, 
            features_dim: int = 256,
            nb_features: int = 4,
        ):
        super().__init__(observation_space, features_dim+nb_features)
        # We assume CxHxW images (channels first)
        n_input_channels, H_in, W_in = observation_space["depth_map"].shape

        layer1 = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(64),
            nn.ReLU())
        layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(128),
            nn.ReLU())
        layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        
        #layers = [layer1, layer2, layer3, layer4]
        layers = [layer1, layer2, layer3]


        self.cnn = nn.Sequential(*layers,nn.Flatten())

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space["depth_map"].sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.LeakyReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        images_enc = self.linear(self.cnn(observations["depth_map"]))
        return th.cat([images_enc, observations["position"], observations['target_location']], dim=1)
    
    
# import rl_env
# env = gym.make('airsim_drone-v0')
# model = MultiFE(observation_space=env.observation_space, features_dim=128)

# print(model)

# import torchinfo
# torchinfo.summary(model, (1,84, 84), batch_dim = 0, col_names = ("input_size", "output_size", "num_params"), verbose = 1)
