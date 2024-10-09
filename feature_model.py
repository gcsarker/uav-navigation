import gymnasium as gym
import torch as T
from torch import nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torchvision.models import VisionTransformer

# resnet = models.resnet18()

class feature_model(BaseFeaturesExtractor):
    def __init__(
            self, 
            observation_space: gym.spaces.Dict, 
            features_dim: int = 256+64,
        ):
        super().__init__(observation_space, features_dim)
        
        self.depth_inp_dims = (1, 224, 224)
        

        # depth features strided
        self.depth_features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
        )

        self.depth_kernel_size = self._get_kernel_size(self.depth_inp_dims, self.depth_features)


        #self.lstm = nn.LSTM(128, 16, 1, batch_first=True, bidirectional=True)
        #self.gru = nn.GRU(128, 64, 1, batch_first=True, bidirectional = True)
        self.gru = nn.GRU(128*self.depth_kernel_size[0]*self.depth_kernel_size[1],
                         256, 1, batch_first = True)
        
        self.angle_features = nn.Sequential(
            nn.Linear(1,32),
            nn.ReLU(inplace=True),
            nn.Linear(32,64),
            nn.ReLU(inplace=True)
        )


    def _get_kernel_size(self, input_dims, layer):
        dummy_input = T.zeros(1, *input_dims)  # batch size, input channels, height, width
        dummy_output = layer(dummy_input)
        return (dummy_output.shape[2], dummy_output.shape[3])

    def forward(self, observations):
        depth_map = observations['depth_map']
        angle = observations['angle']
        batch, timestep, c, h, w = depth_map.size()
        d = depth_map.view(batch * timestep, c, h, w)
        d = self.depth_features(d)
        #d = d.mean([2,3])
        d = d.view(batch, timestep, -1)

        # h0 = T.zeros(1*2, batch, 16)
        # c0 = T.zeros(1*2, batch, 16)
        # d, _ = self.lstm(d, (h0,c0))

        h0 = T.zeros(1, batch, 256)
        d, _ = self.gru(d, h0)
        d = d[: ,-1, :]
        d = nn.Tanh()(d)

        a = self.angle_features(angle)

        x = T.cat([d,a], dim = 1)
        
        return x

