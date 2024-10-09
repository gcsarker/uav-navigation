import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import torchvision
import torchvision._utils
import torch.nn.functional as F

class mhaFeatureExtractor(nn.Module):
    def __init__(self, img_size=112, patch_size=16, in_channels=1, embed_dim=256, num_heads=8):
        super(mhaFeatureExtractor, self).__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Patch Embedding
        self.patch_embed = nn.Conv2d(64, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Position Embedding
        self.position_embed = nn.Parameter(T.zeros(1, self.num_patches, embed_dim))
        
        # Multi-head Self Attention
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        
        # Fully connected layer for feature extraction
        self.fc = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        # Extract patches and embed them
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = self.patch_embed(x)  # Shape: (batch_size, embed_dim, num_patches, num_patches)
        x = x.flatten(2)  # Shape: (batch_size, embed_dim, num_patches)
        x = x.transpose(1, 2)  # Shape: (batch_size, num_patches, embed_dim)
        
        # Add position embedding
        x = x + self.position_embed
        
        # Apply multi-head self-attention
        x = x.transpose(0, 1)  # Shape: (num_patches, batch_size, embed_dim)
        x, _ = self.attention(x, x, x)
        x = x.transpose(0, 1)  # Shape: (batch_size, num_patches, embed_dim)
        
        # Flatten and apply fully connected layer for feature extraction
        x = x.mean(dim=1)  # Shape: (batch_size, embed_dim)
        x = self.fc(x)  # Shape: (batch_size, embed_dim)
        
        return x
    
class ActorNetwork(nn.Module):
    def __init__(self, n_actions, depth_inp_dims, alpha,
                 chkpt_dir='checkpoints/'):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')

        self.depth_inp_dims = depth_inp_dims
        self.depth_features = mhaFeatureExtractor()

        # self.lstm = nn.LSTM(128, 16, 1, batch_first=True, bidirectional=True)
        self.gru = nn.GRU(256,128, 1, batch_first = True)
        
        self.angle_features = nn.Sequential(
            nn.Linear(1,32),
            nn.ReLU(inplace=True),
            nn.Linear(32,64),
            nn.ReLU(inplace=True)
        )


        self.actor = nn.Sequential(
                nn.Linear(128+64, 32),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(32, n_actions),
                nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        # self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.device = T.device('cpu')
        self.to(self.device)

    def forward(self, depth_map, angle):
        # batch, timestep, (channel, height, width)

        batch, timestep, c, h, w = depth_map.size()
        d = depth_map.view(batch * timestep, c, h, w)
        d = self.depth_features(d)
        d = d.view(batch, timestep, -1)

        # h0 = T.zeros(1*2, batch, 16) # num_layer*2 for bidirectional
        # c0 = T.zeros(1*2, batch, 16)
        # d, _ = self.lstm(d, (h0,c0))

        h0 = T.zeros(1, batch, 128) # num_layer*2 for bidirectional
        d, _ = self.gru(d, h0)
        d = d[: ,-1, :]
        d = nn.Tanh()(d)
        
        a = self.angle_features(angle)

        x = T.cat([d,a], dim = 1)

        x = self.actor(x)
        dist = Categorical(x)
        
        return dist

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))



class CriticNetwork(nn.Module):
    def __init__(self, depth_inp_dims, alpha, 
            chkpt_dir='checkpoints/'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')
        self.depth_inp_dims = depth_inp_dims
        

        # depth features strided
        self.depth_features = mhaFeatureExtractor()
        
        self.gru = nn.GRU(256, 128, 1, batch_first = True)
        
        self.angle_features = nn.Sequential(
            nn.Linear(1,32),
            nn.ReLU(inplace=True),
            nn.Linear(32,64),
            nn.ReLU(inplace=True)
        )
        

        self.critic = nn.Sequential(
                nn.Linear(128+64, 32),
                nn.ReLU(inplace= True),
                nn.Dropout(0.2),
                nn.Linear(32, 1),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        # self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.device = T.device('cpu')
        self.to(self.device)

    def forward(self, depth_map, angle):
        
        batch, timestep, c, h, w = depth_map.size()
        d = depth_map.view(batch * timestep, c, h, w)
        d = self.depth_features(d)
        d = d.view(batch, timestep, -1)

        # h0 = T.zeros(1*2, batch, 16)
        # c0 = T.zeros(1*2, batch, 16)
        # d, _ = self.lstm(d, (h0,c0))

        h0 = T.zeros(1, batch, 128)
        d, _ = self.gru(d, h0)
        d = d[: ,-1, :]
        d = nn.Tanh()(d)

        a = self.angle_features(angle)

        x = T.cat([d,a], dim = 1)
        value = self.critic(x)
        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

