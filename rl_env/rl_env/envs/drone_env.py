import gymnasium as gym
from gymnasium import spaces
import time
import numpy as np
import airsim
import math
from PIL import Image
import random

import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

model_configs = {
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
}

encoder = 'vits' # or 'vitb', 'vits'
depth_anything = DepthAnything(model_configs[encoder])
depth_anything.load_state_dict(torch.load(f'./checkpoints/depth_anything_{encoder}14.pth'))

transform = Compose([
    Resize(
        width=518,
        height=518,
        resize_target=False,
        keep_aspect_ratio=True,
        ensure_multiple_of=14,
        resize_method='lower_bound',
        image_interpolation_method=cv2.INTER_CUBIC,
    ),
    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    PrepareForNet(),
])

# map_size = (20, 20)
map_height = 10
image_shape = (84,84)
velocity = 4 # m/s
minimum_flight_height = -5
maximum_flight_height = -8
random_seed = 2024

class drone_env(gym.Env):

    def __init__(self):
        
        self.observation_space = spaces.Dict(
            {
                # "depth_map": spaces.Box(low=0,high=255, shape=(image_shape[0],image_shape[1],1), dtype= float),
                # for pytorch channel first
                "depth_map": spaces.Box(low=0,high=255, shape=(1,image_shape[0],image_shape[1]), dtype= float),
                "position": spaces.Box(low=np.array([-30,-2]),
                                       high=np.array([30,50]),
                                       shape=(2,),
                                       dtype=float),
                'target_location': spaces.Box(low=np.array([-30,-2]),
                                       high=np.array([30,50]),
                                       shape=(2,),
                                       dtype=float),
                #"target_direction": spaces.Box(low = 0, high= 1, shape = (1,), dtype = float)
            }
        )
        
        self.action_space = spaces.Discrete(3, start=0, seed=random_seed)
        
        self.states = {
                "depth_map": np.zeros((image_shape[0],image_shape[1],1), dtype= float),
                "position": np.zeros((3,), dtype= float),
                "target_location": np.zeros((2,), dtype=float)
            }

        # available target locations
        self.available_target_locations = [(0,40)]
        #self.image_request = airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)
        self.image_request = airsim.ImageRequest("0", airsim.ImageType.Scene, True, False)

        self.step_counter = 0
        
        self.drone = airsim.MultirotorClient()
        self.drone.confirmConnection()
        #self._setup_flight()

    def _setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)
        self.drone.takeoffAsync().join()
        orientation = self.drone.simGetVehiclePose().orientation
        pitch, roll, yaw = airsim.utils.to_eularian_angles(orientation)
        self.drone.rotateToYawAsync(90).join()
        self.drone.moveToZAsync(z=-6, velocity=velocity).join()
        self.target_location = random.choice(self.available_target_locations)
        self.states['target_location'] = self.target_location


    def transform_obs(self, responses):
        img1d = np.array(responses[0].image_data_float, dtype= float)
        # this line below makes the difference between close distances bigger, for easier visualization
        img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
        depth = np.reshape(img1d, (responses[0].height, responses[0].width))
        depth =(depth - depth.min()) / (depth.max() - depth.min() + 0.000001) * 255.0
        depth = Image.fromarray(depth)
        depth = np.array(depth.resize((84, 84)).convert("L"), dtype = float)
        depth = depth.reshape([image_shape[0], image_shape[1], 1])
        depth = np.moveaxis(depth,-1,0) # to make channel first
        return depth

    def _get_obs(self):
        self.drone_state = self.drone.getMultirotorState()

        # # depth map
        responses = self.drone.simGetImages([self.image_request])
        try:
            image = self.transform_obs(responses)
        except:
            image = np.zeros(shape=(1,image_shape[0],image_shape[1]))
        h, w = image.shape[:2]

        image = torch.from_numpy(image).unsqueeze(0).to('CPU')
    
        with torch.no_grad():
            depth = depth_anything(image)
        
        depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        image = depth.cpu().numpy().astype(np.uint8)

        self.states['depth_map'] = image

        # Position
        position = self.drone_state.kinematics_estimated.position
        self.states["position"] = (position.x_val, position.y_val)
        
        # Goal direction
        # _, _, yaw  = airsim.utils.to_eularian_angles(self.drone.simGetVehiclePose().orientation)
        # yaw = math.degrees(yaw) 
        # pos_angle = math.atan2(self.target_location[1] - position.y_val, self.target_location[0]- position.x_val)
        # pos_angle = math.degrees(pos_angle) % 360
        # track = math.radians(pos_angle - yaw) 
        # self.direction = ((math.degrees(track) - 180) % 360) - 180 
        
        # self.states['target_direction'] = direction
        # self.states['target_direction'] = self.direction
        
        # Collision
        # collision = self.drone.simGetCollisionInfo().has_collided
        # self.states["collision"] = collision

        return self.states

    def _get_info(self, collision = False):
        
        position = self.states['position']
        target_location = self.states['target_location']

        # Distance
        delta_x = (position[0]- self.target_location[0])**2
        delta_y = (position[1]- self.target_location[1])**2
        self.distance = (delta_x + delta_y)**(1/2)

        _, _ , yaw = airsim.utils.to_eularian_angles(self.drone.simGetVehiclePose().orientation)
        self.heading_angle = math.atan2((target_location[1]-position[1]),(target_location[0]-position[0])) - yaw

        return {
            'position' : position,
            'distance': self.distance,
            'collision': collision,
            'heading_angle': self.heading_angle,
            'total_reward': self.total_reward
            #'direction': self.direction
        }


    def reset(self, seed = None, options = None):
        self.step_counter = 0
        self.total_reward = 0
        self._setup_flight()
        collision = self.drone.simGetCollisionInfo().has_collided
        return self._get_obs(), self._get_info(collision)  #reset needs to return (obs, info)

    def step(self, action):
        position = self.drone.simGetVehiclePose().position
        orientation = self.drone.simGetVehiclePose().orientation
        flight_height = position.z_val

        # keeping the position within range
        if (position.z_val < maximum_flight_height) or (position.z_val > minimum_flight_height):
            flight_height = minimum_flight_height-1
            self.drone.moveToZAsync(z = flight_height, velocity=velocity).join()

        
        pitch, roll, yaw = airsim.utils.to_eularian_angles(orientation)

        vx = math.cos(yaw) * velocity
        vy = math.sin(yaw) * velocity
        
        prev_distance = self.distance

        # Straight
        if action == 0:
            self.drone.moveByVelocityZAsync(vx=vx,
                                            vy=vy,
                                            z= flight_height,
                                            duration=1,
                                            drivetrain=airsim.DrivetrainType.ForwardOnly).join()

        # left
        if action == 1:
            self.drone.rotateByYawRateAsync(yaw_rate= -30, duration=1).join() 

        # right
        if action == 2:
            self.drone.rotateByYawRateAsync(yaw_rate= 30, duration=1).join()

        self.step_counter += 1

        obs = self._get_obs()
        collision = self.drone.simGetCollisionInfo().has_collided
        info = self._get_info(collision)
        
        
        reward, terminated = self._compute_reward(collision, prev_distance)
        self.total_reward = self.total_reward+reward

        return obs, reward, terminated, False, info

    def _compute_reward(self, collision, prev_distance):
        terminated = False
        #Collision penalty
        collision_reward = 0
        if collision:
            collision_reward = -100
            terminated = True
        
        #Target Reached Reward
        target_reached_reward = 0
        if self.distance < 5:
            target_reached_reward = 100
            terminated = True

        # Delay penalty:
        # step_reward = -self.step_counter
        if self.step_counter == 100:
            terminated = True


        # # Distance to Target Reward
        # multiplier = 0.1
        # distance_reward = -self.distance*multiplier

        multiplier = 2
        distance_reward = (prev_distance-self.distance)*multiplier

        # Heading reward
        heading_multiplier = 2
        heading_reward = heading_multiplier*self.heading_angle

        # Total reward
        reward = collision_reward+ target_reached_reward+ distance_reward + heading_reward

        return reward, terminated
    
    def render(self):
        pass
    
    def close(self):
        pass
        
