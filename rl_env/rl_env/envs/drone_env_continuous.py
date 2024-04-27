import gymnasium as gym
from gymnasium import spaces
import time
import numpy as np
import airsim
import math
from PIL import Image
import random

map_size = (2000, 2000)
map_height = 10
image_shape = (84,84)
velocity = 10 # cm/s

class drone_env(gym.Env):

    def __init__(self):
        
        self.observation_space = spaces.Dict(
            {
                "position": spaces.Box(low=np.array([0,0,0]),
                                       high=np.array([map_size[0],map_size[1],map_height]),
                                       shape=(3,),
                                       dtype=float),
                "depth_map": spaces.Box(low=0,high=255, shape=(image_shape[0],image_shape[1],1), dtype= float),
                "collision": spaces.Discrete(2),
                "target_direction": spaces.Box(low = 0, high= 1, shape = (1,), dtype = float)
            }
        )
        self.action_space = spaces.Box(low= 0, high= velocity, shape=(3,), dtype= float)
        self.states = {
                "position": np.zeros((3,), dtype= float),
                "depth_map": np.zeros((image_shape[0],image_shape[1],1), dtype= float),
                "collision": False,
                "target_direction": 0.0
            }

        # available target locations
        self.available_target_locations = [(950,950),(1900,1900)]
        self.image_request = airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)
        
        self.drone = airsim.MultirotorClient()
        self.drone.confirmConnection()
        #self._setup_flight()

    def _setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)
        self.drone.takeoffAsync().join()
        self.target_location = random.choice(self.available_target_locations)


    def transform_obs(self, responses):
        img1d = np.array(responses[0].image_data_float, dtype= float)
        # this line below makes the difference between close distances bigger, for easier visualization
        img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
        depth = np.reshape(img1d, (responses[0].height, responses[0].width))
        depth =(depth - depth.min()) / (depth.max() - depth.min() + 0.000001) * 255.0
        depth = Image.fromarray(depth)
        depth = np.array(depth.resize((84, 84)).convert("L"), dtype = float)
        depth = depth.reshape([image_shape[0], image_shape[1], 1])
        return depth

    def _get_obs(self):
        self.drone_state = self.drone.getMultirotorState()

        # depth map
        responses = self.drone.simGetImages([self.image_request])
        image = self.transform_obs(responses)
        self.states['depth_map'] = image

        # Position
        position = self.drone_state.kinematics_estimated.position
        self.states["position"] = np.array((position.x_val, position.y_val, -position.z_val), dtype=float)
        
        # direction
        delta_x = (position.x_val- self.target_location[0])**2
        delta_y = (position.y_val- self.target_location[1])**2
        self.distance = (delta_x + delta_y)**(1/2)
        direction = math.atan2(delta_y, delta_x)
        
        # self.states['target_direction'] = direction
        self.states['target_direction'] = 0.1

        # Collision
        collision = self.drone.simGetCollisionInfo().has_collided
        self.states["collision"] = collision

        return self.states

    def _get_info(self):
        return {
            'distance': self.distance,
        }


    def reset(self, seed = None, options = None):
        self._setup_flight()
        return self._get_obs(), self._get_info()  #reset needs to return (obs, info)

    def step(self, action):
        self.drone.moveByVelocityAsync(
            action[0],
            action[1],
            action[2],
            5,
        ).join()
        obs = self._get_obs()
        reward, terminated = self._compute_reward()
        info = self._get_info()
        return obs, reward, terminated, False, info

    def _compute_reward(self):
        terminated = False
        #Collision Reward
        collision_reward = 0
        if self.states['collision']:
            collision_reward = -10
        
        #Target Reached Reward
        target_reached_reward = 0
        if self.distance < 10:
            target_reached_reward = 100
            terminated = True


        # Distance to Target Reward
        multiplier = 1
        distance_reward = -self.distance*multiplier

        # Total reward
        reward = collision_reward+ target_reached_reward+ distance_reward

        return reward, terminated
    
    def render(self):
        pass
    
    def close(self):
        pass
        
