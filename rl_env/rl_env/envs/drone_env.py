import gymnasium as gym
from gymnasium import spaces
import numpy as np
import airsim
import math
import random
import os
import cv2
from collections import deque
from scipy.ndimage import zoom

import torch
import torch.nn.functional as F
# from torchvision.transforms import Compose
# from depth_anything.dpt import DepthAnything
# from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet


velocity = 3 # m/s
minimum_flight_height = -5
maximum_flight_height = -8
random_seed = 2024
timestep = 4

scene_dim = (144,256)
depthmap_dim = (144, 256)
d_height, d_width = depthmap_dim

# Cropping Depthmap
cropped_d_height, cropped_d_width = (40, 256)
start_dx = (d_width // 2) - (cropped_d_width // 2)
end_dx = start_dx + cropped_d_width
start_dy = (d_height // 2) - (cropped_d_height // 2)
end_dy = start_dy + cropped_d_height

cropped_resize_height, cropped_resize_width = (112, 112)
angle_multiplier = 1

map_x = 20 - 0.5
map_y = 15/2

# encoder = 'vits'
# depth_anything = DepthAnything({'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]})
# depth_anything.load_state_dict(torch.load(f'./checkpoints/depth_anything_{encoder}14.pth'))

# transform = Compose([
#     Resize(
#         width=252,
#         height=140,
#         resize_target=False,
#         keep_aspect_ratio=True,
#         ensure_multiple_of=14,
#         resize_method='lower_bound',
#         image_interpolation_method=cv2.INTER_CUBIC,
#     ),
#     NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     PrepareForNet(),
# ])

class drone_env(gym.Env):

    def __init__(self):
        
        self.observation_space = spaces.Dict(
            {  
            'depth_map': spaces.Box(low=0, high=1, shape=(timestep, 1, cropped_resize_height, cropped_resize_width), dtype= float),
            'angle' : spaces.Box(low=-math.pi*angle_multiplier, high=math.pi*angle_multiplier, shape=(1,), dtype=float),
            }
        )
        
        self.action_space = spaces.Discrete(3, seed=random_seed)
        
        self.states = {
            'depth_map': np.zeros((timestep, 1, cropped_resize_height, cropped_resize_width), dtype=float), # for recurrent
            'angle': np.zeros([1,], dtype=float),
        }
        
        self.drone = airsim.MultirotorClient()
        self.drone.confirmConnection()
        
        # Image requests to airsim
        self.sceneReq = airsim.ImageRequest(0, airsim.ImageType.Scene, False, False)
        self.depthReq = airsim.ImageRequest(0, airsim.ImageType.DepthPerspective, True, False)

        self.episode_counter = 0
        self.drone.simDestroyObject('locSelector')


    def compute_distance(self, current_position, target_position):
        delta_x = (target_position[0] - current_position[0])**2
        delta_y = (target_position[1] - current_position[1])**2
        return (delta_x + delta_y)**(1/2)
    
    def compute_relative_angle(self, current_pos, target_pos, yaw, mode = 'radian'):
        angle_to_target = math.atan2(target_pos[1] - current_pos[1], target_pos[0]- current_pos[0])

        if mode == 'degree':    
            angle_to_target = math.degrees(angle_to_target)
            yaw = math.degrees(yaw)
        
            relative_angle = angle_to_target - yaw
            while relative_angle >= 180:
                relative_angle -= 360
            while relative_angle < -180:
                relative_angle += 360

        else:
            relative_angle = angle_to_target - yaw
            while relative_angle >= math.pi:
                relative_angle -= 2*math.pi
            while relative_angle < -math.pi:
                relative_angle += 2*math.pi
        
        return relative_angle

    
    def get_random_scale(self, min_scale = 2, max_scale = 2):  # min_scale = 0.2, max_scale = 5 for final simulation
        x_scale = 3.0
        y_scale = random.uniform(min_scale,max_scale)
        z_scale = 10.0
        return airsim.Vector3r(x_val=x_scale, y_val=y_scale, z_val=z_scale), x_scale, y_scale


    def reset(self, seed = None, options = None):
        # for monitoring reward information
        self.R_dist= 0  #reward for moving forward
        self.R_col = 0  # collision penalty
        self.R_heading = 0 # reward for relative heading angle
        self.R_step = 0 # delay penalty
        self.R_target = 0 # reward for target reached
        self.step_counter = 0 # step counting for each episode
        self.total_reward = 0 # total reward for each episode
        self.reward_on_step = 0 # per step reward
        self.R_int_target = 0 # reward for reaching intermediate target
        
        self.episode_counter +=1 
        # self.episode_path = 'episodes/episode_'+str(self.episode_counter)
        # os.makedirs(self.episode_path, exist_ok=True)

        # to hold the depth values
        self.depth_deque = deque(maxlen=timestep)
        for _ in range(timestep):
            self.depth_deque.append(np.zeros((1, cropped_resize_height, cropped_resize_width), dtype=float))

        # Set obstacle scale
        obstacle_pose, obs_scale_x, obs_scale_y = self.get_random_scale(min_scale=1, max_scale=4)
        self.drone.simSetObjectScale('obstacle1', obstacle_pose)

        # set obstacle perimeter
        obstacle_perimeter_pose = airsim.Vector3r(
            x_val=4, y_val=obs_scale_y, z_val=1.0)
        self.drone.simSetObjectScale('ObstaclePerimeter2', obstacle_perimeter_pose)
        
        # Selecting spawn loction
        spawn_side = np.random.choice([-1, 1]) # -1 for left, 1 for right
        self.spawn_location = np.round(
                np.array((
                    0.0,
                    spawn_side*random.uniform(0, map_y-1)
                    ), dtype = float),
                decimals=2
                )
        
        # selecting target location without checking
        target_side = np.random.choice([-1, 1]) # -1 for left, 1 for right
        target_location = np.round(
                np.array((
                    map_x-5,
                    target_side*random.uniform(0, map_y-2)
                    ), dtype = float),
                decimals=2
                )

        # Putting a marker to indicate the target location
        target_marker_pose = airsim.Pose(
            position_val=airsim.Vector3r(x_val=target_location[0], y_val=target_location[1], z_val=0),
            orientation_val=None
            )
        self.drone.simSetObjectPose('locMarker', target_marker_pose)
        self.target_location= np.array((target_location[0]/map_x, target_location[1]/map_y))

        # set intermediate goal location
        int_goal = np.array((6.0, spawn_side*(0.5*obs_scale_y + 2.5)))

        # Putting a marker to indicate the intermediate target location
        int_target_marker_pose = airsim.Pose(
            position_val=airsim.Vector3r(x_val=int_goal[0], y_val= int_goal[1], z_val=0),
            orientation_val=None
            )
        self.drone.simSetObjectPose('locMarker2', int_target_marker_pose)
        self.int_goal = np.array((int_goal[0]/map_x, int_goal[1]/map_y))

        self._setup_flight()

        collision = self.drone.simGetCollisionInfo().has_collided
        obs = self._get_obs()
        info = self._get_info(collision)
        return obs, info 

    def _setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)

        # moving to z value then spawn location otherwise problem occurs
        self.drone.moveToZAsync(z= -7.0, velocity=velocity).join()
        
        pos = airsim.Vector3r(x_val= self.spawn_location[0], y_val= self.spawn_location[1], z_val= -7.0)
        orient = airsim.Quaternionr(w_val= 1.0, x_val= 0.0, y_val= 0.0, z_val= 0)
        pose = airsim.Pose(position_val=pos, orientation_val=orient)
        self.drone.simSetVehiclePose(pose,ignore_collision=True)


    def _get_obs(self):
        if self.step_counter == 0:
            pos = airsim.Vector3r(x_val= self.spawn_location[0], y_val= self.spawn_location[1], z_val= -7.0)
            orient = airsim.Quaternionr(w_val= 1.0, x_val= 0.0, y_val= 0.0, z_val= 0)
            pose = airsim.Pose(position_val=pos, orientation_val=orient)
        else:
            pose = self.drone.simGetVehiclePose()

        # Position          add /map_size        
        self.position = np.array((pose.position.x_val/map_x, pose.position.y_val/map_y), dtype=float)
        
        # distance to intermediate goal
        self.dist_to_int_goal = self.compute_distance(self.position, self.int_goal)
        
        # distance to target
        self.distance = self.compute_distance(self.position, self.target_location)

        #Relative Distance
        relative_x = self.target_location[0]-self.position[0]
        relative_y = self.target_location[1]-self.position[1]
        self.relative_distance = np.array([relative_x,relative_y], dtype=float)
        
        # relative angle to the target
        _, _, yaw  = airsim.utils.to_eularian_angles(pose.orientation) 
        self.angle = self.compute_relative_angle(self.position, self.target_location, yaw, mode = 'radian')
        self.states['angle'] = np.array([self.angle*angle_multiplier], dtype = float)


        # # depth_map estimation
        # scene_response = self.drone.simGetImages([self.sceneReq])[0]
        # scene = np.frombuffer(scene_response.image_data_uint8, dtype=np.uint8)
        # scene = scene.reshape(scene_dim[0], scene_dim[1], 3)/255
        # scene = transform({'image': scene})['image']
        # scene = torch.from_numpy(scene).unsqueeze(0)
        # with torch.no_grad():
        #     depth = depth_anything(scene)
        # depth = F.interpolate(depth[None], (depthmap_dim[0], depthmap_dim[1]), mode='bilinear', align_corners=False)[0, 0]
        # depth = depth.numpy().astype('float64')

        #Taking depth from airsim
        depth_response = self.drone.simGetImages([self.depthReq])[0]
        depth = np.array(depth_response.image_data_float, dtype=float)
        depth = 255 / np.maximum(np.ones(depth.size), depth)
        depth = np.reshape(depth, (depth_response.height, depth_response.width))

        # # cropping
        depth = depth[start_dy:end_dy, start_dx:end_dx]

        # resizing
        depth = zoom(depth, (cropped_resize_height/depth.shape[0], cropped_resize_width/depth.shape[1]))
        
        # adding gaussian noise
        # gaussian_noise = np.random.normal(0, 5, (cropped_resize_height, cropped_resize_width)).astype(np.float32)
        # depth = depth + gaussian_noise
        depth = np.clip(depth, 1, 255)
        
        # normalizing depth
        # depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0 
        # depth = (depth - depth.min()) / (depth.max() - depth.min())
        depth = depth/255

        self.depth_deque.append(np.expand_dims(depth, axis=0))
        self.states['depth_map'] = np.array(self.depth_deque)
        
        # saving images
        # cv2.imwrite(self.episode_path+'/'+str(self.step_counter)+'.jpg', depth*255)

        return self.states
    
    def _get_info(self, collision = False):

        return {
            'position' : self.position,
            'spawn_location' : self.spawn_location,
            'target_location': self.target_location,
            'distance': self.distance,
            'int_goal_distance':self.dist_to_int_goal,
            'collision': collision,
            'angle': math.degrees(self.angle),
            'distance_reward' : self.R_dist,
            'heading_reward' : self.R_heading,
            'target_reached_reward': self.R_target,
            'collision_reward' : self.R_col,
            'step_reward': self.reward_on_step,
            'total_reward': self.total_reward,
            'relative_distance': self.relative_distance,
            'step_counter': self.step_counter
        }

    def step(self, action):
        position = self.drone.simGetVehiclePose().position
        orientation = self.drone.simGetVehiclePose().orientation
        flight_height = position.z_val

        # keeping the position within range
        if (position.z_val < maximum_flight_height) or (position.z_val > minimum_flight_height):
            self.drone.moveToZAsync(z = -7.0, velocity=velocity).join()

        
        _, _, yaw = airsim.utils.to_eularian_angles(orientation)

        vx = math.cos(yaw) * velocity
        vy = math.sin(yaw) * velocity
        
        prev_distance = self.distance
        prev_angle = self.angle

        # Straight
        if action == 0:
            self.drone.moveByVelocityZAsync(vx=vx,
                                            vy=vy,
                                            z= flight_height,
                                            duration=1,
                                            drivetrain=airsim.DrivetrainType.ForwardOnly).join()

        # left
        if action == 1:
            self.drone.rotateByYawRateAsync(yaw_rate= -25, duration=1).join() 

        # right
        if action == 2:
            self.drone.rotateByYawRateAsync(yaw_rate= 25, duration=1).join()

        self.step_counter += 1
        obs = self._get_obs()
        collision = self.drone.simGetCollisionInfo()
        reward, terminated, truncated = self._compute_reward(collision, prev_distance, prev_angle, action)
        self.reward_on_step = reward
        self.total_reward = self.total_reward+reward
        info = self._get_info(collision)

        return obs, reward, terminated, truncated, info

    def _compute_reward(self, collision, prev_distance, prev_angle, action):
        terminated = False
        truncated = False

        # Stop after 15 steps
        if self.step_counter == 30:
            truncated = True
        
        # Collision penalty
        collision_reward = 0
        if collision.has_collided:
            if collision.object_name == 'BlockingCube':
                truncated = True
            else:
                collision_reward = -20
                terminated = True
        self.R_col = collision_reward

        # heading reward
        heading_reward = 0
        # heading_reward = (np.abs(prev_angle)- np.abs(self.angle))*10
        self.R_heading = heading_reward

        # step_reward    
        distance_reward = (prev_distance-self.distance)*50
        self.R_dist = distance_reward

        #Target Reached Reward
        target_reached_reward = 0
        if self.distance < 0.3:
            #target_reached_reward = 10
            print('target reached!')
            terminated = True
        self.R_target = target_reached_reward

        # intermediate Target Reached Reward
        int_target_reached_reward = 0
        # if self.dist_to_int_goal < 0.4:
        #     int_target_reached_reward = 10
        #     self.int_goal = np.array((2,2))
        self.R_int_target = int_target_reached_reward

        # Delay penalty:
        delay_penalty = -1
        # self.R_step = delay_penalty

        reward = collision_reward +  heading_reward + distance_reward + target_reached_reward + int_target_reached_reward + delay_penalty
        return reward, terminated, truncated

    
    def render(self):
        pass
    
    def close(self):
        pass
        
