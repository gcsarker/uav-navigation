import airsim
import os
import pygame
import numpy as np
import threading
import cv2
import matplotlib.pyplot as plt
import math
import time
import torch as th

FORWARD = pygame.K_w
BACKWARD = pygame.K_s
LEFT = pygame.K_a
RIGHT = pygame.K_d

keys = [FORWARD, BACKWARD, LEFT, RIGHT]
key_pressed = np.zeros((4,))
sim = True

def pygame_run():
    pygame.init()

    SCREEN_WIDTH = 500
    SCREEN_HEIGHT = 500

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    run = True
    while run:
        pressed = pygame.key.get_pressed()
        
        for idx, key in enumerate(keys):
                key_pressed[idx] = pressed[key]

        #print(key_pressed)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

    sim = False    
    pygame.quit()

key_event_thread = threading.Thread(target=pygame_run)
key_event_thread.start()

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Async methods returns Future. Call join() to wait for task to complete.
client.takeoffAsync().join()
orientation = client.simGetVehiclePose().orientation
pitch, roll, yaw = airsim.utils.to_eularian_angles(orientation)
print(yaw)
client.rotateToYawAsync(90).join()
pose = client.simGetVehiclePose()
z = -6
maximum_flight_height = -8
minimum_flight_height = -5
client.moveToZAsync(z=z, velocity=2).join()


velocity = 2
multiplier = 0.5

step_count = 1
image_request1 = airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)
image_request2 = airsim.ImageRequest(0, airsim.ImageType.Scene, False, False)
from PIL import Image
image_shape = (84,84)

while sim:
    responses = client.simGetImages([image_request1, image_request2])
    img1d = np.array(responses[0].image_data_float, dtype= float)
    # this line below makes the difference between close distances bigger, for easier visualization
    img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
    depth = np.reshape(img1d, (responses[0].height, responses[0].width))
    depth =(depth - depth.min()) / (depth.max() - depth.min() + 0.000001) * 255.0
    cv2.imwrite('temp/depth/depth_'+str(step_count)+'.jpg',depth)


    real = responses[1]
    # get numpy array
    real = np.fromstring(real.image_data_uint8, dtype=np.uint8) 

    # reshape array to 4 channel image array H X W X 4
    real = real.reshape(responses[1].height, responses[1].width, 3)

    # original image is fliped vertically
    real = np.flipud(real)
    cv2.imwrite('temp/real/real_'+str(step_count)+'.jpg',real)


    position = client.simGetVehiclePose().position
    orientation = client.simGetVehiclePose().orientation
    flight_height = position.z_val

    print(f"x: {position.x_val}, y : {position.y_val}")
    # keeping the position within range
    if (position.z_val < maximum_flight_height) or (position.z_val > minimum_flight_height):
        flight_height = minimum_flight_height-1
        client.moveToZAsync(z = flight_height, velocity=velocity).join()


    
    pitch, roll, yaw = airsim.utils.to_eularian_angles(orientation)
    vx = math.cos(yaw) * velocity
    vy = math.sin(yaw) * velocity

    if key_pressed[0]:
        client.moveByVelocityZAsync(vx=vx, vy=vy, z= flight_height, duration=1, drivetrain=airsim.DrivetrainType.ForwardOnly).join()

    elif key_pressed[2]:
        client.rotateByYawRateAsync(yaw_rate= -30, duration=1).join() #left

    elif key_pressed[3]:
        client.rotateByYawRateAsync(yaw_rate= 30, duration=1).join() #right
    else:
        client.moveByVelocityZAsync(vx=vx, vy=vy, z= position.z_val, duration=1, drivetrain=airsim.DrivetrainType.ForwardOnly).join()


    if not key_event_thread.is_alive():
         sim = False
    
    step_count = step_count+1

client.landAsync().join()