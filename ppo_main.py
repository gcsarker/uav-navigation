import gymnasium as gym
import rl_env
import numpy as np
from ppo_agent import Agent
from torch.utils.tensorboard import SummaryWriter
import os
import datetime

def print_info(info, policy_or_auto_action = 0, mean_depth = 0):

    pos = info['position']
    pos_x = pos[0]
    pos_y = pos[1]

    step_reward = info['step_reward']
    dist = info['distance']

    distance_reward = info['distance_reward']
    heading_reward = info['heading_reward']
    angle = info['angle']


    action_method = ''
    if policy_or_auto_action:
        action_method = 'policy'
    else:
        action_method = 'auto'

    print(
        f"Pos : ({pos_x : .2f},{pos_y : .2f}), "
        f"angle : {angle: .2f}, "
        f"R_dist : {distance_reward: .2f}, "
        f"R_head : {heading_reward: .2f}, "
        f"R/step : {step_reward: .2f}, "
        f"policy/auto: {action_method} ({mean_depth : .4f})"
        )


env = gym.make('airsim_drone-v0')

sample_obs = env.observation_space.sample()

depth_inp_dims = np.array((1,112, 112))

N = 20
batch_size = 4
n_epochs = 5
alpha = 0.0001
agent = Agent(n_actions=env.action_space.n,
              depth_inp_dims=depth_inp_dims,
              batch_size=batch_size,
              alpha=alpha,
              n_epochs=n_epochs,
              load_models=False,
              ent_coef=0.01)
n_games = 200

best_score = 0
score_history = []

learn_iters = 0
avg_score = 0
n_steps = 0

# tensorboard_logging
log_dir = os.path.join("tb_logs", "ppo_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
writer = SummaryWriter(log_dir)

for i in range(n_games):
    observation, info = env.reset()

    # episode beginning info
    spawn = info['spawn_location']
    spawn_x = spawn[0]
    spawn_y = spawn[1]

    t_pos = info['target_location']
    t_pos_x = t_pos[0]
    t_pos_y = t_pos[1]

    print(
        f"Episode : {i+1}, "
        f"spawn at : ({spawn_x : .2f}, {spawn_y : .2f}), "
        f"Dest: ({t_pos_x: .2f}, {t_pos_y : .2f})"
        )
    terminated = False
    truncated = False
    score = 0
    policy_or_auto_action = 0
    
    while (not terminated) and (not truncated):
        depth_map = observation['depth_map']
        angle = observation['angle']
        
        policy_or_auto_action = 1
        action, prob, val = agent.choose_action(depth_map, angle)
        observation_, reward, terminated, truncated, info = env.step(action)
        n_steps += 1
        score += reward

        # comment out this part for inference
        agent.remember(depth_map, angle, action, prob, val, reward, terminated)
        if n_steps % N == 0:
            agent.learn()
            learn_iters += 1
        
        observation = observation_
    
        print_info(info, policy_or_auto_action, np.mean(depth_map))

    
    score_history.append(score)
    avg_score = np.mean(score_history[-30:])

    # tb logging
    writer.add_scalar("score", score, i)
    writer.add_scalar("avg_score", avg_score, i)

    if avg_score > best_score:
        best_score = avg_score
        agent.save_models()

    print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
            'time_steps', n_steps, 'learning_steps', learn_iters)
