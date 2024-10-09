from gymnasium.envs.registration import register

register(
    id="airsim_drone-v0", entry_point="rl_env.envs:drone_env", max_episode_steps=100
)