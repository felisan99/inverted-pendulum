import time
import mujoco_sim.simulation as simulation
import gym_envs.pendulum_env as pendulum_env
from gymnasium import spaces

def run_random_agent(env: pendulum_env.PendulumEnv):
    for episode in range(5):
        print(f"Starting new episode {episode + 1}")
        obs, info = env.reset()
        for step in range(500):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            env.render()
            time.sleep(0.01)
        
        print("Episode has finished...")
        time.sleep(0.5)

    env.close()