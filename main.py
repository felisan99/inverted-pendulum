import gym_envs.pendulum_env as pendulum_env
import time as time
import test.random_episode_test as random_episode_test
import agents.ppo_agent

agents.ppo_agent.train(max_timesteps=300, total_timesteps=1000)




