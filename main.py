import gym_envs.pendulum_env as pendulum_env
import time as time
import test.random_episode_test as random_episode_test

env = pendulum_env.PendulumEnv(render_mode="human", max_steps=1000)
random_episode_test.run_random_agent(env)



