import gym_envs.pendulum_env as pendulum_env
import time as time
import test.random_episode_test as random_episode_test
import agents.ppo_agent

#agents.ppo_agent.train(max_timesteps=1000, total_timesteps=10000, render_window=True)

env = pendulum_env.PendulumEnv(render_mode="human", max_steps=10000)
env.reset()
while True:
    obs = env.step(0)
    print(obs)




