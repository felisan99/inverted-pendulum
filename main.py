import gym_envs.pendulum_env as pendulum_env
import time as time
import test.random_episode_test as random_episode_test
import agents.ppo_agent
import utils.plotting as plotting

#agents.ppo_agent.train(max_timesteps=1000, total_timesteps=10000, render_window=True)

env = pendulum_env.PendulumEnv(render_mode="human", max_steps=10000)
env.reset()
observations = []
for i in range(1000):
    obs, reward, done, _ = env.step(0)
    observations.append(obs)
    print(obs)
env.close()

plotting.one_graph_per_observation(observations)



