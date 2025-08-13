import gym_envs.pendulum_env as pendulum_env
import time as time
import test.random_episode_test as random_episode_test
import agents.ppo_agent
import utils.plotting as plotting

#agents.ppo_agent.train(max_timesteps=1_000, total_timesteps=1_000_000, render_window=False)
agents.ppo_agent.show_model()

#env = pendulum_env.PendulumEnv(render_mode="human", max_steps=10000)
#env.reset()
#while True:
#    env.step(1)
# observations = []
# for i in range(1000):
#     obs, reward, done, _ = env.step(1)
#     observations.append(obs)
#     print(obs)
# env.close()

# plotting.one_graph_per_observation(observations)






