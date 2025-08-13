import gym_envs.pendulum_env as pendulum_env
import time as time
import test.random_episode_test as random_episode_test
import agents.ppo_agent
import numpy as np
import matplotlib.pyplot
matplotlib.use('Agg')

#agents.ppo_agent.train(max_timesteps=1000, total_timesteps=10000, render_window=True)

env = pendulum_env.PendulumEnv(render_mode="human", max_steps=10000)
env.reset()
observations = []
for i in range(1000):
    obs, reward, done, _ = env.step(0)
    observations.append(obs)
    print(obs)
env.close()


obs_array = np.array(observations)
for i in range(6):

    data = obs_array[:, i]
    steps = np.arange(len(data))

    matplotlib.pyplot.figure(figsize=(8,4))
    matplotlib.pyplot.plot(steps, data)
    matplotlib.pyplot.xlabel("Step")
    matplotlib.pyplot.ylabel(f"Columna {i}")

    matplotlib.pyplot.grid(True)
    matplotlib.pyplot.savefig(f"images/GRAFICA{i}.png")

colores = ['r', 'g', 'b', 'c', 'm', 'y']
matplotlib.pyplot.figure(figsize=(8,4))
for i in range(6):
    matplotlib.pyplot.plot(steps, obs_array[:, i], label=f"Columna {i}", color=colores[i])

matplotlib.pyplot.xlabel("Step")
matplotlib.pyplot.ylabel("Observaciones")
matplotlib.pyplot.legend()
matplotlib.pyplot.grid(True)
matplotlib.pyplot.savefig("images/GRAFICA_ALL.png")



