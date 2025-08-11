import mujoco_sim.simulation as simulation
import mujoco_sim.pendulum_env as pendulum_env
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time as time

matplotlib.use('Agg')

# PLOT = True

# trajectory1, trajectory2 = simulation.simulation(PLOT)
# print("Trajectory 1:", trajectory1)
# print("Trajectory 2:", trajectory2)
    
# if PLOT:
#     x = np.linspace(0, 10, 500)
#     plt.plot(x, trajectory1, label='Joint 1 Position')
#     plt.plot(x, trajectory2, label='Joint 2 Position')
#     plt.xlabel('Time (s)')
#     plt.ylabel('Position (rad)')
#     plt.title('Joint Positions Over Time')
#     plt.legend()
#     plt.savefig("images/trajectory_plot.png")

# Create an instance of the PendulumEnv
env = pendulum_env.PendulumEnv(render_mode="human", max_steps=1000)
env.render()
# Reset the environment to get the initial observation
obs, _ = env.reset()
print("Initial Observation:", obs)
# Take a step in the environment with a random action
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
print("Observation after step:", obs)

time.sleep(10)  # Wait for 2 seconds to visualize the environment



