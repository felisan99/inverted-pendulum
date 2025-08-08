import prueba_mujoco
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('Agg')

PLOT = True

if __name__ == "__main__":
    trajectory1, trajectory2 = prueba_mujoco.simulation(PLOT)
    print("Trajectory 1:", trajectory1)
    print("Trajectory 2:", trajectory2)
    
    if PLOT:
        x = np.linspace(0, 10, 500)
        plt.plot(x, trajectory1, label='Joint 1 Position')
        plt.plot(x, trajectory2, label='Joint 2 Position')
        plt.xlabel('Time (s)')
        plt.ylabel('Position (rad)')
        plt.title('Joint Positions Over Time')
        plt.legend()
        plt.savefig("images/trajectory_plot.png")

