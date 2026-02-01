import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
colores = ['r', 'g', 'b', 'c', 'm', 'y']

def one_graph_per_observation(observations):
    obs_array = np.array(observations)
    for i in range(obs_array.shape[1]):
        data = obs_array[:, i]
        steps = np.arange(len(data))
        plt.figure(figsize=(8,4))
        plt.plot(steps, data)
        plt.xlabel("Tiempo (Steps)")
        plt.ylabel(f"Columna {i}")
        plt.grid(True)
        plt.savefig(f"images/GRAFICA{i}.png")

def all_observations_in_one_graph(observations):
    plt.figure(figsize=(8,4))
    obs_array = np.array(observations)
    steps = np.arange(len(obs_array))
    for i in range(obs_array.shape[1]):
        plt.plot(steps, obs_array[:, i], label=f"Columna {i}", color=colores[i])

    plt.xlabel("Tiempo (Steps)")
    plt.ylabel("Observaciones")
    plt.legend()
    plt.grid(True)
    plt.savefig("images/GRAFICA_ALL.png")

def plot_monitor_data(run_dir, save_dir, monitor_file, output_name, is_train=True):
    monitor_path = os.path.join(run_dir, monitor_file)
    data = pd.read_csv(monitor_path, skiprows=1)

    steps = data["l"].cumsum()
    rewards = data["r"]

    plt.figure(figsize=(10, 5))
    label = "Reward por episodio (train)" if is_train else "Reward por episodio (val)"
    plt.plot(steps, rewards, label=label)
    plt.xlabel("Steps totales")
    plt.ylabel("Reward")
    plt.title("Curva de Aprendizaje")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, output_name))
    plt.close()