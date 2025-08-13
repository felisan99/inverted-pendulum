import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
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