import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os

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