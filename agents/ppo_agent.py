from stable_baselines3 import PPO
from gym_envs.pendulum_env import PendulumEnv
from pathlib import Path

def train(*, max_timesteps = 1_000, total_timesteps = 10_000):
    # Crea el entorno
    env = PendulumEnv(render_mode="human", max_steps=max_timesteps)

    # Crea el agente PPO
    model = PPO("MlpPolicy", env, verbose=1)

    # Entrena el agente
    model.learn(total_timesteps=total_timesteps, progress_bar=True, reset_num_timesteps=False)

    # Guarda el modelo entrenado
    models_dir = Path(__file__).resolve().parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "ppo_model"
    model.save(model_path)

    # Cierra el entorno
    env.close()
