from stable_baselines3 import PPO
from gym_envs.pendulum_env import PendulumEnv
from pathlib import Path

def ppo_agent_train(*, max_timesteps = 1_000, total_timesteps = 10_000):
    # Crea el entorno
    env = PendulumEnv(render_mode="human", max_steps=1000)

    # Crea el agente PPO
    model = PPO("MlpPolicy", env, verbose=1)

    # Entrena el agente
    model.learn(total_timesteps=10000)

    # Guarda el modelo entrenado
    model.save("ppo_pendulum")

    # Cierra el entorno
    env.close()
