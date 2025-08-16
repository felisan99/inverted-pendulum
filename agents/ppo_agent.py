from stable_baselines3 import PPO
from gym_envs.pendulum_env import PendulumEnv
from pathlib import Path

def train(*, max_timesteps = 1_000, total_timesteps = 10_000, render_window = False):
    # Crea el entorno
    env = PendulumEnv(render_mode= "human" if render_window else None, max_steps=max_timesteps)

    # Crea el agente PPO
    model = PPO("MlpPolicy", env, verbose=1)

    # Entrena el agente    
    model.learn(total_timesteps=total_timesteps, progress_bar=True, reset_num_timesteps=False)        

    # Guarda el modelo entrenado
    ROOT_DIR = Path(__file__).resolve().parent.parent
    models_dir = ROOT_DIR / "models"
    model.save(models_dir / "ppo_agent")

    # Cierra el entorno
    env.close()

def show_model():
    ROOT_DIR = Path(__file__).resolve().parent.parent
    models_dir = ROOT_DIR / "models"
    model = PPO.load(models_dir / "ppo_agent")

    env = PendulumEnv(render_mode="human", max_steps=1000)
    obs, info = env.reset()

    for i in range(1000):
        action, _ = model.predict(obs)
        #if i == 500:
            #env.data.qvel[1] += 1.2
        obs, reward, done, truncated, info = env.step(action)        
    
    env.close()
    
