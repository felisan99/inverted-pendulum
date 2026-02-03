import os
import gymnasium as gym
from datetime import datetime
from stable_baselines3 import PPO, SAC, DQN, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from gym_envs.pendulum_env import PendulumEnv
from utils.plotting import plot_monitor_data
import torch

class RLTrainer:
    def __init__(self, agent_type="PPO", xml_file=None, policy="MlpPolicy", agent_kwargs=None, seed=None, render_mode="human", max_steps=None, create_run_dir=True, task="equilibrium"):
        self.agent_type = agent_type.upper()
        self.xml_file = xml_file
        self.policy = policy
        self.agent_kwargs = agent_kwargs or {}
        self.seed = seed
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.create_run_dir = create_run_dir
        self.task = task

        self.base_dir = "results"
        self.run_dir = self._create_run_dir() if create_run_dir else None
        
        self.agents = {
            "PPO": PPO,
            "SAC": SAC,
            "A2C": A2C
        }
    def _get_device(self):
        """
        Devuelve el dispositivo a usar (CPU o GPU).
        """
        if torch.backends.mps.is_available():
            return "mps"  # for macbook
        elif torch.cuda.is_available():
            return "cuda" # for colab
        return "cpu"


    def _create_run_dir(self):
        """
        Crea un nuevo directorio donde contener todos los resultados del entrenamiento.
        """
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        
        existing_runs = [d for d in os.listdir(self.base_dir) if d.startswith("run_")]
        run_numbers = [int(d.split("_")[1]) for d in existing_runs if d.split("_")[1].isdigit()]
        
        next_run = max(run_numbers, default=0) + 1
        run_dir = os.path.join(self.base_dir, f"run_{next_run}")
        os.makedirs(run_dir)
        return run_dir

    def _get_model(self, env, verbose: int = 0):
        """
        Crea y devuelve el modelo de RL según el agente configurado.
        
        Args:
            env (gym.Env): entorno de entrenamiento
            verbose (int): nivel de logging del agente

        Returns:
            BaseAlgorithm: modelo de Stable-Baselines3
        """
        if self.agent_type not in self.agents:
            raise ValueError(f"Agente {self.agent_type} no soportado.")

        agent_cls = self.agents[self.agent_type]
        device = self._get_device()

        return agent_cls(
            self.policy,
            env,
            verbose=verbose,
            tensorboard_log=self.run_dir,
            device=device,
            **self.agent_kwargs,
        )
    
    def _make_env(self, monitor_path=None, for_prediction=False):
        """
        Crea una instancia del entorno con los wrappers necesarios.
        """
        env = PendulumEnv(xml_file=self.xml_file, render_mode=self.render_mode if not for_prediction else "human", max_steps=self.max_steps, task=self.task)

        if self.seed is not None:
            env.reset(seed=self.seed)

        if monitor_path is not None:
            env = Monitor(env, monitor_path)

        return env
    
    def train(self, total_timesteps=20000, eval_freq=5000, n_eval_episodes=5, resume_from=None):
        print(f"Directorio de resultados: {self.run_dir}")
        
        train_env = self._make_env(monitor_path=os.path.join(self.run_dir, "train_monitor.csv"))
        val_env = self._make_env(monitor_path=os.path.join(self.run_dir, "val_monitor.csv"))

        if resume_from is not None:
            model = self.agents[self.agent_type].load(resume_from, env=train_env)
            print(f"Resumiendo entrenamiento desde {resume_from}")
        else:
            model = self._get_model(train_env)
        
        eval_callback = EvalCallback(
            val_env,
            best_model_save_path=self.run_dir,
            log_path=self.run_dir,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            render=False,
        )
        
        print(f"Entrenando {self.agent_type} por {total_timesteps} pasos.")
        model.learn(total_timesteps=total_timesteps, callback=eval_callback)
        
        model.save(os.path.join(self.run_dir, "model_final"))
        self._save_results()

    def _save_results(self):
        plot_monitor_data(
            run_dir=self.run_dir,
            save_dir=self.run_dir,
            monitor_file="train_monitor.csv",
            output_name="learning_curve_train.png",
            is_train=True
        )
        plot_monitor_data(
            run_dir=self.run_dir,
            save_dir=self.run_dir,
            monitor_file="val_monitor.csv",
            output_name="learning_curve_val.png",
            is_train=False
        )

    def predict(self, model_path, episodes=5, deterministic=True, render_mode="human"):
        model = self.agents[self.agent_type].load(model_path)

        env = self._make_env(monitor_path=None, for_prediction=True)

        for ep in range(episodes):
            obs, info = env.reset()
            done = False
            total_reward = 0.0

            while not done:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward

            print(f"Episodio {ep + 1} - Reward total: {total_reward:.2f}")

        env.close()