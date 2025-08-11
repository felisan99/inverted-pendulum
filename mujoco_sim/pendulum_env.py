import gymnasium as gym
from pathlib import Path
import mujoco
from gymnasium import spaces
import numpy as np

class PendulumEnv(gym.Env):
    def __init__(self, model_path: str | None = None, render_mode: str = "human", max_steps: int = 1000):
        super().__init__()
        
        # Si no pasa la ruta al modelo se usa la ruta por defecto
        if model_path is None:
            ROOT_DIR = Path(__file__).resolve().parent.parent
            model_path = ROOT_DIR / "models" / "pendulum_model.xml"
        
        # Verifica que el archivo del modelo existe
        model_path = Path(model_path).resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Carga el modelo y los datos
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

        # Define espacio de acciones y observaciones
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)    
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)

        # Variables internas
        self.render_mode = render_mode
        self.viewer = None
        self.max_steps = int(max_steps)
        self.current_step = 0
    
    # HAY QUE VER ACA QUE ES LO QUE QUIERO DEVOLVER COMO OBSERVACION
    def get_observation(self):
        """
        Devuelve el estado actual [theta1, vel1, theta2, vel2]
        theta1: Angulo de la primera articulacion
        vel1: Velocidad de la primera articulacion
        theta2: Angulo de la segunda articulacion
        vel2: Velocidad de la segunda articulacion

        Se usa np.sin y np.cos para representar los angulos sin saltos de 0 a 360
        """
        theta1 = self.data.qpos[0]
        vel1 = self.data.qvel[0]
        theta2 = self.data.qpos[1]
        vel2 = self.data.qvel[1]

        return np.array([np.sin(theta1), np.cos(theta1), vel1, np.sin(theta2), np.cos(theta2), vel2], dtype=np.float32)
    
    def reset(self, *, seed = None, options = None):
        # Por convencion
        super().reset(seed=seed)

        self.data = mujoco.MjData(self.model)
        self.current_step = 0
        obs = self.get_observation()
        return obs, {}
    
    def _compute_reward(self, obs: np.ndarray, action: np.ndarray) -> float:
        # HAY QUE VER ACA QUE ES LO QUE QUIERO DEVOLVER COMO RECOMPENSA
        return 0

    def step(self, action: np.ndarray):
        # Asegura que la accion es valida
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        self.data.ctrl[0] = action[0]
        mujoco.mj_step(self.model, self.data)
        obs = self.get_observation()
        reward = self._compute_reward(obs, action)
        self.current_step += 1

        done = self.current_step >= self.max_steps

        return obs, reward, done, False, {}
    
    def render(self):
        if self.render_mode != "human":
            return

        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data, mode=self.render_mode)
        
        mujoco.viewer.sync(self.viewer, self.data)        

    def close(self):
        if self.viewer is not None:
            try:
                self.viewer.close()
            except mujoco.viewer.ViewerError:
                pass
            self.viewer = None
        