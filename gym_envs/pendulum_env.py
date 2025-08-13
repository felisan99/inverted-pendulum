import gymnasium as gym
from pathlib import Path
import mujoco
from gymnasium import spaces
import numpy as np
import mujoco.viewer
import time

# Paso tipico en motores Nema
STEP_ANGLE = np.deg2rad(1.8)

class PendulumEnv(gym.Env):
    def __init__(self, model_path: str | None = None, render_mode: str = "human", max_steps: int = 1000):
        super().__init__()
        
        # Si no pasa la ruta al modelo se usa la ruta por defecto
        if model_path is None:
            ROOT_DIR = Path(__file__).resolve().parent.parent
            model_path = ROOT_DIR / "mujoco_sim" / "xml_models" / "pendulum_model.xml"
        
        # Verifica que el archivo del modelo existe
        model_path = Path(model_path).resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Carga el modelo y los datos
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

        # Define espacio de acciones y observaciones
        """
            Espacion de acciones mas parecido a un Nema
            0 -> No mover
            1 -> CW
            2 -> CCW
        """
        self.action_space = spaces.Discrete(3)
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        # Variables internas
        self.render_mode = render_mode
        self.viewer = None
        self.max_steps = int(max_steps)
        self.current_step = 0
    
    # HAY QUE VER ACA QUE ES LO QUE QUIERO DEVOLVER COMO OBSERVACION
    def get_observation(self):
        """
        Devuelve el estado actual [theta1, vel1, theta2, vel2]
        theta_motor: Angulo de la primera articulacion
        vel_motor: Velocidad de la primera articulacion
        theta_pendulum: Angulo de la segunda articulacion
        vel_pendulum: Velocidad de la segunda articulacion

        Se usa np.sin y np.cos para representar los angulos sin saltos de 0 a 360
        """
        theta_motor = self.data.qpos[0]
        vel_motor = self.data.qvel[0]
        theta_pendulum = self.data.qpos[1]
        vel_pendulum = self.data.qvel[1]

        return np.array([np.sin(theta_motor), np.cos(theta_motor), vel_motor, np.sin(theta_pendulum), np.cos(theta_pendulum), vel_pendulum], dtype=np.float32)
    
    def reset(self, *, seed = None, options = None):
        # Por convencion
        super().reset(seed=seed)
        self.current_step = 0
        
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0

        # Randomiza la posicion inicial del pendulo
        random_angle = self.np_random.uniform(low=-np.pi, high=np.pi)
        self.data.qpos[0] = 0.0
        self.data.qpos[1] = random_angle

        # Avanzar un paso para que los cambios tengan efecto
        mujoco.mj_step(self.model, self.data)
        
        obs = self.get_observation()
        return obs, {}
    
    def compute_reward(self, obs: np.ndarray, action: np.ndarray) -> float:
        motor_pos_sin, motor_pos_cos, motor_vel, pend_pos_sin, pend_pos_cos, pend_vel = obs
        # Penalizacion por velocidades altas, bueno cercano a cero
        vel_penalty = motor_vel**2 + pend_vel**2
        # Penalizacion por error de posicion del pendulo, bueno cercano a cero
        pos_penalty = 1 - pend_pos_cos
        reward = - (2.0 * pos_penalty + 0.1 * vel_penalty)
        return reward

    def step(self, action: np.ndarray):
        # Asegura que la accion es valida
        match action:
            case 0:
                # No mover
                self.data.ctrl[0] = 0.0
            case 1:
                # Mover en sentido horario
                self.data.ctrl[0] = STEP_ANGLE + self.data.qpos[0]
            case 2:
                # Mover en sentido antihorario
                self.data.ctrl[0] = -STEP_ANGLE + self.data.qpos[0]
        
        mujoco.mj_step(self.model, self.data)
        obs = self.get_observation()
        reward = self.compute_reward(obs, action)
        self.current_step += 1
        self.render()

        done = self.current_step >= self.max_steps

        return obs, reward, done, {}
    
    def render(self):
        if self.render_mode != "human":
            return
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        
        self.viewer.sync()
        time.sleep(0.004)       

    def close(self):
        if self.viewer is not None:
            try:
                self.viewer.close()
            except mujoco.viewer.ViewerError:
                pass
            self.viewer = None
        