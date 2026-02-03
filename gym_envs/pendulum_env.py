import gymnasium as gym
from pathlib import Path
import mujoco
from gymnasium import spaces
import numpy as np
import mujoco_viewer
import time
from math import exp

class PendulumEnv(gym.Env):
    # DC Motor Parameters (12V, 1A nominal)
    MAX_VOLTAGE = 12.0
    R = 2.0        # Ohms
    K_T = 0.023    # Nm/A
    K_E = 0.023    # V/(rad/s)

    def __init__(self, xml_file: str | None = None, render_mode: str = "human", max_steps: int = 2000, task="equilibrium"):
        super().__init__()

        self.task = task

        self._reward_functions = {
            "equilibrium": self._reward_equilibrium,
            "swing_up": self._reward_swing_up
        }

        if self.task not in self._reward_functions:
            raise ValueError(f"Unknown task: {self.task}. Available tasks: {list(self._reward_functions.keys())}")
        
        # Si no pasa la ruta al modelo se usa la ruta por defecto
        if xml_file is None:
            ROOT_DIR = Path(__file__).resolve().parent.parent
            xml_file = ROOT_DIR / "mujoco_sim" / "xml_models" / "pendulum_model_v2.xml"
        
        # Verifica que el archivo del modelo existe
        xml_file = Path(xml_file).resolve()
        if not xml_file.exists():
            raise FileNotFoundError(f"Model file not found: {xml_file}")
        
        # Carga el modelo y los datos
        self.xml_file = mujoco.MjModel.from_xml_path(str(xml_file))
        self.data = mujoco.MjData(self.xml_file)

        # Define espacio de acciones (Voltaje continuo)
        self.action_space = spaces.Box(low=-self.MAX_VOLTAGE, high=self.MAX_VOLTAGE, shape=(1,), dtype=np.float32)
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        # Variables internas
        self.render_mode = render_mode
        self.viewer = None
        self.max_steps = int(max_steps)
        self.current_step = 0
        self.render_frequency = 10

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

        return np.array(
            [np.sin(theta_motor), 
            np.cos(theta_motor), 
            vel_motor, 
            np.sin(theta_pendulum), 
            np.cos(theta_pendulum), 
            vel_pendulum], 
            dtype=np.float32)
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        mujoco.mj_resetData(self.xml_file, self.data)

        random_angle = self.np_random.uniform(low=-0.1, high=0.1)
        self.data.qpos[0] = 0.0
        self.data.qpos[1] = random_angle

        mujoco.mj_forward(self.xml_file, self.data)

        return self.get_observation(), {}
    
    def compute_reward(self, obs: np.ndarray, action: np.ndarray) -> float:
        reward_function = self._reward_functions[self.task]
        computed_reward_for_task = reward_function(obs, action)
        return computed_reward_for_task
    
    def _reward_equilibrium(self, obs: np.ndarray, action: np.ndarray) -> float:
        motor_pos_sin, motor_pos_cos, motor_vel, pend_pos_sin, pend_pos_cos, pend_vel = obs
        
        # 1. Error de angulo del pendulo (0 cuando esta vertical hacia arriba)
        # 1 - cos(theta) va de 0 a 2
        theta_error = (1 - pend_pos_cos) 
        
        # 2. Penalizacion por velocidad del pendulo (estabilizacion)
        pend_vel_penalty = pend_vel ** 2

        # 3. Penalizacion por velocidad del motor (evitar giro infinito del brazo)
        motor_vel_penalty = motor_vel ** 2

        # 4. Penalizacion por esfuerzo de control (voltaje)
        # action es un array, tomamos el valor escalar o la norma
        effort_penalty = np.sum(action ** 2)

        # Pesos de la recompensa
        w_theta = 5.0
        w_pend_vel = 0.1
        w_motor_vel = 0.001
        w_effort = 0.001

        # Recompensa negativa (costo)
        reward = -(w_theta * theta_error + 
                   w_pend_vel * pend_vel_penalty + 
                   w_motor_vel * motor_vel_penalty + 
                   w_effort * effort_penalty)
                   
        return reward
    
    def _reward_swing_up(self, obs: np.ndarray, action: np.ndarray) -> float:
        # pend_pos_cos: 1.0 (arriba), -1.0 (abajo)
        motor_sin, motor_cos, motor_vel, pend_sin, pend_pos_cos, pend_vel = obs
        
        # 1. Recompensa por Altura (Exponencial)
        # Usamos una función exponencial para que la recompensa sea muy alta 
        # SOLO cuando está cerca de la vertical.
        target_reward = exp(-(1 - pend_pos_cos)**2 / 0.25)
        
        # 2. Recompensa por Ángulo (Lineal/Cuadrática para el swing-up)
        upward_reward = (pend_pos_cos + 1) / 2
        
        # 3. Penalización de Velocidad del Péndulo
        # Solo penalizamos fuerte la velocidad si ya está en la zona de arriba (> 0.7)
        if pend_pos_cos > 0.7:
            vel_penalty = 0.1 * (pend_vel ** 2)
        else:
            # Abajo permitimos velocidad, pero no infinita (evitar inestabilidad numérica)
            vel_penalty = 0.001 * (pend_vel ** 2)

        # 4. Penalización de posición del brazo (Mantenerlo centrado)
        # Asumiendo que theta_motor = 0 es el centro.
        # Como usamos sin/cos, podemos penalizar que el sin(motor) sea distinto de 0
        motor_pos_penalty = 0.1 * (motor_sin ** 2)

        # 5. Esfuerzo de Control (Voltaje)
        effort_penalty = 0.01 * np.sum(action ** 2)

        # Suma total, los pesos se deberan ajustar
        reward = (10.0 * target_reward + 
                2.0 * upward_reward - 
                vel_penalty - 
                motor_pos_penalty - 
                effort_penalty)
                
        return reward

    def step(self, action: np.ndarray):
        # Aplica voltaje al motor DC simulado
        voltage = np.clip(action, -self.MAX_VOLTAGE, self.MAX_VOLTAGE).item()
        avg_omega = self.data.qvel[0]
        torque = (self.K_T / self.R) * (voltage - self.K_E * avg_omega)
        self.data.ctrl[0] = torque
        
        mujoco.mj_step(self.xml_file, self.data)
        obs = self.get_observation()
        motor_pos_sin, motor_pos_cos, motor_vel, pend_pos_sin, pend_pos_cos, pend_vel = obs
        reward = self.compute_reward(obs, action)
        self.current_step += 1

        if self.render_mode == "human":
            self.render()

        done = self.current_step >= self.max_steps
        
        # Info extra para debugging
        info = {
            "torque": torque,
            "voltage": voltage,
            "rpm_motor": (avg_omega * 60) / (2 * np.pi),
            "terminated reason": ("fallen" if pend_pos_cos < 0.0 else "pendulum velocity" if abs(pend_vel) > 15.0 else "motor velocity" if abs(motor_vel) > 50.0 else "none")
       }
        
        terminated = (pend_pos_cos < 0.0 or abs(pend_vel) > 15.0 or abs(motor_vel) > 50.0)
        truncated = self.current_step >= self.max_steps
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        # Inicializar viewer si no existe
        if self.viewer is None:
            self.viewer = mujoco_viewer.MujocoViewer(self.xml_file, self.data)
            self.last_render_time = time.time()
        
        # Solo actualizar la interfaz gráfica cada N pasos para evitar "slow motion"
        if self.current_step % self.render_frequency == 0:
            self.viewer.render()
            
            # Sincronización para Tiempo Real (Evitar cámara rápida)
            # Tiempo que debió pasar en simulación durante estos N pasos
            sim_time_expected = self.xml_file.opt.timestep * self.render_frequency
            
            # Tiempo que realmente pasó desde el último render
            real_time_passed = time.time() - self.last_render_time
            
            # Si la CPU fue muy rápida, dormimos la diferencia
            if real_time_passed < sim_time_expected:
                time.sleep(sim_time_expected - real_time_passed)
            
            self.last_render_time = time.time()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        