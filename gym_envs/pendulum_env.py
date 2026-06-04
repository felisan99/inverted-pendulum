import gymnasium as gym
from gymnasium import spaces
import numpy as np
from math import exp

from gym_envs.backend import PendulumBackend
from gym_envs.pendulum_sim import PendulumSim, PWM_MAX
from gym_envs.observation import ObservationEncoder
from gym_envs.sim_config import SimConfig


class PendulumEnv(gym.Env):
    MAX_VOLTAGE = 12.0

    def __init__(self, xml_file: str | None = None, render_mode: str = "human", max_steps: int = 2000,
                 task="equilibrium", starting_offset: float = 0.4,
                 sim_config: SimConfig | None = None, seed: int | None = None,
                 backend: PendulumBackend | None = None):
        super().__init__()

        self.task = task
        self.starting_offset = starting_offset

        self._reward_functions = {
            "equilibrium": self._reward_equilibrium,
            "swing_up": self._reward_swing_up
        }

        if self.task not in self._reward_functions:
            raise ValueError(f"Unknown task: {self.task}. Available tasks: {list(self._reward_functions.keys())}")

        self._backend = backend or PendulumSim(xml_file=xml_file, render_mode=render_mode,
                                               sim_config=sim_config, seed=seed)
        self._encoder = ObservationEncoder()

        self.action_space = spaces.Box(low=-self.MAX_VOLTAGE, high=self.MAX_VOLTAGE, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        self.render_mode = render_mode
        self.max_steps = int(max_steps)
        self.current_step = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        if self.task == "swing_up":
            reading = self._backend.reset(pendulum_down=True, seed=seed)
        else:
            random_angle = self.np_random.uniform(low=-self.starting_offset, high=self.starting_offset)
            reading = self._backend.reset(initial_angle_rad=random_angle, seed=seed)

        obs = self._encoder.reset(reading)
        return obs, {}

    def _compute_reward(self, obs: np.ndarray, action: np.ndarray) -> float:
        reward_function = self._reward_functions[self.task]
        return reward_function(obs, action)

    def _reward_equilibrium(self, obs: np.ndarray, action: np.ndarray) -> float:
        motor_pos_sin, motor_pos_cos, motor_vel, pend_pos_sin, pend_pos_cos, pend_vel = obs

        theta_error = (1 - pend_pos_cos)
        pend_vel_penalty = pend_vel ** 2
        motor_vel_penalty = motor_vel ** 2
        effort_penalty = np.sum(action ** 2)

        w_theta = 5.0
        w_pend_vel = 0.1
        w_motor_vel = 0.001
        w_effort = 0.001

        reward = -(w_theta * theta_error +
                   w_pend_vel * pend_vel_penalty +
                   w_motor_vel * motor_vel_penalty +
                   w_effort * effort_penalty)

        return reward

    def _reward_swing_up(self, obs: np.ndarray, action: np.ndarray) -> float:
        motor_sin, motor_cos, motor_vel, pend_sin, pend_pos_cos, pend_vel = obs

        target_reward = exp(-(1 - pend_pos_cos)**2 / 0.25)
        upward_reward = (pend_pos_cos + 1) / 2

        if pend_pos_cos > 0.7:
            vel_penalty = 0.1 * (pend_vel ** 2)
        else:
            vel_penalty = 0.001 * (pend_vel ** 2)

        motor_pos_penalty = 0.1 * (motor_sin ** 2)
        effort_penalty = 0.01 * np.sum(action ** 2)

        reward = (10.0 * target_reward +
                2.0 * upward_reward -
                vel_penalty -
                motor_pos_penalty -
                effort_penalty)

        return reward

    def step(self, action: np.ndarray):
        voltage = np.clip(action, -self.MAX_VOLTAGE, self.MAX_VOLTAGE).item()
        pwm = int(round(voltage * PWM_MAX / self.MAX_VOLTAGE))

        reading = self._backend.step(pwm)
        obs = self._encoder.update(reading)
        motor_pos_sin, motor_pos_cos, motor_vel, pend_pos_sin, pend_pos_cos, pend_vel = obs
        reward = float(self._compute_reward(obs, action))
        self.current_step += 1

        done = self.current_step >= self.max_steps
        truncated = self.current_step >= self.max_steps
        if self.task == "equilibrium":
            terminated = bool(pend_pos_cos < 0.0 or abs(pend_vel) > 15.0 or abs(motor_vel) > 50.0)
        elif self.task == "swing_up":
            terminated = False

        info = {
            "voltage": voltage,
            "pwm": pwm,
            "rpm_motor": (motor_vel * 60) / (2 * np.pi),
            "terminated": str(terminated),
        }

        return obs, reward, terminated, truncated, info

    def close(self):
        self._backend.close()
