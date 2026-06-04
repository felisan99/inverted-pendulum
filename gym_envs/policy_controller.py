"""
Deployment harness: a trained SB3 policy driving the pendulum through the same
Controller interface the GUI and hardware use.

A `ModelController` wraps a Stable-Baselines3 model and exposes the controller
contract (`reset()` + `compute(pend_enc, motor_enc, t_us) -> pwm`). Internally it
uses the project's ObservationEncoder, the single source of truth for feature
extraction, so the network sees exactly the observation it was trained on. This is
the concrete "ObservationEncoder + trained network" path described in
gym_envs/CLAUDE.md, with no PendulumEnv around it.
"""

from __future__ import annotations

import numpy as np

from gym_envs.backend import SensorReading
from gym_envs.observation import ObservationEncoder
from gym_envs.pendulum_sim import MAX_VOLTAGE, PWM_MAX


class ModelController:
    def __init__(self, model, deterministic: bool = True) -> None:
        self._model = model
        self._deterministic = deterministic
        self._encoder = ObservationEncoder()
        self._first = True

    def reset(self) -> None:
        self._encoder = ObservationEncoder()
        self._first = True

    def compute(self, pend_enc: int, motor_enc: int, t_us: int) -> int:
        reading = SensorReading(t_us, motor_enc, pend_enc)
        if self._first:
            obs = self._encoder.reset(reading)
            self._first = False
        else:
            obs = self._encoder.update(reading)

        action, _ = self._model.predict(obs, deterministic=self._deterministic)
        voltage = float(np.clip(action, -MAX_VOLTAGE, MAX_VOLTAGE).reshape(-1)[0])
        return int(round(voltage * PWM_MAX / MAX_VOLTAGE))
