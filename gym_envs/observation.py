"""
Observation encoder: raw encoder counts -> policy feature vector.

This is the single source of truth for how a SensorReading becomes the 6-dim
observation the RL policy consumes. The exact same code runs during training
(inside PendulumEnv) and at deployment on the ESP32 host, so there is no
sim-to-real drift in the feature extraction.

    obs = [sin(motor), cos(motor), vel_motor, sin(pend), cos(pend), vel_pend]

The pendulum angle is mapped to [-pi, pi] centred on the upright, so the finite
difference velocity is continuous near 0 (equilibrium). The discontinuity sits at
+-pi (the hanging position), so swing_up velocity is not supported by this encoder.
"""

import math

import numpy as np

from gym_envs.backend import SensorReading

_PENDULUM_LSB = 2 * math.pi / 4096
_MOTOR_LSB    = 2 * math.pi / 1716


def _pend_to_rad(pend_enc: int) -> float:
    angle = pend_enc * _PENDULUM_LSB
    if angle > math.pi:
        angle -= 2.0 * math.pi
    return angle


class ObservationEncoder:
    """Stateful: keeps the previous reading for finite-difference velocity."""

    def reset(self, reading: SensorReading) -> np.ndarray:
        self._prev_motor = reading.motor_enc * _MOTOR_LSB
        self._prev_pend  = _pend_to_rad(reading.pend_enc)
        self._prev_t_us  = reading.t_us
        return self._obs(self._prev_motor, self._prev_pend, 0.0, 0.0)

    def update(self, reading: SensorReading) -> np.ndarray:
        motor = reading.motor_enc * _MOTOR_LSB
        pend  = _pend_to_rad(reading.pend_enc)
        dt    = max((reading.t_us - self._prev_t_us) * 1e-6, 1e-6)
        motor_vel = (motor - self._prev_motor) / dt
        pend_vel  = (pend  - self._prev_pend)  / dt
        self._prev_motor, self._prev_pend, self._prev_t_us = motor, pend, reading.t_us
        return self._obs(motor, pend, motor_vel, pend_vel)

    @staticmethod
    def _obs(motor: float, pend: float, motor_vel: float, pend_vel: float) -> np.ndarray:
        return np.array([math.sin(motor), math.cos(motor), motor_vel,
                         math.sin(pend),  math.cos(pend),  pend_vel], dtype=np.float32)
