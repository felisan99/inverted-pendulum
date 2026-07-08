"""
LQR balance controller for the Furuta pendulum with a Savitzky-Golay (order-1)
velocity estimator that uses the real timestamps between samples.

Control law
-----------
    u = -(Kp*phi + Kd*phi_dot + Ka*theta_arm + Kb*omega_arm)

Velocity estimation
-------------------
Velocities are the least-squares slope of the last N (angle, time) pairs:

    slope = sum_i (t_i - t_mean)(y_i - y_mean) / sum_i (t_i - t_mean)^2

Using measured t_us (not a fixed dt) is essential over the serial link: the host
polls the firmware's 1 kHz snapshot with OS/USB jitter, so samples are not
uniformly spaced. A fixed-kernel SG would mis-scale velocity under non-uniform
spacing and saturate the PWM. The real-timestamp slope degrades gracefully and,
when spacing IS uniform (e.g. onboard at 1 kHz), reproduces the fixed SG kernel.

Zero-order hold
---------------
compute() runs every host tick but only updates at the configured control rate
(configs/control_config.toml: sample_freq_hz), holding the last PWM in between.
"""

import math
from collections import deque
from pathlib import Path

import numpy as np

from gym_envs.control_config import ControlConfig
from gym_envs.observation import MOTOR_LSB, pend_to_rad
from gym_envs.pendulum_sim import PWM_MAX

_DIVERGENCE_RAD = math.radians(25.0)
_CONFIG_PATH    = Path(__file__).resolve().parent.parent / "configs" / "control_config.toml"


def _ls_slope(times: np.ndarray, values: np.ndarray) -> float:
    """Order-1 least-squares slope of `values` against `times` (rad/s).

    Returns 0.0 if the timestamps have zero spread (degenerate window).
    """
    t_mean = times.mean()
    dt = times - t_mean
    denom = float(np.dot(dt, dt))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(dt, values - values.mean()) / denom)


class Controller:

    Kp =  9938.3  # [PWM / rad]      pendulum angle           (LQR K_phi)
    Kd =  1516.3  # [PWM / (rad/s)]  pendulum angular velocity (LQR K_phi_dot)
    Ka =  -254.6  # [PWM / rad]      arm position             (LQR K_theta1)
    Kb =  -221.9  # [PWM / (rad/s)]  arm angular velocity      (LQR K_theta1_dot)

    def __init__(self) -> None:
        cfg = ControlConfig.from_toml(_CONFIG_PATH) if _CONFIG_PATH.exists() else ControlConfig()
        self._period_us = cfg.sample_period_us
        self._window    = int(cfg.sg_window)

    def reset(self) -> None:
        self._t_hist:     deque = deque(maxlen=self._window)
        self._phi_hist:   deque = deque(maxlen=self._window)
        self._theta_hist: deque = deque(maxlen=self._window)
        self._last_compute_t: int | None = None
        self._last_pwm:       int        = 0
        self.term_p: float = 0.0
        self.term_d: float = 0.0
        self.term_a: float = 0.0
        self.term_b: float = 0.0

    def compute(self, pend_enc: int, motor_enc: int, t_us: int) -> int:
        phi = pend_to_rad(pend_enc)

        if abs(phi) > _DIVERGENCE_RAD:
            self.reset()
            return 0

        if self._last_compute_t is None:
            self._t_hist.append(t_us * 1e-6)
            self._phi_hist.append(phi)
            self._theta_hist.append(motor_enc * MOTOR_LSB)
            self._last_compute_t = t_us
            return 0

        if t_us - self._last_compute_t < self._period_us:
            return self._last_pwm

        theta_arm = motor_enc * MOTOR_LSB

        self._t_hist.append(t_us * 1e-6)
        self._phi_hist.append(phi)
        self._theta_hist.append(theta_arm)

        if len(self._t_hist) >= 2:
            t_arr     = np.asarray(self._t_hist)
            phi_d     = _ls_slope(t_arr, np.asarray(self._phi_hist))
            omega_arm = _ls_slope(t_arr, np.asarray(self._theta_hist))
        else:
            phi_d = omega_arm = 0.0

        self._last_compute_t = t_us

        self.term_p = -self.Kp * phi
        self.term_d = -self.Kd * phi_d
        self.term_a = -self.Ka * theta_arm
        self.term_b = -self.Kb * omega_arm
        output = self.term_p + self.term_d + self.term_a + self.term_b

        self._last_pwm = int(max(-PWM_MAX, min(PWM_MAX, output)))
        return self._last_pwm
