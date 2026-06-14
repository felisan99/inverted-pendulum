"""
LQR balance controller for the Furuta pendulum, with Savitzky-Golay velocity
estimation (variant of balance.py).

Same control law and same gains as balance.py; the ONLY change is how the
pendulum and arm angular velocities are estimated. balance.py uses a two-sample
finite difference smoothed by an EMA. Both sensors (AS5600 pendulum, motor
quadrature) are discrete-step encoders, so a two-point derivative at the 1 ms
control period turns one LSB into several rad/s of velocity noise, which the EMA
only partially removes; lowering the EMA cutoff to remove more adds phase lag
that erodes the fast closed-loop pole.

This controller instead fits a low-order polynomial by least squares to a sliding
window of the last N angle samples and reads its slope. A Savitzky-Golay slope is
a derivative estimator matched to a polynomial trend (a p>=1 fit follows a ramp in
angle, i.e. constant velocity, with zero lag), so it rejects sensor noise far
better than a generic low-pass on a noisy difference, for the same or less lag.

Control law (unchanged from balance.py)
---------------------------------------
    u = -(Kp*phi + Ki*integral + Kd*phi_dot + Ka*theta_arm + Kb*omega_arm)

Savitzky-Golay velocity estimation
----------------------------------
For a uniform window of N samples and polynomial order p, the least-squares slope
is a fixed FIR kernel c_i that depends only on N, p and dt (not on the data), so
the estimate is just  phi_dot[k] = sum_i c_i * phi[k-i]. The kernel is built once
at construction; on the ESP32 it becomes a const float[N] and an N-tap
multiply-accumulate over a ring buffer, with no matrix work at runtime.

The window is causal (past samples only), so the estimate carries a group delay:
  - p = 1 (linear fit): delay = (N-1)/2 samples, lowest noise. The slope is the
    velocity at the window centre. With N = 21 at 1 kHz the delay is 10 ms.
  - p >= 2: the polynomial extrapolates curvature, reducing the delay at the cost
    of more noise.
Defaults are N = 21, p = 1: a 10 ms delay traded for maximal noise rejection,
the right corner when the control signal is dominated by sensor noise. The fast
LQR pole is at 37 rad/s (5.9 Hz), where 10 ms is about 21 deg of phase, so
verify stability in simulation before enlarging N.

Sampling rate and zero-order hold (unchanged from balance.py)
-------------------------------------------------------------
The control rate is read from configs/control_config.toml. compute() runs every
physics tick (1 kHz) but only updates every sample period, holding the last PWM
in between (matching a fixed-rate control ISR on the ESP32). The SG window is fed
one sample per control update, so its sample spacing equals the control period.
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


def _savgol_first_deriv(window: int, polyorder: int, dt: float) -> np.ndarray:
    """First-derivative Savitzky-Golay kernel for a causal window.

    The buffer is ordered oldest..newest; the most recent sample sits at t=0, so
    the derivative is evaluated "now". The kernel already includes the 1/dt scale,
    so the dot product with the angle window yields rad/s directly.
    """
    idx = np.arange(window) - (window - 1)        # oldest = -(N-1) ... newest = 0
    t = idx * dt
    A = np.vander(t, polyorder + 1, increasing=True)
    return np.linalg.pinv(A)[1]                    # row 1 = d/dt at t=0


class Controller:

    Kp    =  9938.3  # [PWM / rad]         pendulum angle           (LQR K_phi)
    Ki    =     0.0  # [PWM / (rad·s)]     pendulum angle integral
    Kd    =  1516.3  # [PWM / (rad/s)]     pendulum angular velocity (LQR K_phi_dot)
    Ka    =  -254.6  # [PWM / rad]         arm position             (LQR K_theta1, negative)
    Kb    =  -221.9  # [PWM / (rad/s)]     arm angular velocity      (LQR K_theta1_dot, negative)
    i_max =    0.5   # anti-windup clamp [rad·s]

    SG_WINDOW    = 21  # samples in the Savitzky-Golay window (10 ms group delay at 1 kHz, p=1)
    SG_POLYORDER = 1   # polynomial order of the local least-squares fit

    def __init__(self) -> None:
        cfg = ControlConfig.from_toml(_CONFIG_PATH) if _CONFIG_PATH.exists() else ControlConfig()
        self._period_us = cfg.sample_period_us
        self._sg_coeffs = _savgol_first_deriv(self.SG_WINDOW, self.SG_POLYORDER,
                                              self._period_us * 1e-6)

    def reset(self) -> None:
        self._integral       = 0.0
        self._phi_hist:   deque = deque(maxlen=self.SG_WINDOW)
        self._theta_hist: deque = deque(maxlen=self.SG_WINDOW)
        self._prev_phi:       float | None = None
        self._prev_motor_enc: int   | None = None
        self._last_compute_t: int   | None = None
        self._last_pwm:       int          = 0

    def compute(self, pend_enc: int, motor_enc: int, t_us: int) -> int:
        phi = pend_to_rad(pend_enc)

        if abs(phi) > _DIVERGENCE_RAD:
            self.reset()
            return 0

        if self._last_compute_t is None:
            self._prev_phi       = phi
            self._prev_motor_enc = motor_enc
            self._last_compute_t = t_us
            return 0

        # Zero-order hold: only update at the configured control rate.
        if t_us - self._last_compute_t < self._period_us:
            return self._last_pwm

        dt = max((t_us - self._last_compute_t) * 1e-6, 1e-6)

        theta_arm = motor_enc * MOTOR_LSB

        assert self._prev_phi is not None
        assert self._prev_motor_enc is not None

        self._phi_hist.append(phi)
        self._theta_hist.append(theta_arm)

        if len(self._phi_hist) == self.SG_WINDOW:
            # Savitzky-Golay slope over the full window (FIR dot product).
            phi_d     = float(np.dot(self._sg_coeffs, np.asarray(self._phi_hist)))
            omega_arm = float(np.dot(self._sg_coeffs, np.asarray(self._theta_hist)))
        else:
            # Window still filling: fall back to a plain two-point difference.
            phi_d     = (phi - self._prev_phi) / dt
            omega_arm = (motor_enc - self._prev_motor_enc) * MOTOR_LSB / dt

        # Integral with anti-windup
        self._integral = max(-self.i_max,
                         min( self.i_max, self._integral + phi * dt))

        self._prev_phi       = phi
        self._prev_motor_enc = motor_enc
        self._last_compute_t = t_us

        output = -(self.Kp * phi
                   + self.Ki * self._integral
                   + self.Kd * phi_d
                   + self.Ka * theta_arm
                   + self.Kb * omega_arm)

        self._last_pwm = int(max(-PWM_MAX, min(PWM_MAX, output)))
        return self._last_pwm
