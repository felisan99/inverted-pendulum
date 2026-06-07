"""
PID balance controller for the Furuta pendulum.

Control law
-----------
    u = -(Kp*phi + Kd*phi_dot_f + Ka*theta_arm + Kb*omega_arm_f)

Four-state feedback is required for this geometry. The pendulum hinge is
ALONG the arm axis, so arm rotation drags the pendulum in the same direction
(coupling ratio ~0.28). A PD on pendulum angle alone fails because:
  1. The arm reaches motor no-load speed (back-EMF cancels torque).
  2. At no-load speed, motor torque is zero and gravity pulls the pendulum.
  3. When the arm decelerates, coupling pushes the pendulum further away.

Ka and Kb are NEGATIVE. They are not a "restoring force" on the arm: the LQR
solution uses the arm position actively, applying torque in the same direction
as the arm displacement, which is the optimal behavior for a Furuta pendulum
without an arm-position integrator. The negative Kb damps the arm velocity.

Gains come from the LQR design (Bryson's method) in the project thesis
(notes/analisis-lqr/), mapped into this control law. They are validated in
simulation against PendulumSim and recover the upright for perturbations up to
roughly +-12 deg (the GUI default is 5 deg); beyond that the PWM saturates and
the pendulum cannot be caught before the divergence guard trips.

Sampling rate and EMA derivative filter
---------------------------------------
The control rate and the filter cutoff are read from configs/control_config.toml
(see gym_envs/control_config.py); the defaults reproduce the original 1 kHz
behaviour. The controller self-decimates: compute() is called every physics tick
(1 kHz) but only updates the output every sample period, holding the last PWM
(zero-order hold) in between, matching how a fixed-rate control ISR runs on the ESP32.

phi_dot and omega_arm are estimated by finite difference and smoothed with EMA to
reject quantization noise:
    y_filt[k] = alpha * y_raw[k] + (1 - alpha) * y_filt[k-1]
    alpha = 1 - exp(-2*pi*fc/fs)
The alpha is derived from the configured fs and fc, so the cutoff stays fixed when
the sampling rate changes (1 kHz/26 Hz -> 0.15, 250 Hz/26 Hz -> 0.48).

Tuning order
------------
  1. Kp, Kd (pendulum PD) — bring phi toward zero for a small perturbation.
  2. Ka (arm position) — gently return arm to zero without large phi excursions.
  3. Kb (arm velocity) — damp arm oscillations without over-braking.
  4. Ki last — only if there is a persistent phi offset.
"""

import math
from pathlib import Path

from gym_envs.control_config import ControlConfig

_PEND_RAD_PER_COUNT = 2.0 * math.pi / 4096
_ARM_RAD_PER_COUNT  = 2.0 * math.pi / 1716
_PWM_MAX            = 1023
_DIVERGENCE_RAD     = math.radians(25.0)
_CONFIG_PATH        = Path(__file__).resolve().parent.parent / "configs" / "control_config.toml"


def _pend_to_rad(pend_enc: int) -> float:
    angle = pend_enc * _PEND_RAD_PER_COUNT
    if angle > math.pi:
        angle -= 2.0 * math.pi
    return angle


class Controller:

    Kp    =  8621.6  # [PWM / rad]         pendulum angle           (LQR K_phi)
    Ki    =     0.0  # [PWM / (rad·s)]     pendulum angle integral
    Kd    =  1295.1  # [PWM / (rad/s)]     pendulum angular velocity (LQR K_phi_dot)
    Ka    =  -254.6  # [PWM / rad]         arm position             (LQR K_theta1, negative)
    Kb    =  -215.5  # [PWM / (rad/s)]     arm angular velocity      (LQR K_theta1_dot, negative)
    i_max =    0.5   # anti-windup clamp [rad·s]

    def __init__(self) -> None:
        cfg = ControlConfig.from_toml(_CONFIG_PATH) if _CONFIG_PATH.exists() else ControlConfig()
        self.alpha      = cfg.ema_alpha       # EMA for phi_dot and omega_arm, derived from fs and fc
        self._period_us = cfg.sample_period_us

    def reset(self) -> None:
        self._integral        = 0.0
        self._phi_d_filt      = 0.0
        self._omega_arm_filt  = 0.0
        self._prev_phi:       float | None = None
        self._prev_motor_enc: int   | None = None
        self._last_compute_t: int   | None = None
        self._last_pwm:       int          = 0

    def compute(self, pend_enc: int, motor_enc: int, t_us: int) -> int:
        phi = _pend_to_rad(pend_enc)

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

        theta_arm = motor_enc * _ARM_RAD_PER_COUNT

        assert self._prev_phi is not None
        assert self._prev_motor_enc is not None

        # Filtered pendulum angular velocity
        phi_d_raw        = (phi - self._prev_phi) / dt
        self._phi_d_filt = (self.alpha * phi_d_raw
                            + (1.0 - self.alpha) * self._phi_d_filt)

        # Filtered arm angular velocity
        omega_arm_raw        = (motor_enc - self._prev_motor_enc) * _ARM_RAD_PER_COUNT / dt
        self._omega_arm_filt = (self.alpha * omega_arm_raw
                                + (1.0 - self.alpha) * self._omega_arm_filt)

        # Integral with anti-windup
        self._integral = max(-self.i_max,
                         min( self.i_max, self._integral + phi * dt))

        self._prev_phi       = phi
        self._prev_motor_enc = motor_enc
        self._last_compute_t = t_us

        output = -(self.Kp * phi
                   + self.Ki * self._integral
                   + self.Kd * self._phi_d_filt
                   + self.Ka * theta_arm
                   + self.Kb * self._omega_arm_filt)

        self._last_pwm = int(max(-_PWM_MAX, min(_PWM_MAX, output)))
        return self._last_pwm
