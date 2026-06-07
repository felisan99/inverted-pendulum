"""
Template PID controller for the inverted pendulum GUI monitor.

Interface contract (required by the GUI):
  - Class named Controller
  - reset() -> None        called once before control starts
  - compute(...) -> int    called at ~1 kHz, returns PWM in [-1023, +1023]

Encoder conventions (same as PendulumSim / ESP32 firmware):
  pend_enc  : 0-4095  (AS5600 12-bit absolute)
              0    = upright (unstable equilibrium)
              2048 = hanging down
  motor_enc : cumulative Hall counts (signed, unbounded)
              1716 counts/rev at output shaft
  t_us      : simulation time in microseconds
"""

import math

_PEND_RAD_PER_COUNT = 2 * math.pi / 4096   # radians per AS5600 count
_PWM_MAX            = 1023


class Controller:

    # PID gains — tune these for your system
    Kp: float = 150.0
    Ki: float =   8.0
    Kd: float =  18.0

    def reset(self) -> None:
        self._integral  = 0.0
        self._prev_error = 0.0
        self._prev_t_us: int | None = None

    def compute(self, pend_enc: int, motor_enc: int, t_us: int) -> int:
        # Convert pend_enc to angle in radians, wrapped to [-pi, pi].
        # 0 rad = upright, ±pi = hanging.
        angle_rad = pend_enc * _PEND_RAD_PER_COUNT
        if angle_rad > math.pi:
            angle_rad -= 2 * math.pi

        dt = 0.001  # nominal 1 kHz
        if self._prev_t_us is not None:
            dt = max((t_us - self._prev_t_us) * 1e-6, 1e-6)
        self._prev_t_us = t_us

        error             = angle_rad          # setpoint is 0 (upright)
        self._integral   += error * dt
        derivative        = (error - self._prev_error) / dt
        self._prev_error  = error

        pwm = self.Kp * error + self.Ki * self._integral + self.Kd * derivative
        return int(max(-_PWM_MAX, min(_PWM_MAX, pwm)))
