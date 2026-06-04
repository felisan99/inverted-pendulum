"""
PID balance controller — 4 ms sampling variant.

Identical gains to pid_balance.py. The controller decimates itself using t_us:
compute() updates only when >= PERIOD_US have elapsed and returns the last
computed PWM (zero-order hold) on intermediate 1 kHz ticks.

EMA alpha is recalculated for fs = 250 Hz to maintain the same ~26 Hz cutoff:
    alpha = 1 - exp(-2*pi*26/250) ≈ 0.48
"""

import math

_PEND_RAD_PER_COUNT = 2.0 * math.pi / 4096
_ARM_RAD_PER_COUNT  = 2.0 * math.pi / 1716
_PWM_MAX            = 1023
_DIVERGENCE_RAD     = math.radians(25.0)
_PERIOD_US          = 4000


def _pend_to_rad(pend_enc: int) -> float:
    angle = pend_enc * _PEND_RAD_PER_COUNT
    if angle > math.pi:
        angle -= 2.0 * math.pi
    return angle


class Controller:

    Kp    = 20349.7
    Ki    =     0.0
    Kd    =  1570.6
    Ka    = -1018.6
    Kb    =  -289.4
    alpha =    0.48  # EMA cutoff ≈ 26 Hz @ 250 Hz (4 ms period)
    i_max =    0.5

    def reset(self) -> None:
        self._integral        = 0.0
        self._phi_d_filt      = 0.0
        self._omega_arm_filt  = 0.0
        self._prev_phi:        float | None = None
        self._prev_motor_enc:  int   | None = None
        self._last_compute_t:  int   | None = None
        self._last_pwm:        int          = 0

    def compute(self, pend_enc: int, motor_enc: int, t_us: int) -> int:
        phi = _pend_to_rad(pend_enc)

        if abs(phi) > _DIVERGENCE_RAD:
            self.reset()
            return 0

        if self._last_compute_t is None:
            self._prev_phi        = phi
            self._prev_motor_enc  = motor_enc
            self._last_compute_t  = t_us
            return 0

        if t_us - self._last_compute_t < _PERIOD_US:
            return self._last_pwm

        dt = max((t_us - self._last_compute_t) * 1e-6, 1e-6)

        theta_arm = motor_enc * _ARM_RAD_PER_COUNT

        assert self._prev_phi is not None
        assert self._prev_motor_enc is not None

        phi_d_raw        = (phi - self._prev_phi) / dt
        self._phi_d_filt = (self.alpha * phi_d_raw
                            + (1.0 - self.alpha) * self._phi_d_filt)

        omega_arm_raw        = (motor_enc - self._prev_motor_enc) * _ARM_RAD_PER_COUNT / dt
        self._omega_arm_filt = (self.alpha * omega_arm_raw
                                + (1.0 - self.alpha) * self._omega_arm_filt)

        self._integral = max(-self.i_max,
                         min( self.i_max, self._integral + phi * dt))

        self._prev_phi        = phi
        self._prev_motor_enc  = motor_enc
        self._last_compute_t  = t_us

        output = -(self.Kp * phi
                   + self.Ki * self._integral
                   + self.Kd * self._phi_d_filt
                   + self.Ka * theta_arm
                   + self.Kb * self._omega_arm_filt)

        self._last_pwm = int(max(-_PWM_MAX, min(_PWM_MAX, output)))
        return self._last_pwm
