import numpy as np

from gym_envs.pendulum_sim import PWM_MAX
from gym_envs.policy_controller import ModelController


class FakeModel:
    """Returns a fixed voltage action; records the observations it was asked about."""

    def __init__(self, voltage: float) -> None:
        self.voltage = voltage
        self.seen = []

    def predict(self, obs, deterministic=True):
        self.seen.append(np.asarray(obs, dtype=np.float32).copy())
        return np.array([self.voltage], dtype=np.float32), None


def test_compute_returns_int_pwm_in_range():
    ctrl = ModelController(FakeModel(6.0))
    pwm = ctrl.compute(pend_enc=0, motor_enc=0, t_us=0)
    assert isinstance(pwm, int)
    assert -PWM_MAX <= pwm <= PWM_MAX
    assert pwm == round(6.0 * PWM_MAX / 12.0)


def test_voltage_is_clamped_to_actuator_range():
    assert ModelController(FakeModel(999.0)).compute(0, 0, 0) == PWM_MAX
    assert ModelController(FakeModel(-999.0)).compute(0, 0, 0) == -PWM_MAX


def test_first_compute_has_zero_velocity():
    model = FakeModel(0.0)
    ctrl = ModelController(model)
    ctrl.compute(pend_enc=100, motor_enc=50, t_us=0)
    motor_vel, pend_vel = model.seen[0][2], model.seen[0][5]
    assert motor_vel == 0.0 and pend_vel == 0.0


def test_reset_rearms_the_encoder():
    model = FakeModel(0.0)
    ctrl = ModelController(model)
    ctrl.compute(pend_enc=100, motor_enc=50, t_us=0)
    ctrl.compute(pend_enc=120, motor_enc=60, t_us=1000)
    assert model.seen[1][5] != 0.0  # velocity is non-zero after the first reading

    ctrl.reset()
    model.seen.clear()
    ctrl.compute(pend_enc=200, motor_enc=80, t_us=5000)
    assert model.seen[0][5] == 0.0  # first reading after reset is zero-velocity again
