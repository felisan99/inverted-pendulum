import math
import pytest
from gym_envs.pendulum_sim import PendulumSim
from gym_envs.backend import SensorReading

_PENDULUM_LSB = 2 * math.pi / 4096
_MOTOR_LSB    = 2 * math.pi / 1716


@pytest.fixture
def sim():
    s = PendulumSim()
    yield s
    s.close()


def test_bad_path_raises():
    with pytest.raises(FileNotFoundError):
        PendulumSim(xml_file="nonexistent.xml")


def test_reset_returns_sensor_reading(sim):
    r = sim.reset()
    assert isinstance(r, SensorReading)


def test_reset_t_us_is_zero(sim):
    r = sim.reset()
    assert r.t_us == 0


def test_reset_motor_enc_is_zero(sim):
    r = sim.reset()
    assert r.motor_enc == 0


def test_reset_pendulum_down_counts(sim):
    r = sim.reset(pendulum_down=True)
    assert r.pend_enc == 2048


def test_reset_pendulum_up_counts(sim):
    r = sim.reset(pendulum_down=False)
    assert r.pend_enc == 0


def test_step_returns_sensor_reading(sim):
    sim.reset()
    r = sim.step(0)
    assert isinstance(r, SensorReading)


def test_step_t_us_advances_by_1000(sim):
    sim.reset()
    r = sim.step(0)
    assert r.t_us == 1000


def test_step_t_us_accumulates(sim):
    sim.reset()
    for _ in range(50):
        r = sim.step(0)
    assert r.t_us == 50_000


def test_positive_pwm_advances_motor(sim):
    sim.reset()
    for _ in range(200):
        r = sim.step(1023)
    assert r.motor_enc > 0


def test_negative_pwm_reverses_motor(sim):
    sim.reset()
    for _ in range(200):
        r = sim.step(-1023)
    assert r.motor_enc < 0


def test_pwm_clamp_high_does_not_crash(sim):
    sim.reset()
    sim.step(9999)


def test_pwm_clamp_low_does_not_crash(sim):
    sim.reset()
    sim.step(-9999)


def test_pend_enc_bounded(sim):
    sim.reset(pendulum_down=True)
    for _ in range(500):
        r = sim.step(512)
    assert 0 <= r.pend_enc <= 4095


def test_reset_clears_time(sim):
    sim.reset()
    for _ in range(100):
        sim.step(512)
    r = sim.reset()
    assert r.t_us == 0


def test_reset_clears_motor_enc(sim):
    sim.reset()
    for _ in range(200):
        sim.step(1023)
    r = sim.reset()
    assert r.motor_enc == 0
