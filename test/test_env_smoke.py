import math

import numpy as np
import pytest

from gym_envs.pendulum_env import PendulumEnv
from gym_envs.backend import SensorReading
from gym_envs.observation import _MOTOR_LSB, _PENDULUM_LSB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reading(t_us: int, motor_rad: float = 0.0, pend_deg: float = 0.0) -> SensorReading:
    pend_enc  = int(round(math.radians(pend_deg) / _PENDULUM_LSB)) % 4096
    motor_enc = int(round(motor_rad / _MOTOR_LSB))
    return SensorReading(t_us, motor_enc, pend_enc)


class FakeBackend:
    """Scripted backend implementing the PendulumBackend Protocol. reset() returns
    the first reading; each step() returns the next, holding the last when exhausted."""

    def __init__(self, readings):
        self._readings = list(readings)
        self._i = 0
        self.closed = False

    def reset(self, *args, **kwargs):
        self._i = 0
        return self._readings[0]

    def step(self, pwm):
        self._i += 1
        return self._readings[min(self._i, len(self._readings) - 1)]

    def close(self):
        self.closed = True


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env_eq():
    env = PendulumEnv(render_mode=None, task="equilibrium")
    yield env
    env.close()


@pytest.fixture
def env_su():
    env = PendulumEnv(render_mode=None, task="swing_up")
    yield env
    env.close()


# ---------------------------------------------------------------------------
# Construction and configuration
# ---------------------------------------------------------------------------

def test_unknown_task_raises():
    with pytest.raises(ValueError):
        PendulumEnv(render_mode=None, task="unknown_task")


def test_model_not_found_raises():
    with pytest.raises(FileNotFoundError):
        PendulumEnv(xml_file="/nonexistent/path/model.xml", render_mode=None)


def test_action_space_bounds(env_eq):
    assert env_eq.action_space.shape == (1,)
    assert env_eq.action_space.low[0] == pytest.approx(-12.0)
    assert env_eq.action_space.high[0] == pytest.approx(12.0)


def test_observation_space_shape(env_eq):
    assert env_eq.observation_space.shape == (6,)


# ---------------------------------------------------------------------------
# Reset behavior
# ---------------------------------------------------------------------------

def test_reset_returns_correct_shape(env_eq):
    obs, info = env_eq.reset()
    assert obs.shape == (6,)
    assert isinstance(info, dict)


def test_reset_obs_dtype(env_eq):
    obs, _ = env_eq.reset()
    assert obs.dtype == np.float32


def test_reset_velocity_is_zero(env_eq):
    obs, _ = env_eq.reset()
    assert obs[2] == pytest.approx(0.0), "motor velocity should be 0 at reset"
    assert obs[5] == pytest.approx(0.0), "pendulum velocity should be 0 at reset"


def test_equilibrium_reset_position_within_offset():
    offset = 0.3
    env = PendulumEnv(render_mode=None, task="equilibrium", starting_offset=offset)
    for seed in range(20):
        obs, _ = env.reset(seed=seed)
        pend_sin, pend_cos = obs[3], obs[4]
        angle = math.atan2(pend_sin, pend_cos)
        assert abs(angle) <= offset + _PENDULUM_LSB, (
            f"seed={seed}: pendulum angle {angle:.4f} outside offset {offset}"
        )
    env.close()


def test_swing_up_reset_pendulum_at_bottom(env_su):
    obs, _ = env_su.reset()
    pend_cos = obs[4]
    # pendulum starts pointing down: qpos[1]=pi → cos(pi)=-1
    assert pend_cos == pytest.approx(-1.0, abs=0.01)


def test_reset_seed_reproducibility(env_eq):
    obs1, _ = env_eq.reset(seed=42)
    obs2, _ = env_eq.reset(seed=42)
    np.testing.assert_array_equal(obs1, obs2)


def test_multiple_resets_clean_state(env_eq):
    env_eq.reset(seed=0)
    for _ in range(200):
        env_eq.step(np.array([12.0], dtype=np.float32))
    obs, _ = env_eq.reset()
    # After a full episode and reset, velocity must be zero again
    assert obs[2] == pytest.approx(0.0)
    assert obs[5] == pytest.approx(0.0)


def test_current_step_resets_to_zero(env_eq):
    env_eq.reset()
    for _ in range(10):
        env_eq.step(env_eq.action_space.sample())
    env_eq.reset()
    assert env_eq.current_step == 0


# ---------------------------------------------------------------------------
# Observation properties
# ---------------------------------------------------------------------------

def test_obs_sincos_unit_circle(env_eq):
    """sin² + cos² must equal 1 for both joints at every step."""
    env_eq.reset(seed=0)
    for _ in range(100):
        obs, _, term, trunc, _ = env_eq.step(env_eq.action_space.sample())
        assert obs[0] ** 2 + obs[1] ** 2 == pytest.approx(1.0, abs=1e-5), \
            "motor sin²+cos² != 1"
        assert obs[3] ** 2 + obs[4] ** 2 == pytest.approx(1.0, abs=1e-5), \
            "pendulum sin²+cos² != 1"
        if term or trunc:
            env_eq.reset(seed=0)


def test_obs_sincos_bounded(env_eq):
    env_eq.reset(seed=1)
    for _ in range(100):
        obs, _, term, trunc, _ = env_eq.step(env_eq.action_space.sample())
        for idx in [0, 1, 3, 4]:
            assert -1.0 <= obs[idx] <= 1.0, f"obs[{idx}]={obs[idx]} out of [-1, 1]"
        if term or trunc:
            env_eq.reset(seed=1)


def test_obs_is_finite(env_eq):
    env_eq.reset(seed=2)
    for _ in range(50):
        obs, _, term, trunc, _ = env_eq.step(env_eq.action_space.sample())
        assert np.all(np.isfinite(obs)), f"non-finite observation: {obs}"
        if term or trunc:
            env_eq.reset(seed=2)


# ---------------------------------------------------------------------------
# Action handling
# ---------------------------------------------------------------------------

def test_action_clipping_positive(env_eq):
    env_eq.reset()
    _, _, _, _, info = env_eq.step(np.array([999.0], dtype=np.float32))
    assert info["voltage"] == pytest.approx(12.0)


def test_action_clipping_negative(env_eq):
    env_eq.reset()
    _, _, _, _, info = env_eq.step(np.array([-999.0], dtype=np.float32))
    assert info["voltage"] == pytest.approx(-12.0)


def test_max_action_maps_to_max_pwm(env_eq):
    env_eq.reset()
    _, _, _, _, info = env_eq.step(np.array([12.0], dtype=np.float32))
    assert info["pwm"] == 1023


# ---------------------------------------------------------------------------
# Step return contract
# ---------------------------------------------------------------------------

def test_step_returns_five_values(env_eq):
    env_eq.reset()
    result = env_eq.step(env_eq.action_space.sample())
    assert len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert obs.shape == (6,)
    assert isinstance(terminated, (bool, np.bool_))
    assert isinstance(truncated, (bool, np.bool_))


def test_current_step_increments(env_eq):
    env_eq.reset()
    for i in range(5):
        env_eq.step(env_eq.action_space.sample())
        assert env_eq.current_step == i + 1


# ---------------------------------------------------------------------------
# Termination and truncation (driven through a scripted backend)
# ---------------------------------------------------------------------------

def test_equilibrium_terminates_when_fallen():
    backend = FakeBackend([_reading(0, pend_deg=0.0), _reading(1000, pend_deg=100.0)])
    env = PendulumEnv(render_mode=None, task="equilibrium", backend=backend)
    env.reset()
    _, _, terminated, _, _ = env.step(np.array([0.0], dtype=np.float32))
    assert terminated, "equilibrium should terminate when pendulum falls past 90°"


def test_equilibrium_no_termination_when_upright():
    backend = FakeBackend([_reading(0, pend_deg=0.0), _reading(1000, pend_deg=0.0)])
    env = PendulumEnv(render_mode=None, task="equilibrium", backend=backend)
    env.reset()
    _, _, terminated, _, _ = env.step(np.array([0.0], dtype=np.float32))
    assert not terminated


def test_swing_up_never_terminates_early(env_su):
    env_su.reset(seed=0)
    for _ in range(200):
        _, _, terminated, _, _ = env_su.step(env_su.action_space.sample())
        assert not terminated, "swing_up should never terminate early"


def test_truncation_at_max_steps():
    env = PendulumEnv(render_mode=None, max_steps=5)
    env.reset()
    results = [env.step(env.action_space.sample()) for _ in range(5)]
    truncated_flags = [r[3] for r in results]
    assert not any(truncated_flags[:-1]), "should not truncate before max_steps"
    assert truncated_flags[-1], "should truncate at max_steps"
    env.close()


# ---------------------------------------------------------------------------
# Reward properties
# ---------------------------------------------------------------------------

def test_equilibrium_reward_at_perfect_upright():
    env = PendulumEnv(render_mode=None, task="equilibrium")
    obs_perfect = np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    reward = env._compute_reward(obs_perfect, np.array([0.0]))
    assert reward == pytest.approx(0.0), "reward must be 0 at perfect upright with no action"
    env.close()


def test_equilibrium_reward_is_nonpositive(env_eq):
    """Equilibrium reward is a pure cost: always <= 0."""
    env_eq.reset(seed=0)
    for _ in range(100):
        obs, reward, term, trunc, _ = env_eq.step(env_eq.action_space.sample())
        assert reward <= 0.0, f"expected reward <= 0, got {reward}"
        if term or trunc:
            env_eq.reset(seed=0)


def test_equilibrium_reward_worse_when_fallen(env_eq):
    obs_up   = np.array([0.0, 1.0, 0.0, 0.0,  1.0, 0.0], dtype=np.float32)
    obs_down = np.array([0.0, 1.0, 0.0, 0.0, -1.0, 0.0], dtype=np.float32)
    action   = np.array([0.0])
    r_up   = env_eq._compute_reward(obs_up, action)
    r_down = env_eq._compute_reward(obs_down, action)
    assert r_up > r_down


def test_swing_up_reward_at_perfect_upright():
    env = PendulumEnv(render_mode=None, task="swing_up")
    obs_up = np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    reward = env._compute_reward(obs_up, np.array([0.0]))
    # target_reward=1, upward_reward=1, no penalties → 10*1 + 2*1 = 12
    assert reward == pytest.approx(12.0)
    env.close()


def test_swing_up_reward_higher_when_upright(env_su):
    obs_up   = np.array([0.0, 1.0, 0.0, 0.0,  1.0, 0.0], dtype=np.float32)
    obs_down = np.array([0.0, 1.0, 0.0, 0.0, -1.0, 0.0], dtype=np.float32)
    action   = np.array([0.0])
    assert env_su._compute_reward(obs_up, action) > env_su._compute_reward(obs_down, action)


# ---------------------------------------------------------------------------
# Info dict
# ---------------------------------------------------------------------------

def test_info_keys_present(env_eq):
    env_eq.reset()
    _, _, _, _, info = env_eq.step(env_eq.action_space.sample())
    assert {"voltage", "pwm", "rpm_motor", "terminated"} <= info.keys()


def test_info_values_finite(env_eq):
    env_eq.reset(seed=0)
    for _ in range(20):
        _, _, term, trunc, info = env_eq.step(env_eq.action_space.sample())
        assert np.isfinite(info["voltage"])
        assert np.isfinite(info["rpm_motor"])
        if term or trunc:
            env_eq.reset(seed=0)


def test_info_rpm_matches_obs_motor_vel(env_eq):
    env_eq.reset()
    for _ in range(30):
        obs, _, term, _, info = env_eq.step(np.array([12.0], dtype=np.float32))
        expected_rpm = (obs[2] * 60) / (2 * math.pi)
        assert info["rpm_motor"] == pytest.approx(expected_rpm, rel=1e-5)
        if term:
            break


# ---------------------------------------------------------------------------
# Sensor resolution
# ---------------------------------------------------------------------------

def test_pendulum_lsb_finer_than_motor_lsb():
    """AS5600 (12-bit) should have finer resolution than Hall encoder (1716 steps)."""
    assert _PENDULUM_LSB < _MOTOR_LSB
