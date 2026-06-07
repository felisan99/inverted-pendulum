import importlib.util
import math
from pathlib import Path

import pytest

from gym_envs.observation import pend_to_rad
from gym_envs.pendulum_sim import PendulumSim

_CONTROLLER_PATH = Path(__file__).resolve().parent.parent / "controllers" / "pid_balance.py"
_CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def _load_controller_class(config_path: Path | None = None):
    if not _CONTROLLER_PATH.exists():
        pytest.skip(f"{_CONTROLLER_PATH} not present (untracked controller)")
    spec = importlib.util.spec_from_file_location("pid_balance_under_test", _CONTROLLER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Decouple from the repo's active control_config.toml: a non-existent path
    # falls back to ControlConfig() defaults (1 kHz), a real path selects a rate.
    mod._CONFIG_PATH = config_path if config_path is not None else Path("/nonexistent.toml")
    return mod.Controller


def _simulate(controller, perturb_deg: float, seconds: float):
    """Run the controller against PendulumSim, mirroring the GUI's loop."""
    sim = PendulumSim()
    try:
        reading = sim.reset(initial_angle_rad=math.radians(perturb_deg))
        controller.reset()
        max_abs_deg = 0.0
        tail = []
        for _ in range(int(seconds * 1000)):
            pwm = int(controller.compute(reading.pend_enc, reading.motor_enc, reading.t_us))
            reading = sim.step(pwm)
            abs_deg = abs(math.degrees(pend_to_rad(reading.pend_enc)))
            max_abs_deg = max(max_abs_deg, abs_deg)
            if reading.t_us * 1e-6 >= seconds - 1.0:
                tail.append(abs_deg)
    finally:
        sim.close()
    return max_abs_deg, sum(tail) / len(tail)


@pytest.mark.parametrize("perturb_deg", [5.0, 10.0])
def test_pid_balance_stabilizes(perturb_deg):
    controller = _load_controller_class()()
    max_abs_deg, final_abs_deg = _simulate(controller, perturb_deg, seconds=3.0)
    assert max_abs_deg < 15.0, f"pendulum diverged (max |phi| = {max_abs_deg:.1f} deg)"
    assert final_abs_deg < 3.0, f"did not settle upright (final |phi| = {final_abs_deg:.1f} deg)"


def test_pid_balance_stabilizes_at_250hz():
    config_path = _CONFIGS_DIR / "control_250hz.toml"
    controller = _load_controller_class(config_path)()
    assert controller._period_us == 4000.0
    max_abs_deg, final_abs_deg = _simulate(controller, perturb_deg=5.0, seconds=3.0)
    assert max_abs_deg < 15.0, f"pendulum diverged (max |phi| = {max_abs_deg:.1f} deg)"
    assert final_abs_deg < 3.0, f"did not settle upright (final |phi| = {final_abs_deg:.1f} deg)"
