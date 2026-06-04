import importlib.util
import math
from pathlib import Path

import pytest

from gym_envs.pendulum_sim import PendulumSim

_PEND_RAD_PER_COUNT = 2 * math.pi / 4096
_CONTROLLER_PATH = Path(__file__).resolve().parent.parent / "controllers" / "pid_balance.py"


def _pend_to_rad(pend_enc: int) -> float:
    angle = pend_enc * _PEND_RAD_PER_COUNT
    return angle - 2 * math.pi if angle > math.pi else angle


def _load_controller_class():
    if not _CONTROLLER_PATH.exists():
        pytest.skip(f"{_CONTROLLER_PATH} not present (untracked controller)")
    spec = importlib.util.spec_from_file_location("pid_balance_under_test", _CONTROLLER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Controller


def _simulate(controller, perturb_deg: float, seconds: float):
    """Run the controller against PendulumSim, mirroring gui_monitor's loop."""
    sim = PendulumSim()
    try:
        reading = sim.reset(initial_angle_rad=math.radians(perturb_deg))
        controller.reset()
        max_abs_deg = 0.0
        tail = []
        for _ in range(int(seconds * 1000)):
            pwm = int(controller.compute(reading.pend_enc, reading.motor_enc, reading.t_us))
            reading = sim.step(pwm)
            abs_deg = abs(math.degrees(_pend_to_rad(reading.pend_enc)))
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
