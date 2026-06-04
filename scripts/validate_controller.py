"""
Headless validation harness for external controllers (gui_monitor.py plugins).

Runs a Controller from a script file against PendulumSim at 1 kHz with an
initial perturbation from upright, with NO display, and reports whether it
keeps the pendulum balanced. Mirrors the load + step path of gui_monitor.py
(same importlib loader, same reset(initial_angle_rad=...), same
compute(pend_enc, motor_enc, t_us) call), so a PASS here predicts the GUI.

Usage:
    python scripts/validate_controller.py
    python scripts/validate_controller.py --controller controllers/pid_balance.py --perturb-deg 5 --seconds 5
    python scripts/validate_controller.py --gain-scale 0.25
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from gym_envs.pendulum_sim import PendulumSim, PWM_MAX

_PEND_RAD_PER_COUNT = 2 * math.pi / 4096
_ARM_RAD_PER_COUNT  = 2 * math.pi / 1716


def pend_to_rad(pend_enc: int) -> float:
    angle = pend_enc * _PEND_RAD_PER_COUNT
    if angle > math.pi:
        angle -= 2 * math.pi
    return angle


def load_controller_class(path: Path):
    spec = importlib.util.spec_from_file_location("user_controller", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "Controller"):
        raise AttributeError(f"{path} defines no Controller class")
    return mod.Controller


def scale_gains(controller, scale: float) -> list[str]:
    """Multiply every numeric attribute whose name starts with 'K' by scale.

    Matches the gain naming used by the example controllers (Kp, Ki, Kd, Ka,
    Kb, K_PHI, ...) while leaving alpha, i_max, etc. untouched.
    """
    changed = []
    for name in dir(controller):
        if not name.startswith("K"):
            continue
        val = getattr(controller, name)
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            continue
        setattr(controller, name, val * scale)
        changed.append(f"{name}: {val:g} -> {val * scale:g}")
    return changed


def simulate(controller, perturb_rad: float, n_steps: int):
    sim = PendulumSim()
    reading = sim.reset(initial_angle_rad=perturb_rad)
    controller.reset()

    t, phi, arm, pwm_hist = [], [], [], []
    for _ in range(n_steps):
        pwm = int(controller.compute(reading.pend_enc, reading.motor_enc, reading.t_us))
        pwm = max(-PWM_MAX, min(PWM_MAX, pwm))
        reading = sim.step(pwm)
        t.append(reading.t_us * 1e-6)
        phi.append(pend_to_rad(reading.pend_enc))
        arm.append(reading.motor_enc * _ARM_RAD_PER_COUNT)
        pwm_hist.append(pwm)
    sim.close()
    return t, phi, arm, pwm_hist


def direction_probe(perturb_rad: float, k_test: float = 2000.0, ms: int = 150):
    """Plant-sign check, independent of the loaded controller.

    Applies negative feedback pwm = -k_test * phi from an initial +perturbation.
    The example controllers all use u = -(Kp*phi + ...), i.e. negative feedback
    on phi. If the MuJoCo plant sign matches that convention, |phi| shrinks
    toward upright. If |phi| grows, positive PWM drives the pivot the wrong way
    and every controller using that convention will diverge.
    """
    sim = PendulumSim()
    reading = sim.reset(initial_angle_rad=perturb_rad)
    phi0 = pend_to_rad(reading.pend_enc)
    phi = phi0
    for _ in range(ms):
        pwm = int(max(-PWM_MAX, min(PWM_MAX, -k_test * phi)))
        reading = sim.step(pwm)
        phi = pend_to_rad(reading.pend_enc)
    sim.close()
    return phi0, phi


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--controller", default="controllers/pid_balance.py",
                    help="path to a script with a Controller class")
    ap.add_argument("--perturb-deg", type=float, default=5.0,
                    help="initial pendulum angle from upright [deg]")
    ap.add_argument("--seconds", type=float, default=5.0,
                    help="simulated duration [s]")
    ap.add_argument("--gain-scale", type=float, default=None,
                    help="multiply every K* gain by this factor before running")
    args = ap.parse_args()

    path = Path(args.controller)
    if not path.is_absolute():
        path = root / path
    if not path.exists():
        print(f"ERROR: controller not found: {path}")
        return 2

    perturb_rad = math.radians(args.perturb_deg)
    n_steps = int(args.seconds * 1000)

    controller = load_controller_class(path)()
    if args.gain_scale is not None:
        changed = scale_gains(controller, args.gain_scale)
        print(f"Applied gain scale {args.gain_scale}:")
        for line in changed:
            print(f"  {line}")
        print()

    phi0, phi_probe = direction_probe(perturb_rad)
    direction_ok = abs(phi_probe) < abs(phi0)

    t, phi, arm, pwm_hist = simulate(controller, perturb_rad, n_steps)

    phi_deg = [math.degrees(p) for p in phi]
    abs_phi_deg = [abs(p) for p in phi_deg]
    max_abs_phi = max(abs_phi_deg)

    tail_start = t[-1] - 1.0
    tail = [a for ti, a in zip(t, abs_phi_deg) if ti >= tail_start]
    final_abs_phi = sum(tail) / len(tail)

    arm_final_deg = math.degrees(arm[-1])
    arm_max_deg = max(abs(math.degrees(a)) for a in arm)
    sat = sum(1 for p in pwm_hist if abs(p) == PWM_MAX) / len(pwm_hist)

    balanced = (max_abs_phi < 30.0) and (final_abs_phi < 5.0)

    print(f"Controller     : {path}")
    print(f"Perturbation   : {args.perturb_deg:.1f} deg ({perturb_rad:.4f} rad)")
    print(f"Duration       : {args.seconds:.1f} s ({n_steps} steps @ 1 kHz)")
    print("-" * 56)
    print("Direction probe (pwm = -k*phi, plant sign):")
    print(f"  phi {math.degrees(phi0):+.2f} -> {math.degrees(phi_probe):+.2f} deg"
          f"   {'OK (negative feedback stabilizes)' if direction_ok else 'WRONG SIGN (phi grew)'}")
    print("-" * 56)
    print(f"max |phi|          : {max_abs_phi:7.2f} deg")
    print(f"final |phi| (last 1s): {final_abs_phi:7.2f} deg")
    print(f"arm final / max    : {arm_final_deg:+7.1f} / {arm_max_deg:.1f} deg")
    print(f"PWM saturation     : {sat * 100:5.1f} % of steps at +-{PWM_MAX}")
    print("-" * 56)
    verdict = "PASS (balanced)" if balanced else "FAIL (did not balance)"
    print(f"VERDICT: {verdict}")

    return 0 if balanced else 1


if __name__ == "__main__":
    sys.exit(main())
