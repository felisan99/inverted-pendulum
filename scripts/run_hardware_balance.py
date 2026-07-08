"""
Close the balance loop on the real rig: load a host-side controller, drive the
ESP32 over serial via `HardwareBackend`, log to CSV.

The controller is unchanged from simulation (default: controllers/balance_SG.py).
Each tick: read the latest SensorReading, call compute(pend_enc, motor_enc, t_us),
send the PWM back. This is the same control path the GUI uses, with PendulumSim
swapped for HardwareBackend.

Usage:
    python scripts/run_hardware_balance.py --port /dev/ttyUSB0 --duration 30

Note on rate: balance_SG estimates velocity by least squares over the REAL sample
timestamps and decimates to the control rate in configs/control_config.toml (250 Hz
for the serial link). The internal zero-order hold means the host should poll FASTER
than the control rate so each control update uses a fresh snapshot; let --rate-hz sit
at or above the serial round-trip ceiling (~600 Hz at 1.6 ms) and the loop self-limits.
Do NOT pace --rate-hz down to the control rate. Check the printed latency stats.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
_DEFAULT_CONTROLLER = _ROOT / "controllers" / "balance_SG.py"


def _load_controller(path: Path):
    spec = importlib.util.spec_from_file_location("host_controller", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load controller from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "Controller"):
        raise AttributeError(f"{path} defines no Controller class")
    return mod.Controller()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", required=True, help="serial port, e.g. /dev/ttyUSB0 or COM5")
    p.add_argument("--controller", type=Path, default=_DEFAULT_CONTROLLER)
    p.add_argument("--baud", type=int, default=921600)
    p.add_argument("--duration", type=float, default=30.0, help="seconds")
    p.add_argument("--rate-hz", type=float, default=1000.0, help="host loop pacing")
    p.add_argument("--max-pwm", type=int, default=1023, help="safety clamp on |PWM| (bring-up)")
    p.add_argument("--arm-deg", type=float, default=2.0,
                   help="stay disarmed (PWM=0) until |pendulum angle| exceeds this; "
                        "arms on release so the arm starts from rest (0 disables gating)")
    p.add_argument("--gain-scale", type=float, default=1.0,
                   help="uniformly scale all LQR gains (Kp,Ki,Kd,Ka,Kb)")
    p.add_argument("--kd-scale", type=float, default=1.0,
                   help="additional multiplier applied only to Kd, on top of --gain-scale. "
                        "Effective Kd = Kd_nominal * gain-scale * kd-scale.")
    p.add_argument("--ka-scale", type=float, default=1.0,
                   help="additional multiplier applied only to Ka, on top of --gain-scale. "
                        "Set to 0 to disable arm-position control (safe with a slipring). "
                        "Effective Ka = Ka_nominal * gain-scale * ka-scale.")
    p.add_argument("--kb-scale", type=float, default=1.0,
                   help="additional multiplier applied only to Kb, on top of --gain-scale. "
                        "Use to reduce arm-velocity oscillations independently. "
                        "Effective Kb = Kb_nominal * gain-scale * kb-scale.")
    p.add_argument("--no-arm-control", action="store_true",
                   help="zero Ka only — removes arm-position penalty so the arm can drift freely "
                        "(slipring use case). Kb is kept for dynamic stability.")
    p.add_argument("--pend-bias-deg", type=float, default=0.0,
                   help="software correction for PEND_ZERO_OFFSET error (degrees). "
                        "If the pendulum drifts clockwise, the controller sees a positive "
                        "offset; pass a negative value here to cancel it (e.g. --pend-bias-deg -1.0). "
                        "Converts to counts: counts = round(deg * 4096 / 360). "
                        "Use read_sensors.py to measure the true upright count, then fix "
                        "PEND_ZERO_OFFSET in pins.h for a permanent correction.")
    p.add_argument("--csv", type=Path, default=_ROOT / "data" / "hardware_balance.csv")
    args = p.parse_args()

    import math

    from gym_envs.hardware_backend import HardwareBackend
    from gym_envs.observation import pend_to_rad

    arm_threshold_rad = math.radians(args.arm_deg)
    pend_bias_counts  = round(args.pend_bias_deg * 4096 / 360)
    if pend_bias_counts:
        print(f"pend-bias: {args.pend_bias_deg:+.2f}° = {pend_bias_counts:+d} counts")

    controller = _load_controller(args.controller)
    scaled = []
    for g in ("Kp", "Ki", "Kd", "Ka", "Kb"):
        if not hasattr(controller, g):
            continue
        factor = args.gain_scale
        if g == "Kd":
            factor *= args.kd_scale
        if g == "Ka":
            factor *= args.ka_scale
        if g == "Kb":
            factor *= args.kb_scale
        if factor != 1.0:
            setattr(controller, g, getattr(controller, g) * factor)
        scaled.append(f"{g}={getattr(controller, g):.1f}")
    if args.no_arm_control:
        if hasattr(controller, "Ka"):
            setattr(controller, "Ka", 0.0)
    scaled = [f"{g}={getattr(controller, g):.1f}" for g in ("Kp", "Ki", "Kd", "Ka", "Kb")
              if hasattr(controller, g)]
    if args.gain_scale != 1.0 or args.kd_scale != 1.0 or args.kb_scale != 1.0 or args.no_arm_control:
        print(f"gains: {', '.join(scaled)}")
    backend = HardwareBackend(args.port, baudrate=args.baud)

    period   = 1.0 / args.rate_hz
    rows     = []
    latencies = []

    reading = backend.reset()
    controller.reset()
    print(f"Linked. first reading: t_us={reading.t_us} pend={reading.pend_enc} motor={reading.motor_enc}")

    armed = args.arm_deg <= 0.0
    print("DISARMED: holding PWM=0 until release "
          f"(|angle| > {args.arm_deg}deg). Hold near vertical, then let go." if not armed else "")

    t_start = time.perf_counter()
    next_t  = t_start
    try:
        while time.perf_counter() - t_start < args.duration:
            # Keep the controller warm (fills the SG window) even while disarmed,
            # but do not drive the motor until the pendulum is released. Driving
            # while held lets the LQR arm sub-loop wind up (it is unstable when the
            # pendulum is pinned), throwing the pole on release.
            pend_enc_adj = (reading.pend_enc + pend_bias_counts) % 4096
            pwm = controller.compute(pend_enc_adj, reading.motor_enc, reading.t_us)
            if not armed and abs(pend_to_rad(pend_enc_adj)) > arm_threshold_rad:
                armed = True
                print(f"ARMED @ t={reading.t_us*1e-6:.2f}s (released)")
            pwm = max(-args.max_pwm, min(args.max_pwm, pwm)) if armed else 0

            t0 = time.perf_counter()
            reading = backend.step(pwm)
            latencies.append((time.perf_counter() - t0) * 1e3)   # ms

            tp = getattr(controller, "term_p", 0.0)
            td = getattr(controller, "term_d", 0.0)
            ta = getattr(controller, "term_a", 0.0)
            tb = getattr(controller, "term_b", 0.0)
            rows.append((reading.t_us, reading.pend_enc, reading.motor_enc, pwm, tp, td, ta, tb))

            next_t += period
            slack = next_t - time.perf_counter()
            if slack > 0:
                time.sleep(slack)
            else:
                next_t = time.perf_counter()
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        backend.close()

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_us", "pend_enc", "motor_enc", "pwm", "term_p", "term_d", "term_a", "term_b"])
        w.writerows(rows)

    print(f"wrote {len(rows)} rows to {args.csv}")
    if latencies:
        latencies.sort()
        n = len(latencies)
        print(
            "step() round-trip ms: "
            f"min={latencies[0]:.2f} "
            f"median={latencies[n // 2]:.2f} "
            f"p99={latencies[min(n - 1, int(n * 0.99))]:.2f} "
            f"max={latencies[-1]:.2f}"
        )
    if rows:
        import statistics as _st
        pwms   = [r[3] for r in rows]
        sat    = sum(abs(x) >= 1020 for x in pwms) / len(pwms)
        signs  = sum(1 for a, b in zip(pwms, pwms[1:]) if (a > 0) != (b > 0) and a != 0 and b != 0)
        span_s = max((rows[-1][0] - rows[0][0]) * 1e-6, 1e-9)
        print(f"PWM: sat={sat*100:.1f}%  sign/s={signs/span_s:.1f}  rms={(_st.fmean(x*x for x in pwms))**0.5:.1f}")
        for col, idx, name in ((4, 4, "term_p(Kp)"), (5, 5, "term_d(Kd)"),
                               (6, 6, "term_a(Ka)"), (7, 7, "term_b(Kb)")):
            vals = [r[idx] for r in rows]
            if any(v != 0.0 for v in vals):
                print(f"  {name}: rms={(_st.fmean(x*x for x in vals))**0.5:.1f}  "
                      f"max={max(abs(x) for x in vals):.1f}")


if __name__ == "__main__":
    main()
