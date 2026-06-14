"""
Close the balance loop on the real rig: load a host-side controller, drive the
ESP32 over serial via `HardwareBackend`, log to CSV.

The controller is unchanged from simulation (default: controllers/balance_SG.py).
Each tick: read the latest SensorReading, call compute(pend_enc, motor_enc, t_us),
send the PWM back. This is the same control path the GUI uses, with PendulumSim
swapped for HardwareBackend.

Usage:
    python scripts/run_hardware_balance.py --port /dev/ttyUSB0 --duration 30

Note on rate: balance_SG's zero-order hold and Savitzky-Golay kernel assume a fixed
sample period (configs/control_config.toml, default 1 kHz). Pace --rate-hz to match
that period, or the estimated velocities are mis-scaled. If the serial round-trip
cannot sustain the rate (see the printed latency stats), lower both the control
config rate and --rate-hz together (e.g. configs/control_250hz.toml + --rate-hz 250).
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
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
    p.add_argument("--csv", type=Path, default=_ROOT / "data" / "hardware_balance.csv")
    args = p.parse_args()

    from gym_envs.hardware_backend import HardwareBackend

    controller = _load_controller(args.controller)
    backend = HardwareBackend(args.port, baudrate=args.baud)

    period   = 1.0 / args.rate_hz
    rows     = []
    latencies = []

    reading = backend.reset()
    controller.reset()
    print(f"Linked. first reading: t_us={reading.t_us} pend={reading.pend_enc} motor={reading.motor_enc}")

    t_start = time.perf_counter()
    next_t  = t_start
    try:
        while time.perf_counter() - t_start < args.duration:
            pwm = controller.compute(reading.pend_enc, reading.motor_enc, reading.t_us)

            t0 = time.perf_counter()
            reading = backend.step(pwm)
            latencies.append((time.perf_counter() - t0) * 1e3)   # ms

            rows.append((reading.t_us, reading.pend_enc, reading.motor_enc, pwm))

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
        w.writerow(["t_us", "pend_enc", "motor_enc", "pwm"])
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


if __name__ == "__main__":
    main()
