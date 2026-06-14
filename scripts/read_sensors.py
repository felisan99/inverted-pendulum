"""
Live sensor readout over the ESP32 serial-slave backend. Bring-up helper.

Drives PWM=0 and prints the raw SensorReading at ~10 Hz, plus a running median of
pend_enc. Use it to:
  - find the upright count: hold the pendulum vertical, read the stable median,
    set that value as PEND_ZERO_OFFSET in the firmware's pins.h (then re-flash).
  - check signs: tilt the pendulum / push the arm and watch pend_enc, motor_enc.

Usage:
    python scripts/read_sensors.py --port /dev/ttyUSB0

Optionally command a small PWM to check motor direction (CAUTION: hold the rig):
    python scripts/read_sensors.py --port /dev/ttyUSB0 --pwm 150
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections import deque


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", required=True, help="serial port, e.g. /dev/ttyUSB0 or COM5")
    p.add_argument("--baud", type=int, default=921600)
    p.add_argument("--pwm", type=int, default=0, help="constant PWM to apply (default 0)")
    p.add_argument("--hz", type=float, default=10.0, help="print rate")
    p.add_argument("--window", type=int, default=30, help="samples for the median")
    args = p.parse_args()

    from gym_envs.hardware_backend import HardwareBackend

    backend = HardwareBackend(args.port, baudrate=args.baud)
    pend_hist: deque[int] = deque(maxlen=args.window)
    period = 1.0 / args.hz

    backend.reset()
    print("Reading sensors (Ctrl-C to stop). Hold the pendulum upright to read its count.\n")
    print(f"{'t_us':>12}  {'pend_enc':>8}  {'pend_median':>11}  {'motor_enc':>9}  {'pwm':>5}")
    try:
        while True:
            r = backend.step(args.pwm)
            pend_hist.append(r.pend_enc)
            median = int(statistics.median(pend_hist))
            print(f"{r.t_us:>12}  {r.pend_enc:>8}  {median:>11}  {r.motor_enc:>9}  {args.pwm:>5}",
                  end="\r", flush=True)
            time.sleep(period)
    except KeyboardInterrupt:
        if pend_hist:
            print(f"\n\nSuggested PEND_ZERO_OFFSET (median pend_enc): {int(statistics.median(pend_hist))}")
            print("Set it in backend-esp32/src/pins.h and re-flash. Then hanging should read ~2048.")
    finally:
        backend.close()


if __name__ == "__main__":
    main()
