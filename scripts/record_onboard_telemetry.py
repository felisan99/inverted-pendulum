"""
Record the onboard ESP32 balance controller's telemetry stream to CSV.

For the firmware in /Users/felipe/Developer/tests-tesis/realimentacion-de-estados,
where the whole control loop runs on the ESP32 at 1 kHz. Serial is one-way telemetry
plus single-byte commands; this script sends the commands and logs the stream.

Telemetry frame (39 bytes):
    [0x55][t_us:i64][pend_enc:u16][motor_enc:i32][pwm:i16][flags:u8]
          [phi_d:f32][omega:f32][term_p:f32][term_d:f32][term_b:f32][crc8]
    flags bit0=armed  bit1=running  bit2=diverged  bit3=warm  bit4=pend_stale

Commands (1 byte): 'G' start (arm on release), 'S' stop, 'R' reset, 'A' force-arm.

Usage (hold the pendulum near upright, run, then release when it says ARMED):
    python scripts/record_onboard_telemetry.py --port /dev/ttyUSB0 --duration 20
"""

from __future__ import annotations

import argparse
import csv
import struct
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from gym_envs.serial_protocol import crc8

_SYNC        = 0x55
_PAYLOAD_FMT = "<qHihB5f"                        # t_us, pend_enc, motor_enc, pwm, flags, phi_d, omega, term_p, term_d, term_b
_PAYLOAD_LEN = struct.calcsize(_PAYLOAD_FMT)     # 37
_FRAME_LEN   = _PAYLOAD_LEN + 2                  # 39

_FLAG_ARMED      = 0x01
_FLAG_RUNNING    = 0x02
_FLAG_DIVERGED   = 0x04
_FLAG_WARM       = 0x08
_FLAG_PEND_STALE = 0x10

_PEND_COUNTS_PER_REV  = 4096
_MOTOR_COUNTS_PER_REV = 1716


def _pend_to_deg(pend_enc: int) -> float:
    deg = pend_enc * 360.0 / _PEND_COUNTS_PER_REV
    if deg > 180.0:
        deg -= 360.0
    return deg


def _motor_to_deg(motor_enc: int) -> float:
    return motor_enc * 360.0 / _MOTOR_COUNTS_PER_REV


def _read_frame(ser) -> tuple | None:
    sync = ser.read(1)
    if not sync or sync[0] != _SYNC:
        return None
    payload = ser.read(_PAYLOAD_LEN)
    crc = ser.read(1)
    if len(payload) != _PAYLOAD_LEN or len(crc) != 1:
        return None
    if crc8(payload) != crc[0]:
        return None
    return struct.unpack(_PAYLOAD_FMT, payload)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", default="/dev/cu.usbserial-0001", help="serial port to read from")
    p.add_argument("--baud", type=int, default=921600)
    p.add_argument("--duration", type=float, default=20.0, help="seconds to record after start")
    p.add_argument("--csv", type=Path, default=_ROOT / "data" / "onboard_balance.csv")
    p.add_argument("--no-start", action="store_true", help="do not send G; just record")
    args = p.parse_args()

    import serial

    ser = serial.Serial(args.port, args.baud, timeout=0.2)
    time.sleep(0.3)
    ser.reset_input_buffer()

    if not args.no_start:
        ser.write(b"R")
        time.sleep(0.1)
        ser.write(b"G")
        print("Sent START. Hold near upright; release when it prints ARMED.")

    rows: list[tuple] = []
    seen_armed = False
    t0 = time.perf_counter()
    try:
        while time.perf_counter() - t0 < args.duration:
            frame = _read_frame(ser)
            if frame is None:
                continue
            t_us, pend_enc, motor_enc, pwm, flags, phi_d, omega, term_p, term_d, term_b = frame

            phi_deg       = _pend_to_deg(pend_enc)
            theta_arm_deg = _motor_to_deg(motor_enc)
            armed    = bool(flags & _FLAG_ARMED)
            running  = bool(flags & _FLAG_RUNNING)
            diverged = bool(flags & _FLAG_DIVERGED)
            warm     = bool(flags & _FLAG_WARM)
            stale    = bool(flags & _FLAG_PEND_STALE)

            rows.append((
                t_us, pend_enc, motor_enc, pwm, flags,
                phi_deg, theta_arm_deg,
                phi_d, omega,
                term_p, term_d, term_b,
                int(armed), int(running), int(diverged), int(warm), int(stale),
            ))

            if armed and not seen_armed:
                seen_armed = True
                print(f"ARMED @ t={t_us*1e-6:.2f}s")
            if diverged:
                print(f"DIVERGED @ t={t_us*1e-6:.2f}s  phi={phi_deg:.1f} deg")
                break
            if stale:
                print(f"PEND_STALE @ t={t_us*1e-6:.2f}s")

    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        ser.write(b"S")
        ser.flush()
        time.sleep(0.1)
        ser.close()

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "t_us", "pend_enc", "motor_enc", "pwm", "flags",
            "phi_deg", "theta_arm_deg",
            "phi_d_rad_s", "omega_arm_rad_s",
            "term_p", "term_d", "term_b",
            "armed", "running", "diverged", "warm", "pend_stale",
        ])
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.csv}")

    armed_rows = [r for r in rows if r[13] and r[12]]   # running and armed
    if len(armed_rows) > 1:
        import statistics as _st
        pwms   = [r[3]  for r in armed_rows]
        phis   = [r[5]  for r in armed_rows]
        phi_ds = [r[7]  for r in armed_rows]
        span_s = (armed_rows[-1][0] - armed_rows[0][0]) * 1e-6
        sat    = sum(abs(x) >= 1020 for x in pwms) / len(pwms)
        signs  = sum(1 for a, b in zip(pwms, pwms[1:]) if (a > 0) != (b > 0) and a != 0 and b != 0)
        rms_pwm  = (_st.fmean(x * x for x in pwms)) ** 0.5
        rms_phi  = (_st.fmean(x * x for x in phis)) ** 0.5
        max_phi  = max(abs(x) for x in phis)
        rms_phid = (_st.fmean(x * x for x in phi_ds)) ** 0.5
        stale_n  = sum(r[16] for r in armed_rows)
        print(
            f"balanced  {span_s:.2f}s  "
            f"sat={sat*100:.1f}%  sign/s={signs/max(span_s,1e-9):.1f}  rms_pwm={rms_pwm:.1f}\n"
            f"phi: rms={rms_phi:.2f} deg  max={max_phi:.2f} deg\n"
            f"phi_dot: rms={rms_phid:.2f} rad/s\n"
            f"pend_stale frames: {stale_n}"
        )


if __name__ == "__main__":
    main()
