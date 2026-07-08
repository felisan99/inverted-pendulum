"""
Record the onboard ESP32 balance controller's telemetry stream to CSV, while
also mirroring its live state into a MuJoCo 3D window — a passive digital
twin, no physics stepping.

Same protocol/CSV output as scripts/record_onboard_telemetry.py; requires the
firmware in /Users/felipe/Developer/tests-tesis/realimentacion-de-estados.
Additionally opens a mujoco_viewer.MujocoViewer window on --xml (default
models/pendulum_high_quality.xml) and drives it from the live telemetry.

Serial reading runs on its own background thread, separate from rendering:
mujoco_viewer.MujocoViewer.render() is slow enough (the equivalent offscreen
mujoco.Renderer used by gui/workers.py measures 4-16 ms/call on this same
high-quality model) that calling it inline in the same loop that reads the
~1 kHz telemetry stream would stall the reader long enough to overflow the
OS's fixed-size serial receive buffer and desync the frame parser.

Note the thread assignment is the mirror image of gui/workers.py's split:
there, the GUI's Qt main thread is unavailable for rendering, so the offscreen
(thread-safe) mujoco.Renderer runs on a background QThread. Here we want the
interactive mujoco_viewer window instead, which uses GLFW and — per this
repo's own CLAUDE.md — requires the main thread on macOS (confirmed: creating
it off the main thread crashes the process). So here the *main* thread owns
the viewer/render loop, and *serial reading* moves to the background thread.
Either way the point is the same: a slow render call must never block
ingestion of the telemetry stream. The two threads hand off state through a
lock-protected (qpos, qvel) snapshot, same role as gui/workers.py's
StateSnapshot.

Telemetry frame (39 bytes):
    [0x55][t_us:i64][pend_enc:u16][motor_enc:i32][pwm:i16][flags:u8]
          [phi_d:f32][omega:f32][term_p:f32][term_d:f32][term_b:f32][crc8]
    flags bit0=armed  bit1=running  bit2=diverged  bit3=warm  bit4=pend_stale

Commands (1 byte): 'G' start (arm on release), 'S' stop, 'R' reset, 'A' force-arm.

Usage (hold the pendulum near upright, run, then release when it says ARMED):
    python scripts/mirror_onboard_telemetry.py --port /dev/ttyUSB0 --duration 20
    python scripts/mirror_onboard_telemetry.py --port /dev/ttyUSB0 --video data/mirror.mp4  # also save the 3D window
    python scripts/mirror_onboard_telemetry.py --port /dev/ttyUSB0 --no-viz   # headless, same as record_onboard_telemetry.py
"""

from __future__ import annotations

import argparse
import csv
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np

import mujoco
import mujoco_viewer

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from gym_envs.serial_protocol import crc8
from gym_envs.observation import pend_to_rad, MOTOR_LSB

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

_RENDER_FPS = 30
_WINDOW_W   = 800   # same as gui/workers.py's RENDER_W/RENDER_H — without an explicit
_WINDOW_H   = 600   # size, MujocoViewer opens at the full primary-monitor resolution


def _pend_to_deg(pend_enc: int) -> float:
    deg = pend_enc * 360.0 / _PEND_COUNTS_PER_REV
    if deg > 180.0:
        deg -= 360.0
    return deg


def _motor_to_deg(motor_enc: int) -> float:
    return motor_enc * 360.0 / _MOTOR_COUNTS_PER_REV


def _grab_frame(viewer) -> np.ndarray:
    """Read back the current window frame as an RGB uint8 array.

    mujoco_viewer.read_pixels() refuses to run in 'window' mode, so we
    replicate its body against the viewer's own scene/context: re-render into
    the window's back buffer and mjr_readPixels it, without swapping (the
    viewer's own render() owns what's displayed, so this leaves the visible
    frame untouched)."""
    import glfw
    w, h = glfw.get_framebuffer_size(viewer.window)
    viewer.viewport.width, viewer.viewport.height = w, h
    mujoco.mjv_updateScene(
        viewer.model, viewer.data, viewer.vopt, viewer.pert, viewer.cam,
        mujoco.mjtCatBit.mjCAT_ALL.value, viewer.scn,
    )
    mujoco.mjr_render(viewer.viewport, viewer.scn, viewer.ctx)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    mujoco.mjr_readPixels(img, None, viewer.viewport, viewer.ctx)
    return np.flipud(img)


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


class _StateSnapshot:
    """Lock-protected latest (qpos, qvel) handoff between the serial-reader
    thread and the main/render thread. Same role as gui/workers.py's
    StateSnapshot, minus the numpy/QThread machinery this single-writer/
    single-reader case doesn't need."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._qpos = (0.0, 0.0)
        self._qvel = (0.0, 0.0)

    def publish(self, qpos: tuple[float, float], qvel: tuple[float, float]) -> None:
        with self._lock:
            self._qpos = qpos
            self._qvel = qvel

    def read(self) -> tuple[tuple[float, float], tuple[float, float]]:
        with self._lock:
            return self._qpos, self._qvel


def _reader_loop(
    ser, args: argparse.Namespace, snapshot: _StateSnapshot | None,
    stop_event: threading.Event, rows: list[tuple],
) -> None:
    """Owns the serial port end-to-end: commands, frame reads, CSV rows,
    and the hardware-safety stop command on the way out. Runs on a
    background thread so a slow render call on the main thread never
    blocks it from draining the ~1 kHz telemetry stream."""
    if not args.no_start:
        ser.write(b"R")
        time.sleep(0.1)
        ser.write(b"G")
        print("Sent START. Hold near upright; release when it prints ARMED.")

    seen_armed = False
    t0 = time.perf_counter()
    try:
        while time.perf_counter() - t0 < args.duration and not stop_event.is_set():
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

            if snapshot is not None:
                qpos = (motor_enc * MOTOR_LSB, pend_to_rad(pend_enc))   # arm (cumulative), pendulum (wrapped, upright=0)
                qvel = (omega, phi_d)                                   # arm, pendulum angular velocities, rad/s
                snapshot.publish(qpos, qvel)
    finally:
        ser.write(b"S")
        ser.flush()
        time.sleep(0.1)
        ser.close()
        stop_event.set()   # tell the main/render loop we're done, whatever the exit reason


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", default="/dev/cu.usbserial-0001", help="serial port to read from")
    p.add_argument("--baud", type=int, default=921600)
    p.add_argument("--duration", type=float, default=20.0, help="seconds to record after start")
    p.add_argument("--csv", type=Path, default=_ROOT / "data" / "onboard_balance_mirror.csv")
    p.add_argument("--no-start", action="store_true", help="do not send G; just record")
    p.add_argument("--xml", type=Path, default=_ROOT / "models" / "pendulum_high_quality.xml",
                   help="MuJoCo model to mirror the live telemetry into")
    p.add_argument("--no-viz", action="store_true",
                   help="skip the MuJoCo mirror window; behaves like record_onboard_telemetry.py")
    p.add_argument("--video", type=Path, default=None,
                   help="also save the mirror window to this .mp4 (ignored with --no-viz)")
    args = p.parse_args()

    if args.video is not None and args.no_viz:
        p.error("--video needs the mirror window; it is incompatible with --no-viz")

    import serial

    ser = serial.Serial(args.port, args.baud, timeout=0.2)
    time.sleep(0.3)
    ser.reset_input_buffer()

    stop_event = threading.Event()
    rows: list[tuple] = []
    snapshot = None if args.no_viz else _StateSnapshot()

    reader_thread = threading.Thread(
        target=_reader_loop, args=(ser, args, snapshot, stop_event, rows), daemon=True,
    )
    reader_thread.start()

    try:
        if snapshot is not None:
            model = mujoco.MjModel.from_xml_path(str(args.xml))
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            viewer = mujoco_viewer.MujocoViewer(model, data, width=_WINDOW_W, height=_WINDOW_H)
            period = 1.0 / _RENDER_FPS
            frames: list[np.ndarray] = [] if args.video is not None else None
            try:
                while not stop_event.is_set():
                    if not viewer.is_alive:
                        print("viewer window closed, stopping")
                        stop_event.set()
                        break
                    qpos, qvel = snapshot.read()
                    data.qpos[0], data.qpos[1] = qpos
                    data.qvel[0], data.qvel[1] = qvel
                    mujoco.mj_forward(model, data)
                    viewer.render()
                    if frames is not None:
                        frames.append(_grab_frame(viewer))
                    time.sleep(period)
            finally:
                if frames:
                    import mediapy as media
                    args.video.parent.mkdir(parents=True, exist_ok=True)
                    media.write_video(str(args.video), frames, fps=_RENDER_FPS)
                    print(f"wrote {len(frames)} frames ({len(frames)/_RENDER_FPS:.1f}s) to {args.video}")
                viewer.close()
        else:
            while not stop_event.is_set():
                reader_thread.join(timeout=0.2)
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        stop_event.set()
        reader_thread.join(timeout=2.0)

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
