"""
Low-level simulation interface that mirrors the ESP32 firmware API.

Input:  signed 10-bit PWM command (-1023..+1023)
Output: SensorReading(t_us, motor_enc, pend_enc) — raw encoder counts

Usage:
    from gym_envs.pendulum_sim import PendulumSim

    sim = PendulumSim(render_mode="human")
    reading = sim.reset(pendulum_down=True)

    for _ in range(5000):          # 5 s at 1 kHz
        pwm = 256                  # PID / LQR output here
        reading = sim.step(pwm)
        # reading.t_us        microseconds
        # reading.motor_enc   cumulative Hall counts (signed)
        # reading.pend_enc    AS5600 absolute counts 0-4095

    sim.close()

Encoder conventions:
    motor_enc  : cumulative counts, can be negative.
                 1716 counts/rev at the output shaft (Hall x2 decoding).
                 To convert to radians: motor_enc * 2*pi / 1716
    pend_enc   : absolute position 0-4095, wraps on full revolution.
                 4096 counts/rev (AS5600 12-bit).
                 To convert to radians: pend_enc * 2*pi / 4096
"""

from __future__ import annotations

import math
from collections import namedtuple
from pathlib import Path

import mujoco

_PENDULUM_LSB = 2 * math.pi / 4096
_MOTOR_LSB    = 2 * math.pi / 1716

PWM_MAX     = 1023
MAX_VOLTAGE = 12.0

SensorReading = namedtuple("SensorReading", ["t_us", "motor_enc", "pend_enc"])


class PendulumSim:

    def __init__(self, xml_file: str | None = None, render_mode: str | None = None) -> None:
        if xml_file is None:
            root = Path(__file__).resolve().parent.parent
            xml_file = root / "mujoco_sim" / "xml_models" / "pendulum_model_v3.xml"

        xml_file = Path(xml_file).resolve()
        if not xml_file.exists():
            raise FileNotFoundError(f"Model file not found: {xml_file}")

        self._model = mujoco.MjModel.from_xml_path(str(xml_file))
        self._data  = mujoco.MjData(self._model)
        self._render_mode = render_mode
        self._viewer = None

    def reset(self, pendulum_down: bool = True,
              initial_angle_rad: float | None = None) -> SensorReading:
        mujoco.mj_resetData(self._model, self._data)
        if initial_angle_rad is not None:
            self._data.qpos[1] = initial_angle_rad
        else:
            self._data.qpos[1] = math.pi if pendulum_down else 0.0
        mujoco.mj_forward(self._model, self._data)
        return self._read_sensors()

    def step(self, pwm: int) -> SensorReading:
        pwm = max(-PWM_MAX, min(PWM_MAX, int(pwm)))
        self._data.ctrl[0] = pwm * MAX_VOLTAGE / PWM_MAX
        mujoco.mj_step(self._model, self._data)
        if self._render_mode == "human":
            self._render()
        return self._read_sensors()

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    def _read_sensors(self) -> SensorReading:
        t_us      = int(self._data.time * 1_000_000)
        motor_enc = int(round(self._data.qpos[0] / _MOTOR_LSB))
        pend_enc  = int(round(self._data.qpos[1] / _PENDULUM_LSB)) % 4096
        return SensorReading(t_us, motor_enc, pend_enc)

    def _render(self) -> None:
        if self._viewer is None:
            import mujoco_viewer
            self._viewer = mujoco_viewer.MujocoViewer(self._model, self._data)
        if self._viewer.is_alive:
            self._viewer.render()
