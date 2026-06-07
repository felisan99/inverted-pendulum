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

Non-ideal effects (sensor noise, I2C latency, FreeRTOS timing jitter) are driven
by a SimConfig. With the default SimConfig() the simulation is ideal and matches
the previous behaviour exactly.
"""

from __future__ import annotations

import math
import time
from collections import deque
from pathlib import Path

import mujoco
import numpy as np

from gym_envs.backend import SensorReading
from gym_envs.observation import PENDULUM_LSB, MOTOR_LSB
from gym_envs.sim_config import SimConfig

PWM_MAX     = 1023
MAX_VOLTAGE = 12.0

_RENDER_EVERY = 10


class PendulumSim:

    def __init__(self, xml_file: str | None = None, render_mode: str | None = None,
                 sim_config: SimConfig | None = None, seed: int | None = None) -> None:
        if xml_file is None:
            root = Path(__file__).resolve().parent.parent
            xml_file = root / "models" / "pendulum_model_v3.xml"

        xml_file = Path(xml_file).resolve()
        if not xml_file.exists():
            raise FileNotFoundError(f"Model file not found: {xml_file}")

        self._model = mujoco.MjModel.from_xml_path(str(xml_file))
        self._data  = mujoco.MjData(self._model)
        self._render_mode = render_mode
        self._viewer = None

        self._config   = sim_config or SimConfig()
        self._rng      = np.random.default_rng(seed)
        self._dt_us    = self._model.opt.timestep * 1e6
        self._clock_us = 0.0
        lat = self._config.sensor_latency_steps
        self._latency_buf: deque[SensorReading] = deque(maxlen=max(lat + 1, 1))

    def reset(self, pendulum_down: bool = True,
              initial_angle_rad: float | None = None,
              seed: int | None = None) -> SensorReading:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._clock_us = 0.0
        self._latency_buf.clear()
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
        if self._config.dt_jitter_sigma > 0:
            self._clock_us += max(self._dt_us + self._rng.normal(0, self._config.dt_jitter_sigma), 1.0)
        else:
            self._clock_us += self._dt_us
        if self._render_mode == "human":
            self._render()
        return self._read_sensors()

    def apply_disturbance(self, delta_rad: float) -> None:
        """Instantly displace the pendulum angle, as if someone hit it.

        A step disturbance on the pendulum joint position. Velocities are left
        untouched, so the finite-difference encoder sees a one-step spike and the
        active controller has to recover.
        """
        self._data.qpos[1] += delta_rad
        mujoco.mj_forward(self._model, self._data)

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    def _read_sensors(self) -> SensorReading:
        cfg = self._config
        t_us = int(self._clock_us)

        motor_enc = int(round(self._data.qpos[0] / MOTOR_LSB))
        pend_enc  = int(round(self._data.qpos[1] / PENDULUM_LSB)) % 4096
        if cfg.pend_noise_sigma > 0:
            pend_enc  = (pend_enc + int(round(self._rng.normal(0, cfg.pend_noise_sigma)))) % 4096
        if cfg.motor_noise_sigma > 0:
            motor_enc += int(round(self._rng.normal(0, cfg.motor_noise_sigma)))

        fresh = SensorReading(t_us, motor_enc, pend_enc)
        if cfg.sensor_latency_steps == 0:
            return fresh
        self._latency_buf.append(fresh)
        return self._latency_buf[0]

    def _render(self) -> None:
        if self._viewer is None:
            import mujoco_viewer
            self._viewer = mujoco_viewer.MujocoViewer(self._model, self._data)
            self._last_render_time = time.time()
            self._render_count = 0
        self._render_count += 1
        if self._render_count % _RENDER_EVERY == 0:
            if self._viewer.is_alive:
                self._viewer.render()
            expected = self._model.opt.timestep * _RENDER_EVERY
            elapsed  = time.time() - self._last_render_time
            if elapsed < expected:
                time.sleep(expected - elapsed)
            self._last_render_time = time.time()
