from __future__ import annotations

import threading
import time
from pathlib import Path

import mujoco
import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

from gym_envs.pendulum_sim import PendulumSim, PWM_MAX
from gym_envs.sim_config import SimConfig

_SIM_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "sim_config.toml"

_SIM_TIMESTEP    = 0.001
_DATA_EMIT_EVERY = 10

RENDER_W   = 800
RENDER_H   = 600
RENDER_FPS = 30


class StateSnapshot:
    """Lock-protected handoff of (qpos, qvel) from the physics thread to the render thread."""

    def __init__(self, nq: int, nv: int) -> None:
        self._lock = threading.Lock()
        self._qpos = np.zeros(nq)
        self._qvel = np.zeros(nv)

    def publish(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        with self._lock:
            self._qpos[:] = qpos
            self._qvel[:] = qvel

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            return self._qpos.copy(), self._qvel.copy()


class SimWorker(QObject):
    """Runs PendulumSim at 1 kHz in a background QThread."""

    data_ready = Signal(int, int, int)   # t_us, motor_enc, pend_enc
    ctrl_error = Signal(str)             # controller compute() raised

    def __init__(self) -> None:
        super().__init__()
        cfg = SimConfig.from_toml(_SIM_CONFIG_PATH) if _SIM_CONFIG_PATH.exists() else SimConfig()
        self._sim        = PendulumSim(sim_config=cfg)
        self._snapshot   = StateSnapshot(self._sim._model.nq, self._sim._model.nv)
        self._pwm: int   = 0
        self._running    = False

        self._reset_flag = False

        self._ctrl_active            = False
        self._ctrl_start_flag        = False
        self._ctrl_stop_flag         = False
        self._controller             = None
        self._ctrl_initial_angle_rad = 0.0

        self._disturb_pending_rad = 0.0

    @Slot()
    def start_sim(self) -> None:
        last_reading = self._sim.reset(pendulum_down=True)
        self._running = True
        step_count    = 0
        wall_next     = time.perf_counter()

        while self._running:

            if self._reset_flag:
                self._reset_flag  = False
                self._ctrl_active = False
                last_reading = self._sim.reset(pendulum_down=True)
                wall_next = time.perf_counter()

            if self._ctrl_start_flag:
                self._ctrl_start_flag = False
                self._ctrl_active     = True
                last_reading = self._sim.reset(
                    initial_angle_rad=self._ctrl_initial_angle_rad
                )
                wall_next = time.perf_counter()

            if self._ctrl_stop_flag:
                self._ctrl_stop_flag = False
                self._ctrl_active    = False
                self._pwm            = 0

            if self._disturb_pending_rad != 0.0:
                self._sim.apply_disturbance(self._disturb_pending_rad)
                self._disturb_pending_rad = 0.0

            if self._ctrl_active and self._controller is not None:
                try:
                    pwm = self._controller.compute(
                        last_reading.pend_enc,
                        last_reading.motor_enc,
                        last_reading.t_us,
                    )
                except Exception as e:
                    self._ctrl_active = False
                    self.ctrl_error.emit(str(e))
                    pwm = 0
            else:
                pwm = self._pwm

            last_reading = self._sim.step(pwm)
            step_count  += 1

            if step_count % _DATA_EMIT_EVERY == 0:
                self.data_ready.emit(
                    last_reading.t_us,
                    last_reading.motor_enc,
                    last_reading.pend_enc,
                )
                self._snapshot.publish(self._sim._data.qpos, self._sim._data.qvel)

            wall_next += _SIM_TIMESTEP
            slack = wall_next - time.perf_counter()
            if slack > 0:
                time.sleep(slack)

    @Slot(int)
    def set_pwm(self, pwm: int) -> None:
        self._pwm = pwm

    @Slot()
    def request_reset(self) -> None:
        self._pwm        = 0
        self._reset_flag = True

    @Slot()
    def activate_controller(self) -> None:
        self._ctrl_start_flag = True

    @Slot()
    def deactivate_controller(self) -> None:
        self._ctrl_stop_flag = True

    @Slot(float)
    def apply_disturbance(self, delta_rad: float) -> None:
        self._disturb_pending_rad += delta_rad

    @Slot()
    def stop(self) -> None:
        self._running = False


class RenderWorker(QObject):
    """Offscreen 3D render at RENDER_FPS in its own QThread.

    Reads a StateSnapshot published by the physics thread and renders from its own
    MjData, so the expensive render() (which releases the GIL) runs in parallel with
    the 1 kHz loop instead of stalling it.
    """

    frame_ready = Signal(object)   # numpy uint8 H×W×3 RGB

    def __init__(self, model, snapshot: StateSnapshot) -> None:
        super().__init__()
        self._model    = model
        self._snapshot = snapshot
        self._data     = mujoco.MjData(model)
        self._running  = False

    @Slot()
    def run(self) -> None:
        try:
            renderer = mujoco.Renderer(self._model, height=RENDER_H, width=RENDER_W)
        except Exception:
            return

        self._running = True
        period    = 1.0 / RENDER_FPS
        wall_next = time.perf_counter()
        while self._running:
            qpos, qvel = self._snapshot.read()
            self._data.qpos[:] = qpos
            self._data.qvel[:] = qvel
            mujoco.mj_forward(self._model, self._data)
            renderer.update_scene(self._data)
            self.frame_ready.emit(renderer.render().copy())

            wall_next += period
            slack = wall_next - time.perf_counter()
            if slack > 0:
                time.sleep(slack)
            else:
                wall_next = time.perf_counter()

        renderer.close()

    @Slot()
    def stop(self) -> None:
        self._running = False
