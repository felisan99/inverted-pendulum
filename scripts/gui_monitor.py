"""
Real-time GUI monitor for the inverted pendulum simulation.

Runs PendulumSim in a background thread and displays:
  - Live plots of pendulum angle and arm position
  - Offscreen-rendered 3D view of the MuJoCo model
  - Control panel: PWM/voltage apply, hold-to-jog buttons, reset

Usage:
    python scripts/gui_monitor.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import mujoco

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from gym_envs.pendulum_sim import PendulumSim, PWM_MAX

from PySide6.QtCore import QObject, QThread, Signal, Slot, QTimer, Qt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter,
    QVBoxLayout, QGridLayout,
    QLabel, QPushButton, QSpinBox, QDoubleSpinBox, QGroupBox,
)
from PySide6.QtGui import QFontDatabase, QImage, QPixmap
import pyqtgraph as pg

_PEND_DEG_PER_COUNT = 360.0 / 4096
_ARM_DEG_PER_COUNT  = 360.0 / 1716
_SIM_TIMESTEP       = 0.001
_DATA_EMIT_EVERY    = 10   # emit data_ready every N steps → 100 Hz

HISTORY_S    = 5.0
MAXLEN       = int(HISTORY_S * 1000 / _DATA_EMIT_EVERY)  # 500
PLOT_HZ      = 30
RENDER_W     = 320
RENDER_H     = 240
RENDER_EVERY = 33


class RingBuffer:
    """Pre-allocated circular buffer for three float64 time-series."""

    def __init__(self, capacity: int) -> None:
        self._cap  = capacity
        self._t    = np.empty(capacity)
        self._pend = np.empty(capacity)
        self._arm  = np.empty(capacity)
        self._head = 0
        self._size = 0

    def append(self, t: float, pend: float, arm: float) -> None:
        i = self._head % self._cap
        self._t[i]    = t
        self._pend[i] = pend
        self._arm[i]  = arm
        self._head   += 1
        if self._size < self._cap:
            self._size += 1

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (t, pend, arm) in chronological order (views when possible)."""
        n = self._size
        if n == 0:
            empty = np.empty(0)
            return empty, empty, empty
        if n < self._cap:
            return self._t[:n], self._pend[:n], self._arm[:n]
        start = self._head % self._cap
        idx   = np.arange(start, start + self._cap) % self._cap
        return self._t[idx], self._pend[idx], self._arm[idx]

    def clear(self) -> None:
        self._head = 0
        self._size = 0


class SimWorker(QObject):
    """Runs PendulumSim at 1 kHz in a background QThread."""

    data_ready  = Signal(int, int, int)   # t_us, motor_enc, pend_enc
    frame_ready = Signal(object)          # numpy uint8 H×W×3 RGB

    def __init__(self) -> None:
        super().__init__()
        self._sim        = PendulumSim()
        self._pwm: int   = 0
        self._running    = False
        self._reset_flag = False

    @Slot()
    def start_sim(self) -> None:
        renderer = None
        try:
            renderer = mujoco.Renderer(self._sim._model, height=RENDER_H, width=RENDER_W)
        except Exception:
            pass

        self._sim.reset(pendulum_down=True)
        self._running = True
        step_count    = 0
        wall_next     = time.perf_counter()

        while self._running:
            if self._reset_flag:
                self._reset_flag = False
                self._sim.reset(pendulum_down=True)
                wall_next = time.perf_counter()

            reading = self._sim.step(self._pwm)
            step_count += 1

            if step_count % _DATA_EMIT_EVERY == 0:
                self.data_ready.emit(reading.t_us, reading.motor_enc, reading.pend_enc)

            if renderer is not None and step_count % RENDER_EVERY == 0:
                renderer.update_scene(self._sim._data)
                self.frame_ready.emit(renderer.render().copy())

            wall_next += _SIM_TIMESTEP
            slack = wall_next - time.perf_counter()
            if slack > 0:
                time.sleep(slack)

        if renderer is not None:
            renderer.close()

    @Slot(int)
    def set_pwm(self, pwm: int) -> None:
        self._pwm = pwm

    @Slot()
    def request_reset(self) -> None:
        self._pwm        = 0
        self._reset_flag = True

    @Slot()
    def stop(self) -> None:
        self._running = False


class MainWindow(QMainWindow):

    _send_pwm   = Signal(int)
    _send_reset = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Inverted Pendulum Monitor")
        self.resize(1200, 720)

        self._ring = RingBuffer(MAXLEN)

        self._build_ui()
        self._start_worker()

        self._timer = QTimer(self)
        self._timer.setInterval(1000 // PLOT_HZ)
        self._timer.timeout.connect(self._refresh_plots)
        self._timer.start()

    def _build_ui(self) -> None:
        pg.setConfigOptions(antialias=False, background="k", foreground="w")

        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)

        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        plot_layout.setContentsMargins(4, 4, 4, 4)
        plot_layout.setSpacing(4)

        self._pend_plot  = pg.PlotWidget(title="Pendulum angle  (upright = 0°)")
        self._pend_plot.setLabel("left", "Angle", units="deg")
        self._pend_plot.setLabel("bottom", "Time", units="s")
        self._pend_curve = self._pend_plot.plot(pen=pg.mkPen("#4fc3f7", width=1.5))
        self._pend_plot.showGrid(x=True, y=True, alpha=0.3)

        self._arm_plot   = pg.PlotWidget(title="Arm position  (cumulative)")
        self._arm_plot.setLabel("left", "Angle", units="deg")
        self._arm_plot.setLabel("bottom", "Time", units="s")
        self._arm_curve  = self._arm_plot.plot(pen=pg.mkPen("#a5d6a7", width=1.5))
        self._arm_plot.showGrid(x=True, y=True, alpha=0.3)

        plot_layout.addWidget(self._pend_plot)
        plot_layout.addWidget(self._arm_plot)
        splitter.addWidget(plot_widget)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(6, 6, 6, 6)
        right_layout.setSpacing(8)

        view_box    = QGroupBox("3D view")
        view_layout = QVBoxLayout(view_box)
        self._view_label = QLabel("Initializing…")
        self._view_label.setFixedSize(RENDER_W, RENDER_H)
        self._view_label.setAlignment(Qt.AlignCenter)
        self._view_label.setStyleSheet("background:#111; color:#666;")
        view_layout.addWidget(self._view_label, alignment=Qt.AlignCenter)
        right_layout.addWidget(view_box)

        right_layout.addWidget(self._build_control_panel())
        right_layout.addStretch()

        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

    def _build_control_panel(self) -> QWidget:
        panel  = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        mono = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        mono.setPointSize(11)

        status_box    = QGroupBox("Live readings")
        status_layout = QGridLayout(status_box)
        self._lbl_pend = QLabel("-- deg")
        self._lbl_arm  = QLabel("-- deg")
        self._lbl_pend.setFont(mono)
        self._lbl_arm.setFont(mono)
        status_layout.addWidget(QLabel("Pendulum:"), 0, 0)
        status_layout.addWidget(self._lbl_pend,      0, 1)
        status_layout.addWidget(QLabel("Arm:"),      1, 0)
        status_layout.addWidget(self._lbl_arm,       1, 1)
        layout.addWidget(status_box)

        cmd_box    = QGroupBox("Set command")
        cmd_layout = QGridLayout(cmd_box)

        cmd_layout.addWidget(QLabel("PWM  (-1023 … +1023):"), 0, 0, 1, 2)
        self._pwm_spin = QSpinBox()
        self._pwm_spin.setRange(-PWM_MAX, PWM_MAX)
        self._pwm_spin.setSingleStep(10)
        self._pwm_spin.setValue(0)
        self._pwm_spin.valueChanged.connect(self._sync_volt_from_pwm)
        cmd_layout.addWidget(self._pwm_spin, 1, 0, 1, 2)

        cmd_layout.addWidget(QLabel("Voltage  (-12 … +12 V):"), 2, 0, 1, 2)
        self._volt_spin = QDoubleSpinBox()
        self._volt_spin.setRange(-12.0, 12.0)
        self._volt_spin.setSingleStep(0.5)
        self._volt_spin.setDecimals(1)
        self._volt_spin.setValue(0.0)
        self._volt_spin.valueChanged.connect(self._sync_pwm_from_volt)
        cmd_layout.addWidget(self._volt_spin, 3, 0, 1, 2)

        btn_apply = QPushButton("Apply")
        btn_apply.clicked.connect(self._apply_pwm)
        btn_stop  = QPushButton("Stop  (PWM = 0)")
        btn_stop.clicked.connect(self._stop_motor)
        cmd_layout.addWidget(btn_apply, 4, 0)
        cmd_layout.addWidget(btn_stop,  4, 1)
        layout.addWidget(cmd_box)

        jog_box    = QGroupBox("Jog  (hold to move, releases to 0)")
        jog_layout = QGridLayout(jog_box)

        jog_layout.addWidget(QLabel("Magnitude (PWM):"), 0, 0)
        self._jog_spin = QSpinBox()
        self._jog_spin.setRange(1, PWM_MAX)
        self._jog_spin.setValue(300)
        self._jog_spin.setSingleStep(50)
        jog_layout.addWidget(self._jog_spin, 0, 1)

        btn_cw  = QPushButton("Jog +  (CW)")
        btn_ccw = QPushButton("Jog −  (CCW)")
        btn_cw.pressed.connect(lambda: self._send_pwm.emit(self._jog_spin.value()))
        btn_cw.released.connect(lambda: self._send_pwm.emit(0))
        btn_ccw.pressed.connect(lambda: self._send_pwm.emit(-self._jog_spin.value()))
        btn_ccw.released.connect(lambda: self._send_pwm.emit(0))
        jog_layout.addWidget(btn_cw,  1, 0)
        jog_layout.addWidget(btn_ccw, 1, 1)
        layout.addWidget(jog_box)

        btn_reset = QPushButton("Reset simulation")
        btn_reset.clicked.connect(self._reset_sim)
        layout.addWidget(btn_reset)

        return panel

    def _start_worker(self) -> None:
        self._thread = QThread(self)
        self._worker = SimWorker()
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.start_sim)
        self._worker.data_ready.connect(self._on_data)
        self._worker.frame_ready.connect(self._on_frame)

        self._send_pwm.connect(self._worker.set_pwm,         Qt.DirectConnection)
        self._send_reset.connect(self._worker.request_reset, Qt.DirectConnection)

        self._thread.start()

    @Slot(int, int, int)
    def _on_data(self, t_us: int, motor_enc: int, pend_enc: int) -> None:
        self._ring.append(
            t_us * 1e-6,
            pend_enc * _PEND_DEG_PER_COUNT - 180.0,
            motor_enc * _ARM_DEG_PER_COUNT,
        )

    @Slot(object)
    def _on_frame(self, rgb: np.ndarray) -> None:
        h, w = rgb.shape[:2]
        img  = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
        self._view_label.setPixmap(QPixmap.fromImage(img))

    def _refresh_plots(self) -> None:
        t, pend, arm = self._ring.arrays()
        if t.size == 0:
            return
        t_max = t[-1]
        idx   = np.searchsorted(t, t_max - HISTORY_S)

        self._pend_curve.setData(t[idx:], pend[idx:])
        self._arm_curve.setData(t[idx:], arm[idx:])
        self._pend_plot.setXRange(t_max - HISTORY_S, t_max, padding=0)
        self._arm_plot.setXRange(t_max - HISTORY_S, t_max, padding=0)

        self._lbl_pend.setText(f"{pend[-1]:+.1f} deg")
        self._lbl_arm.setText(f"{arm[-1]:+.1f} deg")

    def _sync_volt_from_pwm(self, pwm: int) -> None:
        self._volt_spin.blockSignals(True)
        self._volt_spin.setValue(pwm * 12.0 / PWM_MAX)
        self._volt_spin.blockSignals(False)

    def _sync_pwm_from_volt(self, volts: float) -> None:
        self._pwm_spin.blockSignals(True)
        self._pwm_spin.setValue(int(round(volts * PWM_MAX / 12.0)))
        self._pwm_spin.blockSignals(False)

    def _apply_pwm(self) -> None:
        self._send_pwm.emit(self._pwm_spin.value())

    def _stop_motor(self) -> None:
        self._pwm_spin.setValue(0)
        self._volt_spin.setValue(0.0)
        self._send_pwm.emit(0)

    def _reset_sim(self) -> None:
        self._pwm_spin.setValue(0)
        self._volt_spin.setValue(0.0)
        self._ring.clear()
        self._send_reset.emit()

    def closeEvent(self, event):
        self._timer.stop()
        self._worker.stop()
        self._thread.quit()
        self._thread.wait(2000)
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
