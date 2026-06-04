"""
Real-time GUI monitor for the inverted pendulum simulation.

Runs PendulumSim in a background thread and displays:
  - Live plots of pendulum angle and arm position
  - Offscreen-rendered 3D view of the MuJoCo model
  - Control panel: PWM/voltage apply, hold-to-jog buttons, reset
  - External controller plugin: load any Python script with a Controller class

Usage:
    python scripts/gui_monitor.py
"""

from __future__ import annotations

import importlib.util
import math
import sys
import time
from pathlib import Path

import numpy as np
import mujoco

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from gym_envs.pendulum_sim import PendulumSim, PWM_MAX
from gym_envs.sim_config import SimConfig
from gym_envs.policy_controller import ModelController

_SIM_CONFIG_PATH = root / "configs" / "sim_config.toml"
_RESULTS_DIR     = root / "results"

from PySide6.QtCore import QObject, QThread, Signal, Slot, QTimer, Qt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QSpinBox, QDoubleSpinBox, QGroupBox,
    QScrollArea, QLineEdit, QFileDialog, QComboBox, QCheckBox, QSizePolicy,
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
RENDER_W     = 800
RENDER_H     = 600
RENDER_EVERY = 33

_AGENTS = ("PPO", "SAC", "A2C")


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
    ctrl_error  = Signal(str)             # controller compute() raised

    def __init__(self) -> None:
        super().__init__()
        cfg = SimConfig.from_toml(_SIM_CONFIG_PATH) if _SIM_CONFIG_PATH.exists() else SimConfig()
        self._sim        = PendulumSim(sim_config=cfg)
        self._pwm: int   = 0
        self._running    = False

        # manual reset flag
        self._reset_flag = False

        # controller flags (set from main thread via DirectConnection, GIL-safe)
        self._ctrl_active            = False
        self._ctrl_start_flag        = False
        self._ctrl_stop_flag         = False
        self._controller             = None   # Controller instance
        self._ctrl_initial_angle_rad = 0.0

        # pending disturbance ("hit") in radians, accumulated across clicks
        self._disturb_pending_rad    = 0.0

    @Slot()
    def start_sim(self) -> None:
        renderer = None
        try:
            renderer = mujoco.Renderer(self._sim._model, height=RENDER_H, width=RENDER_W)
        except Exception:
            pass

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

            # choose PWM source
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

            if renderer is not None and step_count % RENDER_EVERY == 0:
                renderer.update_scene(self._sim._data)
                self.frame_ready.emit(renderer.render().copy())

            wall_next += _SIM_TIMESTEP
            slack = wall_next - time.perf_counter()
            if slack > 0:
                time.sleep(slack)

        if renderer is not None:
            renderer.close()

    # --- slots called via DirectConnection from main thread (GIL-safe) ---

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


class MainWindow(QMainWindow):

    _send_pwm        = Signal(int)
    _send_reset      = Signal()
    _send_ctrl_start = Signal()
    _send_ctrl_stop  = Signal()
    _send_disturb    = Signal(float)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Inverted Pendulum Monitor")
        self.resize(1400, 900)

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

        # ── Left: controls (always visible) ─────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(self._build_control_panel())
        scroll.setMinimumWidth(300)
        splitter.addWidget(scroll)

        # ── Center: 3D view (expanding) ─────────────────────────────────
        view_box    = QGroupBox("3D view")
        view_layout = QVBoxLayout(view_box)
        view_layout.setContentsMargins(4, 4, 4, 4)
        self._view_label = QLabel("Initializing…")
        self._view_label.setMinimumSize(320, 240)
        self._view_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._view_label.setAlignment(Qt.AlignCenter)
        self._view_label.setStyleSheet("background:#111; color:#666;")
        view_layout.addWidget(self._view_label)
        splitter.addWidget(view_box)

        # ── Right: plots (stacked, smaller) ─────────────────────────────
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

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([300, 760, 360])

    def _build_control_panel(self) -> QWidget:
        panel  = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        mono = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        mono.setPointSize(11)

        # Live readings
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

        # Set-and-apply command
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

        # Jog
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

        # External controller
        ctrl_box    = QGroupBox("External Controller")
        ctrl_layout = QGridLayout(ctrl_box)

        ctrl_layout.addWidget(QLabel("Script:"), 0, 0)
        self._ctrl_path_edit = QLineEdit()
        self._ctrl_path_edit.setPlaceholderText("path/to/controller.py")
        ctrl_layout.addWidget(self._ctrl_path_edit, 0, 1)
        btn_browse = QPushButton("···")
        btn_browse.setFixedWidth(32)
        btn_browse.clicked.connect(self._browse_controller)
        ctrl_layout.addWidget(btn_browse, 0, 2)

        ctrl_layout.addWidget(QLabel("Perturbation from upright:"), 1, 0)
        perturb_row = QWidget()
        perturb_layout = QHBoxLayout(perturb_row)
        perturb_layout.setContentsMargins(0, 0, 0, 0)
        self._ctrl_perturb_spin = QDoubleSpinBox()
        self._ctrl_perturb_spin.setRange(0.0, 30.0)
        self._ctrl_perturb_spin.setSingleStep(0.5)
        self._ctrl_perturb_spin.setDecimals(1)
        self._ctrl_perturb_spin.setValue(5.0)
        perturb_layout.addWidget(self._ctrl_perturb_spin)
        perturb_layout.addWidget(QLabel("deg"))
        ctrl_layout.addWidget(perturb_row, 1, 1, 1, 2)

        self._ctrl_start_btn = QPushButton("Start control")
        self._ctrl_start_btn.clicked.connect(self._start_control)
        self._ctrl_stop_btn  = QPushButton("Stop control")
        self._ctrl_stop_btn.clicked.connect(self._stop_control)
        self._ctrl_stop_btn.setEnabled(False)
        ctrl_layout.addWidget(self._ctrl_start_btn, 2, 0, 1, 2)
        ctrl_layout.addWidget(self._ctrl_stop_btn,  2, 2)

        self._ctrl_status_lbl = QLabel("Idle")
        self._ctrl_status_lbl.setFont(mono)
        ctrl_layout.addWidget(QLabel("Status:"), 3, 0)
        ctrl_layout.addWidget(self._ctrl_status_lbl, 3, 1, 1, 2)

        layout.addWidget(ctrl_box)

        # Trained model (SB3 .zip)
        model_box    = QGroupBox("Trained Model")
        model_layout = QGridLayout(model_box)

        model_layout.addWidget(QLabel("Model:"), 0, 0)
        self._model_path_edit = QLineEdit()
        self._model_path_edit.setPlaceholderText("results/run_N/best_model.zip")
        model_layout.addWidget(self._model_path_edit, 0, 1)
        btn_model_browse = QPushButton("···")
        btn_model_browse.setFixedWidth(32)
        btn_model_browse.clicked.connect(self._browse_model)
        model_layout.addWidget(btn_model_browse, 0, 2)

        model_layout.addWidget(QLabel("Algorithm:"), 1, 0)
        self._model_algo_combo = QComboBox()
        self._model_algo_combo.addItems(_AGENTS)
        model_layout.addWidget(self._model_algo_combo, 1, 1)
        self._model_deterministic = QCheckBox("Deterministic")
        self._model_deterministic.setChecked(True)
        model_layout.addWidget(self._model_deterministic, 1, 2)

        self._model_start_btn = QPushButton("Start model")
        self._model_start_btn.clicked.connect(self._start_model)
        self._model_stop_btn  = QPushButton("Stop model")
        self._model_stop_btn.clicked.connect(self._stop_control)
        self._model_stop_btn.setEnabled(False)
        model_layout.addWidget(self._model_start_btn, 2, 0, 1, 2)
        model_layout.addWidget(self._model_stop_btn,  2, 2)

        self._model_status_lbl = QLabel("Idle")
        self._model_status_lbl.setFont(mono)
        model_layout.addWidget(QLabel("Status:"), 3, 0)
        model_layout.addWidget(self._model_status_lbl, 3, 1, 1, 2)

        layout.addWidget(model_box)

        # Disturbance ("hit" the pendulum)
        hit_box    = QGroupBox("Disturbance  (hit the pendulum)")
        hit_layout = QGridLayout(hit_box)

        hit_layout.addWidget(QLabel("Magnitude:"), 0, 0)
        self._hit_spin = QDoubleSpinBox()
        self._hit_spin.setRange(0.5, 45.0)
        self._hit_spin.setSingleStep(0.5)
        self._hit_spin.setDecimals(1)
        self._hit_spin.setValue(5.0)
        self._hit_spin.setSuffix(" deg")
        hit_layout.addWidget(self._hit_spin, 0, 1)

        btn_hit_pos = QPushButton("Hit +")
        btn_hit_neg = QPushButton("Hit −")
        btn_hit_pos.clicked.connect(lambda: self._hit(+1))
        btn_hit_neg.clicked.connect(lambda: self._hit(-1))
        hit_layout.addWidget(btn_hit_pos, 1, 0)
        hit_layout.addWidget(btn_hit_neg, 1, 1)

        layout.addWidget(hit_box)

        # Reset
        btn_reset = QPushButton("Reset simulation")
        btn_reset.clicked.connect(self._reset_sim)
        layout.addWidget(btn_reset)

        layout.addStretch(1)
        return panel

    def _start_worker(self) -> None:
        self._thread = QThread(self)
        self._worker = SimWorker()
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.start_sim)
        self._worker.data_ready.connect(self._on_data)
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.ctrl_error.connect(self._on_ctrl_error)

        self._send_pwm.connect(self._worker.set_pwm,               Qt.DirectConnection)
        self._send_reset.connect(self._worker.request_reset,        Qt.DirectConnection)
        self._send_ctrl_start.connect(self._worker.activate_controller,   Qt.DirectConnection)
        self._send_ctrl_stop.connect(self._worker.deactivate_controller,  Qt.DirectConnection)
        self._send_disturb.connect(self._worker.apply_disturbance,         Qt.DirectConnection)

        self._thread.start()

    @Slot(int, int, int)
    def _on_data(self, t_us: int, motor_enc: int, pend_enc: int) -> None:
        pend_deg = pend_enc * _PEND_DEG_PER_COUNT
        if pend_deg > 180.0:
            pend_deg -= 360.0
        self._ring.append(
            t_us * 1e-6,
            pend_deg,
            motor_enc * _ARM_DEG_PER_COUNT,
        )

    @Slot(object)
    def _on_frame(self, rgb: np.ndarray) -> None:
        h, w = rgb.shape[:2]
        img  = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
        pix  = QPixmap.fromImage(img).scaled(
            self._view_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self._view_label.setPixmap(pix)

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

    # ── controller ─────────────────────────────────────────────────────

    def _browse_controller(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select controller script", str(root), "Python files (*.py)"
        )
        if path:
            self._ctrl_path_edit.setText(path)

    def _start_control(self) -> None:
        path = Path(self._ctrl_path_edit.text().strip())
        if not path.exists():
            self._ctrl_status_lbl.setText("Error: file not found")
            return

        spec = importlib.util.spec_from_file_location("user_controller", path)
        mod  = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception as e:
            self._ctrl_status_lbl.setText(f"Error loading: {e}")
            return

        if not hasattr(mod, "Controller"):
            self._ctrl_status_lbl.setText("Error: no Controller class found")
            return

        try:
            ctrl = mod.Controller()
            ctrl.reset()
        except Exception as e:
            self._ctrl_status_lbl.setText(f"Error init: {e}")
            return

        self._begin_control(ctrl, self._ctrl_status_lbl)

    def _browse_model(self) -> None:
        start_dir = _RESULTS_DIR if _RESULTS_DIR.exists() else root
        path, _ = QFileDialog.getOpenFileName(
            self, "Select trained model", str(start_dir), "SB3 models (*.zip)"
        )
        if path:
            self._model_path_edit.setText(path)

    def _start_model(self) -> None:
        path = Path(self._model_path_edit.text().strip())
        if not path.exists():
            self._model_status_lbl.setText("Error: file not found")
            return

        algo = self._model_algo_combo.currentText()
        self._model_status_lbl.setText("Loading…")
        QApplication.processEvents()
        try:
            from stable_baselines3 import PPO, SAC, A2C
            cls   = {"PPO": PPO, "SAC": SAC, "A2C": A2C}[algo]
            model = cls.load(str(path))
        except Exception as e:
            self._model_status_lbl.setText(f"Error loading: {e}")
            return

        ctrl = ModelController(model, deterministic=self._model_deterministic.isChecked())
        ctrl.reset()
        self._begin_control(ctrl, self._model_status_lbl)

    def _begin_control(self, controller, status_lbl: QLabel) -> None:
        self._worker._controller             = controller
        self._worker._ctrl_initial_angle_rad = self._ctrl_perturb_spin.value() * math.pi / 180.0
        self._active_status_lbl              = status_lbl
        self._ring.clear()
        self._send_ctrl_start.emit()

        self._ctrl_status_lbl.setText("Idle")
        self._model_status_lbl.setText("Idle")
        status_lbl.setText("Active")
        self._set_control_active(True)

    def _stop_control(self) -> None:
        self._send_ctrl_stop.emit()
        self._ctrl_status_lbl.setText("Idle")
        self._model_status_lbl.setText("Idle")
        self._set_control_active(False)

    def _set_control_active(self, active: bool) -> None:
        self._ctrl_start_btn.setEnabled(not active)
        self._ctrl_stop_btn.setEnabled(active)
        self._model_start_btn.setEnabled(not active)
        self._model_stop_btn.setEnabled(active)

    def _hit(self, sign: int) -> None:
        self._send_disturb.emit(sign * self._hit_spin.value() * math.pi / 180.0)

    @Slot(str)
    def _on_ctrl_error(self, msg: str) -> None:
        lbl = getattr(self, "_active_status_lbl", None) or self._ctrl_status_lbl
        lbl.setText(f"Error: {msg}")
        self._set_control_active(False)

    # ── manual controls ─────────────────────────────────────────────────

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
        if self._worker._ctrl_active:
            self._ctrl_status_lbl.setText("Idle")
            self._model_status_lbl.setText("Idle")
            self._set_control_active(False)
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
