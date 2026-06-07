from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
from PySide6.QtCore import QThread, Signal, Slot, QTimer, Qt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QSpinBox, QDoubleSpinBox, QGroupBox,
    QScrollArea, QLineEdit, QFileDialog, QComboBox, QCheckBox, QSizePolicy,
)
from PySide6.QtGui import QFontDatabase, QImage, QPixmap
import pyqtgraph as pg

from gym_envs.observation import PENDULUM_LSB, MOTOR_LSB
from gym_envs.pendulum_sim import PWM_MAX, MAX_VOLTAGE
from gym_envs.policy_controller import ModelController
from gui.ring_buffer import RingBuffer
from gui.workers import SimWorker, RenderWorker, _DATA_EMIT_EVERY

_ROOT        = Path(__file__).resolve().parent.parent
_RESULTS_DIR = _ROOT / "results"

_PEND_DEG_PER_COUNT = math.degrees(PENDULUM_LSB)
_ARM_DEG_PER_COUNT  = math.degrees(MOTOR_LSB)

HISTORY_S = 5.0
MAXLEN    = int(HISTORY_S * 1000 / _DATA_EMIT_EVERY)
PLOT_HZ   = 30

_AGENTS = ("PPO", "SAC", "A2C")


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

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(self._build_control_panel())
        scroll.setMinimumWidth(300)
        splitter.addWidget(scroll)

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
        self._volt_spin.setRange(-MAX_VOLTAGE, MAX_VOLTAGE)
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
        self._worker.ctrl_error.connect(self._on_ctrl_error)

        self._send_pwm.connect(self._worker.set_pwm,                     Qt.DirectConnection)
        self._send_reset.connect(self._worker.request_reset,              Qt.DirectConnection)
        self._send_ctrl_start.connect(self._worker.activate_controller,   Qt.DirectConnection)
        self._send_ctrl_stop.connect(self._worker.deactivate_controller,  Qt.DirectConnection)
        self._send_disturb.connect(self._worker.apply_disturbance,        Qt.DirectConnection)

        self._render_thread = QThread(self)
        self._render_worker = RenderWorker(self._worker._sim._model, self._worker._snapshot)
        self._render_worker.moveToThread(self._render_thread)
        self._render_thread.started.connect(self._render_worker.run)
        self._render_worker.frame_ready.connect(self._on_frame)

        self._thread.start()
        self._render_thread.start()

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

    def _browse_controller(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select controller script", str(_ROOT), "Python files (*.py)"
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
        start_dir = _RESULTS_DIR if _RESULTS_DIR.exists() else _ROOT
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

    def _sync_volt_from_pwm(self, pwm: int) -> None:
        self._volt_spin.blockSignals(True)
        self._volt_spin.setValue(pwm * MAX_VOLTAGE / PWM_MAX)
        self._volt_spin.blockSignals(False)

    def _sync_pwm_from_volt(self, volts: float) -> None:
        self._pwm_spin.blockSignals(True)
        self._pwm_spin.setValue(int(round(volts * PWM_MAX / MAX_VOLTAGE)))
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
        self._render_worker.stop()
        self._render_thread.quit()
        self._render_thread.wait(2000)
        self._worker.stop()
        self._thread.quit()
        self._thread.wait(2000)
        super().closeEvent(event)
