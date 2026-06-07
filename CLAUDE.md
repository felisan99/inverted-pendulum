# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a reinforcement learning project for controlling a physical inverted pendulum using a DC motor. The simulation runs in MuJoCo and agents are trained with Stable-Baselines3. The environment is designed to closely mirror real hardware (sensor quantization, motor dynamics) to enable sim-to-real transfer.

## Environment Setup

The project uses `uv` with a `.venv` directory. Activate it before running anything:

```bash
source .venv/bin/activate
```

Python version is pinned in `.python-version`. To install dev dependencies (pytest):

```bash
uv sync --group dev
```

## Common Commands

**Run a random episode to validate the environment:**
```bash
python scripts/random_episode_test.py --xml models/pendulum_high_quality.xml
```

**Run system characterization (fits model params against real bench data):**
```bash
python tools/characterize_system.py --config configs/sim_to_real_validation.toml
```

**Run tests with pytest:**
```bash
pytest test/
```

**Run a trained model:**
```bash
python -m agents.predict --model-path results/run_N/best_model.zip --xml-file models/pendulum_high_quality.xml --agent PPO --task equilibrium
```

## Architecture

### Key Modules

- **`gym_envs/`** - Simulation backend, observation encoding, and RL environment, organized in three layers so a policy trained in sim can later drive the real ESP32 without re-deriving feature extraction. See `gym_envs/CLAUDE.md` for the full design. Summary:
  - **`backend.py`** - `PendulumBackend` (a `runtime_checkable` Protocol) defines the firmware contract: signed 10-bit PWM in, raw `SensorReading(t_us, motor_enc, pend_enc)` out. `SensorReading` lives here (import it from `gym_envs.backend`). A future `HardwareBackend` would satisfy the same contract.
  - **`pendulum_sim.py`** - `PendulumSim`, the MuJoCo implementation of `PendulumBackend`. Mirrors the firmware: PWM in, raw counts out, 1 kHz. Applies optional sensor non-idealities from a `SimConfig` (noise, latency, timing jitter). Single owner of the human 3D viewer. Used by the GUI, the controller tests, and `PendulumEnv`.
  - **`observation.py`** - `ObservationEncoder`, the single source of truth converting a `SensorReading` to the 6-dim observation `[sin(motor), cos(motor), vel_motor, sin(pendulum), cos(pendulum), vel_pendulum]`. Same code in training and at deployment. `MOTOR_LSB`/`PENDULUM_LSB` live here.
  - **`sim_config.py`** - `SimConfig` dataclass (defaults = ideal) with `from_toml()`. Profiles in `configs/sim_ideal.toml` and `configs/sim_config.toml`.
  - **`pendulum_env.py`** - `PendulumEnv(gym.Env)`, a composition adapter over a backend + encoder (NOT a subclass of `PendulumSim`). Continuous action space: voltage `[-12V, +12V]`, routed through the backend's 10-bit PWM channel. Two tasks: `"equilibrium"` and `"swing_up"`. DC motor torque via the XML `general` actuator (`gainprm=0.2184`, `biasprm=-0.2385`). Timestep 0.001 s (1 kHz).

- **`agents/trainer.py`** - `RLTrainer` class wrapping Stable-Baselines3. Supports PPO, SAC, A2C. Saves results to `results/run_N/` with TensorBoard logs, monitor CSVs, and model checkpoints.

- **`agents/predict.py`** - CLI entrypoint for loading and running a saved model.

- **`tools/characterize_system.py`** - Standalone simulation tool for parameter identification. Reads a TOML config, patches the XML model in memory, runs a parameterized simulation, and outputs CSV/plots/video. Does not use the Gymnasium env.

- **`tools/compare_step_response.py`** - Compares a real hardware step-response CSV against the MuJoCo simulation. Extracts ωn and ζ via log-decrement from both signals and prints an error table. `--config` is optional: without it, physical parameters are taken from validated defaults and the step signal is extracted automatically from the CSV's PWM column. `--output <path>` saves the plot to a custom path; `--output` (no value) skips the plot entirely.

- **`tools/visualize_step_response.py`** - Interactive 3D visualizer for the STEP_1023_100 experiment. Opens a MuJoCo viewer window and shows the comparison plot when done. Use `--speed` to control playback rate (default 3.0×). Requires a display; on macOS use `mujoco_viewer.MujocoViewer` — do not use `mujoco.viewer.launch_passive` without `mjpython`.

- **`scripts/`** - Manual validation scripts (`random_episode_test.py`, `max_voltage_test.py`). Run interactively with `--xml models/pendulum_high_quality.xml`; require a display (not headless).

- **`models/`** - MuJoCo XML model files. `pendulum_high_quality.xml` is the validated model with full visual shaders. `pendulum_low_quality.xml` has the same physics but minimal visuals (no shadows, no MSAA, no skybox) for better GUI framerate.

- **`configs/`** - TOML configuration files for `characterize_system.py`. Each file defines physical parameters, initial conditions, input signal, and output paths.

- **`gui/`** - Real-time desktop GUI package (PySide6 + PyQtGraph). Run with `python -m gui`; requires a display. Split into:
  - **`ring_buffer.py`** - `RingBuffer`: pre-allocated circular buffer for plot time-series.
  - **`workers.py`** - `StateSnapshot`, `SimWorker` (1 kHz physics QThread), `RenderWorker` (30 Hz offscreen render QThread).
  - **`main_window.py`** - `MainWindow`: three-column layout (controls, 3D view, plots), controller/model loading UI, disturbance panel.
  - **`__main__.py`** - Entry point.


### Observation Design

Feature extraction lives in `ObservationEncoder` (`gym_envs/observation.py`), shared between training and deployment. Positions are encoded as `(sin, cos)` pairs rather than raw angles to avoid discontinuities at `±π`. Velocities are estimated by finite difference over quantized positions using the measured `dt` from `t_us` (matching firmware behavior on the ESP32), not taken directly from MuJoCo's `qvel`. The pendulum angle is mapped to `[-π, π]` centered on the upright; the velocity is continuous near 0 (equilibrium) but breaks at `±π` (hanging), so `swing_up` velocity is a known limitation of the encoder.

### Simulation Non-Idealities

`SimConfig` (`gym_envs/sim_config.py`) parameterizes hardware imperfections so the same controller or policy can be evaluated under different conditions: `pend_noise_sigma` / `motor_noise_sigma` (AS5600 / Hall jitter in counts), `sensor_latency_steps` (I2C delay), `dt_jitter_sigma` (FreeRTOS timing jitter, microseconds per interval). All fields default to zero (ideal sim, original behavior). Pass a `SimConfig` to `PendulumSim(sim_config=...)`, to `PendulumEnv(sim_config=...)`, or to `RLTrainer(sim_config=...)`. The GUI loads `configs/sim_config.toml` by convention if present. The timing clock advances in `PendulumSim.step()` (not in `_read_sensors()`) so `reset()` always reports `t_us == 0`.

### Results Structure

Training runs save to `results/run_N/` where N is auto-incremented. Each run contains: `model_final.zip`, `best_model.zip`, `train_monitor.csv`, `val_monitor.csv`, TensorBoard event files, and learning curve PNGs.

## Sim-to-Real Validation: STEP_1023_100

The reference experiment applies a 100 ms voltage pulse (PWM 1023, ~5 V) to the motor and records the free oscillation of the pendulum. It is the ground truth for validating that MuJoCo parameters match real hardware.

- Config: `configs/sim_to_real_validation.toml`
- XML model: `models/pendulum_high_quality.xml`
- Real data: `/Users/felipe/Documents/Tesis/Informe-Final/notes/experimento-validacion-sim-real/`
- Full documentation: `.../experimento-validacion-sim-real/README.md`

**Critical parameter — do not change without re-validating:**

`_FRICTION_DEFAULTS["joint1_frictionloss"] = 0.02` in `tools/characterize_system.py`.

This value is intentional. It proxies the mechanical resistance of the gearbox reduction that keeps the arm stationary during the pendulum's free oscillation. Without it, the arm drifts slightly and adds spurious damping, which overestimates ζ.

**Run the interactive visualizer:**

```bash
python tools/visualize_step_response.py --speed 3.0
```

**Run the sim-vs-real comparison:**

```bash
python tools/compare_step_response.py --real <ruta_al_CSV_real>
```

**Validated results (June 2025):**

| Metric | Real | Sim | Error |
|--------|------|-----|-------|
| ωn (rad/s) | 6.261 | 6.255 | 0.09% |
| ζ | 0.01140 | 0.01152 | 1.07% |
| Amplitude ratio | 1.0 | ~0.94 | ~6% |

Rm updated to 5.0 Ω (midpoint between 4.808 Ω measured at 5V and 5.5 Ω measured at 12V, consistent with voltmeter measurement between terminals). Expected to reduce the ~6% amplitude deficit to ~3%.

## GUI Monitor architecture

The `gui/` package uses a threading model designed around the 1 kHz simulation loop.

**Threading (three participants)**
- `SimWorker` (QObject moved to a `QThread`): runs `PendulumSim.step()` in a tight loop at 1 kHz, real-time throttled with `time.sleep`. Every 10 steps (100 Hz) it emits `data_ready` and calls `StateSnapshot.publish(qpos, qvel)`. It does NOT render.
- `RenderWorker` (QObject moved to a second `QThread`): on its own ~30 Hz loop, reads the latest `StateSnapshot`, applies it to its own `MjData`, runs `mj_forward`, then `update_scene` + `render`, and emits `frame_ready`. The `mujoco.Renderer` is created inside this thread (the GL context is thread-affine).
- Main thread: Qt event loop. A `QTimer` at 30 Hz reads a `RingBuffer` and calls `setData` on the PyQtGraph curves; `frame_ready` lands on `_on_frame`, which scales the pixmap to the (resizable) view.

**Why a dedicated render thread (measured)**
`sim.step()` costs ~0.003 ms but `render()` costs ~4 ms (p99 ~15 ms, max 16). With the render inline in the 1 kHz loop, each render iteration stalled the physics loop for 4-16 ms, forcing `wall_next` catch-up bursts that distort the real-time pacing the monitor represents (worse with `SimConfig` jitter). A background-thread benchmark showed a Python counter ran at 40.16 M/s during `sleep` and 39.88 M/s during `render` (99.3%), i.e. **MuJoCo releases the GIL during render**, so the render thread runs in true parallel with the physics loop. After the split, the physics iteration is always sub-millisecond. (`PPO.predict()` in `ModelController` is ~0.056 ms, so running a policy at 1 kHz is not a bottleneck and is left as-is.)

**Why `StateSnapshot` (lock + double copy)**
The two threads each own a separate `MjData`; they cannot share one safely while the physics thread mutates it via `mj_step`. `StateSnapshot` is a tiny `threading.Lock`-protected `(qpos, qvel)` buffer: the physics thread `publish()`es right after `step()` (same thread, no race with `mj_step`), the render thread `read()`s a copy. Contention is negligible (publish ~100/s, read ~30/s, a few floats each). The render thread reconstructs full kinematics from the snapshot with `mj_forward`.

**Why `DirectConnection` for control signals**
The worker loop never returns to the Qt event loop, so queued slots are never delivered. `Qt.DirectConnection` makes the slot execute in the calling (main) thread, setting a plain int/bool flag the loop polls — GIL-safe in CPython.

**Why `data_ready` fires at 100 Hz, not 1 kHz**
The plot refreshes at 30 Hz. Emitting 1000 cross-thread queued signals per second is wasteful. 100 Hz provides ample resolution and reduces event queue pressure by 10x.

**Why `RingBuffer` instead of `deque[tuple]`**
`np.array(deque)` on every plot frame creates a full copy. `RingBuffer` writes directly into pre-allocated `np.ndarray`s; `arrays()` returns views (no allocation).

**Why `np.searchsorted` instead of a boolean mask**
`arr[bool_mask]` creates a copy. `arr[i:]` is a view. `searchsorted` finds the window start in O(log N).

**Why `mujoco.Renderer` instead of `mujoco_viewer`**
`mujoco_viewer` uses GLFW and requires the main thread, conflicting with Qt on macOS. `mujoco.Renderer` is purely offscreen and can be created in any thread.

**Angle conventions**

| Signal | Conversion | Display meaning |
|--------|------------|-----------------|
| Pendulum | `pend_enc * 360/4096 - 180` | 0° = upright, ±180° = hanging |
| Arm | `motor_enc * 360/1716` | Cumulative degrees, unbounded |

**Controller script system**

The GUI can load an arbitrary Python script that defines a `Controller` class with two methods:

```python
class Controller:
    def reset(self) -> None: ...
    def compute(self, pend_enc: int, motor_enc: int, t_us: int) -> int: ...
```

`compute()` is called every simulation step and must return a PWM value in `[-1023, +1023]`. The GUI passes raw encoder counts — `pend_enc` is 0–4095 (0 = upright), `motor_enc` is unbounded signed Hall counts, `t_us` is simulation time in microseconds.

Loading flow: the user selects a script file and an initial perturbation angle, then clicks **Start control**. `SimWorker` re-imports the module fresh (so edits take effect without restarting), calls `reset()`, then routes every `step()` call through `compute()` instead of the manual PWM flag. If `compute()` raises, the worker catches the exception, sets PWM to 0, and emits an error string to the status label. **Stop control** returns to manual mode.

The script file and `Controller` class live entirely outside this repo; no imports from the project are needed. See `docs/custom-controller-tutorial.md` for the full interface contract, sensor reference, and an example (`controllers/pid_example.py`).

**Trained model system**

Besides external scripts, the GUI can load a Stable-Baselines3 `.zip` (the output of `agents/trainer.py`) and let the policy drive the pendulum. The user picks the model file, the algorithm (PPO/SAC/A2C), and a Deterministic toggle, then clicks **Start model**. Internally the GUI wraps the model in `ModelController` (`gym_envs/policy_controller.py`), which has the same `Controller` shape (`reset()` + `compute()`), so `SimWorker` routes it through the exact same control path as a script; only the two UI sections differ. The crucial difference from a script is that `ModelController` reuses the project's `ObservationEncoder`, so the network sees the same 6-dim observation it was trained on (no sim-to-real feature drift). The two sources are mutually exclusive (one `_controller` slot in the worker).

**Disturbance ("hit")**

The control panel has a Disturbance group: a magnitude in degrees plus **Hit +** / **Hit −** buttons. Clicking emits a signed radian delta via `_send_disturb` (a `Signal(float)` on `Qt.DirectConnection`); the worker accumulates it in `_disturb_pending_rad` and, at the top of its loop, calls `PendulumSim.apply_disturbance(delta_rad)`, which bumps the pendulum joint `qpos` and re-runs `mj_forward`. It is a step disturbance on position (velocities untouched), so any active controller (manual, script, or model) has to recover. Works independently of the control source.

**Extending the GUI**
- New command: add a flag to `SimWorker`, apply it in the loop, add a `@Slot` connected with `Qt.DirectConnection`.
- New plot: add a column to `RingBuffer`, update `_on_data` and `_refresh_plots`.
- New controller feature: the interface contract is the only coupling point; anything else can be added to `Controller` freely.

## Tests

There is no CI configured (the GitHub Actions workflow was removed). Run the suite locally with `pytest test/`. Tests use `render_mode=None` and do not require a display.

## Empirical Test Results

Real hardware test data and empirical measurements are stored outside this repository at `/Users/felipe/Documents/Tesis/Informe-Final`. Consult that directory when you need to compare simulation behavior against real bench results or when working on parameter identification.
