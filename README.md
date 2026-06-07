# Furuta Pendulum — RL + Sim-to-Real

MuJoCo simulation and reinforcement learning environment for a Furuta pendulum (rotary inverted pendulum). The simulation model is validated against real hardware.

## Installation

```bash
uv sync
source .venv/bin/activate
```

## Sim-to-real validation

The model `models/pendulum_model_v3.xml` is the only model in the repository. All parameters come from empirical measurements on the physical system.

### Physical parameters — arm (link1)

| Parameter | Symbol | Value | Source |
|---|---|---|---|
| Pivot-to-pivot distance | L1 | 0.0676 m | Measured |
| Arm mass | m1 | 0.022 kg | Bar (13 g) + bearing (9 g) |
| Arm CDM from motor axis | l1 | 0.0490 m | Composite model |
| Arm inertia about CDM | J1zz | 8.382e-6 kg·m² | Composite model (Steiner) |
| Reflected rotor inertia | armature | 8.797e-3 kg·m² | J_total - J1,motor; J_total = tau_rise * (b_arm + K_be) |

### Physical parameters — pendulum (encoder_joint)

| Parameter | Symbol | Value | Source |
|---|---|---|---|
| Pendulum length | L2 | 0.300 m | Measured |
| Pendulum mass + attachment | m2 | 0.034 kg | Rod (16.8 g) + bearing (17.2 g) |
| CDM with attachment | l2 | 0.2149 m | Measured directly |
| Pendulum inertia about CDM | J2zz | 2.550e-4 kg·m² | Thin-rod approximation |
| Pendulum natural frequency | wn | 6.267 rad/s sim / 6.255 rad/s real | Error 0.2% |
| Damping ratio | zeta | 0.0114 | STEP_1023_100 experiment |

Derived canonical parameters: gamma = J2zz + m2*l2² = 1.8252e-3 kg·m², delta = g*m2*l2 = 7.168e-2 N·m.

### Motor actuator parameters (DC motor JGA25-370, n=78, 12 V)

| Parameter | Symbol | Value | Derivation |
|---|---|---|---|
| Actuator gain | gainprm | 0.2184 N·m/V | Kt,sal / Rm = 1.092 / 5.0 |
| Back-EMF damping | biasprm[2] | -0.2385 N·m·s/rad | -Kt,sal² / Rm |
| Control range | ctrlrange | ±12 V | Supply voltage |
| Stall torque | forcerange | ±2.621 N·m | Kt,sal * Vmax / Rm |
| Terminal resistance | Rm | 5.0 Ω | Midpoint between 4.808 Ω (measured at 5V) and 5.5 Ω (measured at 12V) |
| Torque/back-EMF constant (output shaft) | Kt,sal = Ke,sal | 1.092 N·m/A | n * Kt_motor, n=78 |

### Friction and damping

| Parameter | Value | Notes |
|---|---|---|
| joint1 damping (b_arm) | 2.0e-3 N·m·s/rad | Step response without pendulum: b_arm = K_eff/K - K_be |
| joint1 frictionloss (RL) | 0.0 N·m | Kinetic friction negligible at operating speeds |
| joint1 frictionloss (sim-to-real exp) | 0.02 N·m | Static friction proxy to lock arm during free oscillation |
| encoder_joint damping (b2) | 2.60e-4 N·m·s/rad | b2 = 2*zeta*sqrt(gamma*delta), zeta=0.0114 |

### Sensor quantization (RL environment)

| Sensor | Resolution | Implementation |
|---|---|---|
| AS5600 (pendulum) | 4096 counts/rev, 12-bit | `_PENDULUM_LSB = 2*pi/4096` |
| Hall encoder x2 (motor) | 1716 steps/rev at output shaft | `_MOTOR_LSB = 2*pi/1716` |

Velocities are estimated by finite difference over quantized positions, replicating ESP32 firmware behavior. Timestep is 1 ms in both simulation and firmware.

### Validation results (STEP_1023_100, June 2025)

| Metric | Real | Sim | Error |
|---|---|---|---|
| Natural frequency wn | 6.255 rad/s | 6.267 rad/s | 0.2% |
| Damping ratio zeta | 0.0114 | 0.0114 | 0.3% |
| Oscillation amplitude ratio | 1.0 | ~0.94 | ~6% (R uncertainty) |

---

## Architecture: backend, encoder, environment

The simulation code in `gym_envs/` is organized in three decoupled layers so that a
policy trained in simulation can later drive the real ESP32 without re-deriving any
feature extraction. See [gym_envs/CLAUDE.md](gym_envs/CLAUDE.md) for the full design.

| Layer | File | Role |
|---|---|---|
| Contract | `backend.py` | `PendulumBackend` Protocol + `SensorReading`. PWM in, raw counts out. The firmware API. |
| Plant | `pendulum_sim.py` | `PendulumSim`: MuJoCo implementation of the backend, with optional sensor non-idealities. |
| Features | `observation.py` | `ObservationEncoder`: raw counts to the 6-dim policy observation. Shared sim/deploy. |
| Environment | `pendulum_env.py` | `PendulumEnv(gym.Env)`: composition over backend + encoder + reward. Training only. |

`PendulumEnv` is built by composition over a `PendulumBackend`, not by subclassing
`PendulumSim`. At deployment there is no `PendulumEnv`: you use a `HardwareBackend`
(the real ESP32 link, future work) plus the same `ObservationEncoder` and the trained
network.

## Simulating hardware non-idealities

`SimConfig` (`gym_envs/sim_config.py`) parameterizes hardware imperfections so the same
controller or RL policy can be evaluated under different conditions. Every field defaults
to its ideal value (zero effect), so the default is the original noise-free simulation.

| Field | Units | Models |
|---|---|---|
| `pend_noise_sigma` | AS5600 counts | Magnetic jitter on the pendulum encoder |
| `motor_noise_sigma` | Hall counts | Noise on the motor encoder |
| `sensor_latency_steps` | steps @ 1 kHz | I2C / read latency (pure delay) |
| `dt_jitter_sigma` | microseconds | Per-interval FreeRTOS timing jitter |

Two reference profiles ship in `configs/`: `sim_ideal.toml` (all zeros) and
`sim_config.toml` (estimated hardware non-idealities — the active profile). Use a
profile from code:

```python
from gym_envs.sim_config import SimConfig
from gym_envs.pendulum_env import PendulumEnv

cfg = SimConfig.from_toml("configs/sim_config.toml")
env = PendulumEnv(task="equilibrium", sim_config=cfg, seed=0)   # noise is reproducible per seed
```

`PendulumSim`, `PendulumEnv`, and `RLTrainer` all accept a `sim_config=` argument.
The GUI monitor loads `configs/sim_config.toml` by convention at startup if that file
exists. Delete or rename it to return to the ideal (noise-free) sim.

---

## Getting started

Everything below assumes you have activated the virtual environment:

```bash
source .venv/bin/activate
```

All commands are run from the repository root.

---

### 1. Watch the simulation

The fastest way to see something working. Runs the validated STEP_1023_100 experiment in a 3D viewer at 3x real time, then opens the comparison plot.

```bash
python tools/visualize_step_response.py --speed 3.0
```

- Close the 3D window to end the simulation early.
- Change `--speed` to adjust playback rate. Use `--speed 1.0` for real time.
- The comparison plot opens automatically when the simulation finishes.

---

### 2. Interactive GUI monitor

Real-time desktop interface for manual testing and controller development. Runs the simulation at 1 kHz, shows live plots of both joints, an embedded 3D view, and lets you send PWM or voltage commands interactively.

```bash
python -m gui
```

The window has two panels:
- **Left**: rolling 5-second plots of pendulum angle (0° = upright, ±180° = hanging) and arm cumulative position.
- **Right**: offscreen 3D render, PWM/voltage set-and-apply controls, hold-to-move jog buttons, a reset button, and an **External Controller** section.

**Testing a custom controller**

The External Controller section lets you load any Python script that defines a `Controller` class. Your script receives raw encoder values at 1 kHz and returns a PWM command — no project imports needed.

1. In the **External Controller** panel, type the path to your script or click `···` to browse.
2. Set the **Perturbation from upright** angle (2–5° is a reasonable starting point).
3. Click **Start control**. The simulation resets with the pendulum at that angle and your controller takes over.
4. Click **Stop control** to return to manual mode.

The module is re-imported on every Start, so you can edit your script and click Start again without restarting the GUI.

See [docs/custom-controller-tutorial.md](docs/custom-controller-tutorial.md) for the full interface contract, sensor reference, and worked examples. Ready-to-run examples are in `controllers/`.

Requires a display. No CLI arguments.

---

### 3. Run a sim-to-real comparison from a real CSV

You have a CSV file from a real experiment and want to compare it against the simulation.

```bash
python tools/compare_step_response.py \
    --config configs/sim_to_real_validation.toml \
    --real <path_to_csv>
```

The CSV must have columns `t_us, motor_enc, pend_enc, pwm` logged at 1 kHz (standard firmware output). The script extracts the step duration and amplitude from the `pwm` column automatically.

Output image: `data/step_1023_100_comparacion.png`

The plot shows three panels: arm position, pendulum free oscillation with envelopes, and a table with wn and zeta for real vs. sim.

---

### 4. Train an RL agent

```bash
python -m agents.trainer
```

This creates a new run directory at `results/run_N/` and starts training with the default PPO configuration. You will see SB3 progress output in the terminal.

To customize agent type, timesteps, or task:

```python
# in a script or notebook
from agents.trainer import RLTrainer

trainer = RLTrainer(
    agent_type="PPO",       # "PPO", "SAC", or "A2C"
    task="equilibrium",     # "equilibrium" or "swing_up"
    max_steps=2000,
    seed=42,
)
trainer.train(total_timesteps=500_000, eval_freq=10_000)
```

To train with hardware non-idealities (domain randomization for sim-to-real), pass a
`SimConfig` (see [Simulating hardware non-idealities](#simulating-hardware-non-idealities)):

```python
from gym_envs.sim_config import SimConfig

trainer = RLTrainer(
    agent_type="PPO",
    task="equilibrium",
    seed=42,
    sim_config=SimConfig.from_toml("configs/sim_config.toml"),
)
```

Results saved per run:
```
results/run_N/
    best_model.zip          best checkpoint by eval reward
    model_final.zip         model at end of training
    train_monitor.csv       episode rewards during training
    val_monitor.csv         episode rewards during evaluation
    learning_curve_train.png
    learning_curve_val.png
```

Monitor training live with TensorBoard:

```bash
tensorboard --logdir results/
```

---

### 5. Run a trained model

```bash
python -m agents.predict \
    --model-path results/run_N/best_model.zip \
    --xml-file models/pendulum_model_v3.xml \
    --agent PPO \
    --task equilibrium \
    --episodes 3
```

A MuJoCo viewer opens and runs the specified number of episodes. Episode rewards are printed at the end of each one.

---

### 6. Run the raw simulation (no viewer, CSV + plot output)

Runs any TOML config through the full simulation pipeline and saves CSV and plot.

```bash
python tools/characterize_system.py --config configs/sim_to_real_validation.toml
```

Output paths are defined in the `[output]` section of the TOML.

---

### Common issues

**`ModuleNotFoundError` on any script**: make sure the venv is activated (`source .venv/bin/activate`) and you are running from the repo root.

**Viewer does not open / crashes immediately**: `mujoco_viewer` requires a display. On a headless server use the raw simulation command (step 5) instead.

**Training is slow**: `MlpPolicy` runs on CPU by default. This is intentional since small MLP policies are faster on CPU than on GPU. If you switch to a CNN policy, MPS (Apple Silicon) or CUDA will be used automatically.

**Results directory is missing**: it is created automatically on the first training run. Do not create it manually.

---

## Reference

### Usage summary

| Task | Command |
|---|---|
| Interactive GUI monitor | `python -m gui` |
| Watch simulation | `python tools/visualize_step_response.py --speed 3.0` |
| Sim-to-real comparison | `python tools/compare_step_response.py --config configs/sim_to_real_validation.toml --real <csv>` |
| Raw simulation (headless) | `python tools/characterize_system.py --config configs/sim_to_real_validation.toml` |
| Train agent | `python -m agents.trainer` |
| Run trained model | `python -m agents.predict --model-path results/run_N/best_model.zip --agent PPO --task equilibrium --xml-file models/pendulum_model_v3.xml` |
| TensorBoard | `tensorboard --logdir results/` |
| Validate environment (random) | `python scripts/random_episode_test.py --xml models/pendulum_model_v3.xml` |
| Validate environment (12 V) | `python scripts/max_voltage_test.py --xml models/pendulum_model_v3.xml` |

---

## Documentation

| Document | Description |
|---|---|
| [Writing a custom controller](docs/custom-controller-tutorial.md) | How to write and load your own controller script in the GUI monitor |

---

## Repository structure

```
configs/          TOML configs: sim_to_real_validation.toml (characterization) + sim_ideal/sim_config (SimConfig profiles)
controllers/      Controller scripts (pid_example.py example, pid_balance.py LQR balance)
data/             Experiment output data (STEP_1023_100 CSVs, plots, videos)
docs/             Guides and tutorials
gui/              Real-time desktop GUI package (python -m gui)
  ring_buffer.py  RingBuffer: pre-allocated circular buffer for plot data
  workers.py      StateSnapshot, SimWorker (1 kHz), RenderWorker (30 Hz offscreen)
  main_window.py  MainWindow: Qt UI, control panel, plot refresh
gym_envs/         Backend, observation encoder, and RL environment (see gym_envs/CLAUDE.md)
  backend.py      PendulumBackend Protocol + SensorReading (the firmware contract)
  pendulum_sim.py PendulumSim: MuJoCo backend with optional sensor non-idealities
  observation.py  ObservationEncoder: raw counts to observation (shared sim/deploy)
  sim_config.py   SimConfig dataclass (hardware non-idealities)
  pendulum_env.py PendulumEnv: gym.Env composition adapter
agents/           RL training (RLTrainer) and inference (predict.py)
models/           pendulum_model_v3.xml — the only MuJoCo model
tools/            Characterization and analysis scripts
  characterize_system.py     Parameter ID: TOML config → CSV/plot/video
  compare_step_response.py   Sim-vs-real comparison with log-decrement analysis
  visualize_step_response.py Interactive 3D viewer for STEP_1023_100
scripts/          Headless validation scripts (random_episode_test.py, max_voltage_test.py, validate_controller.py)
results/          Training runs (run_N/)
```

Empirical test data lives outside this repository at `/Users/felipe/Documents/Tesis/Informe-Final/notes/`.
