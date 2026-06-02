# Furuta Pendulum — RL + Sim-to-Real

MuJoCo simulation and reinforcement learning environment for a Furuta pendulum (rotary inverted pendulum). The simulation model is validated against real hardware.

## Installation

```bash
uv sync
source .venv/bin/activate
```

## Sim-to-real validation

The model `mujoco_sim/xml_models/pendulum_model_v3.xml` is the only model in the repository. All parameters come from empirical measurements on the physical system.

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
| Actuator gain | gainprm | 0.2116 N·m/V | Kt,sal / Rm = 1.092 / 5.16 |
| Back-EMF damping | biasprm[2] | -0.2311 N·m·s/rad | -Kt,sal² / Rm |
| Control range | ctrlrange | ±12 V | Supply voltage |
| Stall torque | forcerange | ±2.540 N·m | Kt,sal * Vmax / Rm |
| Terminal resistance | Rm | 5.16 Ω | Stall current experiment (range 4.8–5.5 Ω) |
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
python mujoco_sim/visualizar_step_100.py --speed 3.0
```

- Close the 3D window to end the simulation early.
- Change `--speed` to adjust playback rate. Use `--speed 1.0` for real time.
- The comparison plot opens automatically when the simulation finishes.

---

### 2. Interactive GUI monitor

Real-time desktop interface for manual testing. Runs the simulation at 1 kHz, shows live plots of both joints, an embedded 3D view, and lets you send PWM or voltage commands interactively.

```bash
python scripts/gui_monitor.py
```

The window has two panels:
- **Left**: rolling 5-second plots of pendulum angle (0° = upright, ±180° = hanging) and arm cumulative position.
- **Right**: offscreen 3D render, PWM/voltage set-and-apply controls, hold-to-move jog buttons, and a reset button.

Requires a display. No CLI arguments.

---

### 3. Run a sim-to-real comparison from a real CSV

You have a CSV file from a real experiment and want to compare it against the simulation.

```bash
python mujoco_sim/analisis_step_100_comparacion.py \
    --config configs/step_1023_100_sim.toml \
    --real <path_to_csv>
```

The CSV must have columns `t_us, motor_enc, pend_enc, pwm` logged at 1 kHz (standard firmware output). The script extracts the step duration and amplitude from the `pwm` column automatically.

Output image: `results/characterization/step_1023_100_comparacion.png`

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
    --xml-file mujoco_sim/xml_models/pendulum_model_v3.xml \
    --agent PPO \
    --task equilibrium \
    --episodes 3
```

A MuJoCo viewer opens and runs the specified number of episodes. Episode rewards are printed at the end of each one.

---

### 6. Run the raw simulation (no viewer, CSV + plot output)

Runs any TOML config through the full simulation pipeline and saves CSV and plot.

```bash
python -m mujoco_sim.characterize_system --config configs/step_1023_100_sim.toml
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
| Interactive GUI monitor | `python scripts/gui_monitor.py` |
| Watch simulation | `python mujoco_sim/visualizar_step_100.py --speed 3.0` |
| Sim-to-real comparison | `python mujoco_sim/analisis_step_100_comparacion.py --config configs/step_1023_100_sim.toml --real <csv>` |
| Raw simulation (headless) | `python -m mujoco_sim.characterize_system --config configs/step_1023_100_sim.toml` |
| Train agent | `python -m agents.trainer` |
| Run trained model | `python -m agents.predict --model-path results/run_N/best_model.zip --agent PPO --task equilibrium --xml-file mujoco_sim/xml_models/pendulum_model_v3.xml` |
| TensorBoard | `tensorboard --logdir results/` |
| Validate environment (random) | `python scripts/random_episode_test.py --xml mujoco_sim/xml_models/pendulum_model_v3.xml` |
| Validate environment (12 V) | `python scripts/max_voltage_test.py --xml mujoco_sim/xml_models/pendulum_model_v3.xml` |

---

## Repository structure

```
configs/          TOML configuration files for characterize_system.py
gym_envs/         Gymnasium environment (PendulumEnv)
agents/           RL training (RLTrainer) and inference (predict.py)
mujoco_sim/       Simulation scripts and XML model
  xml_models/     pendulum_model_v3.xml — the only model
scripts/          gui_monitor.py (interactive GUI), random_episode_test.py, max_voltage_test.py
utils/            Learning curve plots
results/          Training runs (run_N/) and characterization outputs
```

Empirical test data lives outside this repository at `/Users/felipe/Documents/Tesis/Informe-Final/notes/`.
