# Furuta Pendulum — RL + Sim-to-Real

MuJoCo simulation and reinforcement learning environment for a Furuta pendulum. The simulation model is validated against real hardware; see [CLAUDE.md](CLAUDE.md) for physical parameters and validation results.

## Installation

```bash
uv sync
source .venv/bin/activate
```

All commands below are run from the repository root with the venv active.

## Commands

| Task | Command |
|---|---|
| Interactive GUI | `python -m gui` |
| Watch step response | `python tools/visualize_step_response.py --speed 3.0` |
| Sim-to-real comparison | `python tools/compare_step_response.py --real <csv>` |
| Raw simulation (headless) | `python tools/characterize_system.py --config configs/sim_to_real_validation.toml` |
| Train agent | `python -m agents.trainer` |
| Run trained model | `python -m agents.predict --model-path results/run_N/best_model.zip --agent PPO --task equilibrium --xml-file models/pendulum_high_quality.xml` |
| TensorBoard | `tensorboard --logdir results/` |
| Validate environment (random) | `python scripts/random_episode_test.py --xml models/pendulum_high_quality.xml` |
| Validate environment (12 V) | `python scripts/max_voltage_test.py --xml models/pendulum_high_quality.xml` |
| Run tests | `pytest test/` |

## Repository structure

```
configs/          TOML configs: sim_to_real_validation.toml, sim_ideal.toml, sim_config.toml
controllers/      Controller scripts (pid_example.py, pid_balance.py)
data/             Experiment output (STEP_1023_100 CSVs, plots, videos)
docs/             Guides and tutorials
gui/              Real-time desktop GUI package (python -m gui)
gym_envs/         Backend, observation encoder, and RL environment
agents/           RL training (RLTrainer) and inference (predict.py)
models/           pendulum_high_quality.xml (shaders), pendulum_low_quality.xml (fast render)
tools/            Characterization and analysis scripts
scripts/          Headless validation scripts
results/          Training runs (run_N/)
```

## Documentation

| Document | Description |
|---|---|
| [CLAUDE.md](CLAUDE.md) | Architecture, physical parameters, sim-to-real validation, GUI threading model |
| [gym_envs/CLAUDE.md](gym_envs/CLAUDE.md) | Detailed design of the backend/encoder/env layers |
| [docs/custom-controller-tutorial.md](docs/custom-controller-tutorial.md) | How to write and load a custom controller in the GUI |
