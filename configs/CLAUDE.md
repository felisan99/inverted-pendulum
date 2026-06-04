# configs/ — two unrelated families of TOML

This directory holds two distinct config types. Do not mix their schemas.

## 1. Characterization configs (for `characterize_system.py`)

Files like `step_1023_100_sim.toml`, `characterization_step.toml`. Sections:
`[simulation]`, `[initial_conditions]`, `[friction]`, `[motor]`, `[input]`,
`[output]`. Loaded by `mujoco_sim/characterize_system.py` (and the comparison /
visualizer scripts) via `tomllib` into a plain dict. They patch the XML model in
memory and run a parameterized open-loop simulation. They do NOT touch the RL env.

`step_1023_100_sim.toml` is the sim-to-real ground truth. Its
`joint1_frictionloss = 0.02` is intentional; see the root `CLAUDE.md`.

## 2. SimConfig profiles (for `gym_envs/sim_config.py`)

Files: `sim_ideal.toml`, `sim_realistic.toml`, and the conventional
`sim_config.toml`. They have a single `[sim_config]` section parsed by
`SimConfig.from_toml()` into a `SimConfig` dataclass. They parameterize sensor
non-idealities (noise, latency, timing jitter) for `PendulumSim` / `PendulumEnv`.

- `sim_ideal.toml` / `sim_realistic.toml` are versioned references.
- `sim_config.toml` is the local active profile the GUI loads by convention if it
  exists. It is meant to be edited locally; treat it as disposable.

See `gym_envs/CLAUDE.md` for how SimConfig is applied.
