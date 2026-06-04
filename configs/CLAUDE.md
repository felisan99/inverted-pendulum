# configs/ — three unrelated families of TOML

This directory holds three distinct config types. Do not mix their schemas.

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

## 3. ControlConfig profiles (for `gym_envs/control_config.py`)

Files: `control_config.toml` (active) and `control_250hz.toml` (reference). They
have a single `[control_config]` section parsed by `ControlConfig.from_toml()`.
Two knobs: `sample_freq_hz` (control rate with zero-order hold; capped by the 1 kHz
physics rate) and `filter_cutoff_hz` (EMA derivative-filter cutoff; the alpha is
derived as `1 - exp(-2*pi*fc/fs)`, so the cutoff stays fixed when the rate changes).
Loaded by the classical controllers (`controllers/pid_balance.py`) at construction.

- `control_config.toml` is the local active profile; defaults (1000 Hz, 26 Hz)
  reproduce the original 1 kHz behaviour.
- `control_250hz.toml` is a versioned reference; copy it over `control_config.toml`
  to run the loop at 250 Hz.

Independent of the characterization configs and the SimConfig profiles: these only
shape the classical control loop, not the XML model or sensor non-idealities.
