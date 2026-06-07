# gym_envs/ — simulation backend, observation encoding, and RL environment

This package is organized in three layers so that a policy trained in simulation
can later drive the real ESP32 without re-deriving any feature extraction. The
layers are deliberately decoupled by a Protocol and a shared encoder.

```
backend.py        PendulumBackend (Protocol) + SensorReading       (the contract)
   |
pendulum_sim.py   PendulumSim: MuJoCo + sensor non-idealities       (a backend impl)
   |
observation.py    ObservationEncoder: counts -> obs vector          (shared sim/deploy)
   |
pendulum_env.py   PendulumEnv(gym.Env): backend + encoder + reward   (training only)
```

## The contract: `backend.py`

`PendulumBackend` is a `typing.Protocol` (runtime_checkable). The contract is the
ESP32 firmware API: a signed 10-bit PWM goes in, a raw `SensorReading(t_us,
motor_enc, pend_enc)` comes out.

- `PendulumSim` satisfies it by duck typing (it does not inherit anything).
- A future `HardwareBackend` (link to the real ESP32) would satisfy the same
  contract, and could be injected into `PendulumEnv` unchanged.

`SensorReading` lives here (not in `pendulum_sim.py`) because it is the shared wire
type. Import it from `gym_envs.backend`.

## The plant: `pendulum_sim.py`

`PendulumSim` is the MuJoCo implementation of `PendulumBackend`. It mirrors the
firmware exactly: PWM in, raw encoder counts out, 1 kHz timestep. It is used by the
GUI monitor, the controller tests, and (indirectly) by `PendulumEnv`.

Non-ideal sensor effects are driven by a `SimConfig` (see below) and applied in
`_read_sensors()`:

- Gaussian noise added to `pend_enc` / `motor_enc` (in counts).
- Pure delay via a `deque` latency buffer (returns the oldest buffered reading).
- Timing jitter via an accumulating clock.

**Clock subtlety (important):** `t_us` comes from `self._clock_us`, which is
advanced in `step()` (not in `_read_sensors()`). This preserves the invariant that
`reset()` reports `t_us == 0` and the first `step()` reports the nominal dt. The
jitter is added per interval (`clock += dt_nominal + N(0, sigma)`, floored at 1 us),
so it is always monotonically increasing and never yields a non-positive dt. With
`SimConfig()` (all zeros) the behaviour is bit-for-bit the original ideal sim.

`PendulumSim` is the single owner of the human 3D viewer (`_render()`, throttled to
~100 Hz with real-time sync). `PendulumEnv` no longer renders.

## The feature extractor: `observation.py`

`ObservationEncoder` converts a `SensorReading` into the 6-dim policy observation:

```
[sin(motor), cos(motor), vel_motor, sin(pend), cos(pend), vel_pend]
```

This is the single source of truth for feature extraction. The exact same code runs
during training (inside `PendulumEnv`) and would run at deployment on the ESP32
host. Duplicating this logic is the main sim-to-real risk, so do not reimplement it
elsewhere; reuse this class.

- Velocities are finite differences over the previous reading, using the measured
  `dt` from `t_us` (so timing jitter is reflected in the velocity, matching the
  firmware which computes dt from timestamps).
- The pendulum angle is mapped to `[-pi, pi]` centered on the upright via
  `_pend_to_rad`. This keeps the velocity continuous near 0 (equilibrium). The wrap
  discontinuity sits at +-pi (hanging), so **swing_up velocity is not supported by
  this encoder** (known limitation). To support swing_up, unwrap the difference like
  the firmware `angularDistance`.
- `_MOTOR_LSB` and `_PENDULUM_LSB` live here. Import them from
  `gym_envs.observation`.

## The non-ideal config: `sim_config.py`

`SimConfig` is a dataclass; every field defaults to its ideal value (zero effect),
so `SimConfig()` is the noise-free sim. Fields: `pend_noise_sigma`,
`motor_noise_sigma` (counts), `sensor_latency_steps` (steps), `dt_jitter_sigma`
(microseconds, per interval). `SimConfig.from_toml(path)` reads a `[sim_config]`
TOML section.

Profiles live in `configs/`: `sim_ideal.toml` (versioned reference, all zeros). The
GUI loads `configs/sim_config.toml` by convention if it exists (local active profile,
edit freely).

## The control-loop config: `control_config.py`

`ControlConfig` mirrors `SimConfig` but parameterizes the classical control loop,
not the sim. Two fields: `sample_freq_hz` (control rate; the controller self-decimates
with a zero-order hold so the physics still runs at 1 kHz) and `filter_cutoff_hz` (EMA
derivative-filter cutoff). The EMA `alpha` is a derived property,
`1 - exp(-2*pi*fc/fs)`, so the cutoff is invariant when the rate changes. Defaults
(1000 Hz, 26 Hz) reproduce the original per-tick behaviour. `from_toml(path)` reads a
`[control_config]` section; profiles live in `configs/control_config.toml` (active) and
`configs/control_250hz.toml`. Loaded once by `controllers/pid_balance.py` at construction.

## The RL environment: `pendulum_env.py`

`PendulumEnv` is a `gym.Env` adapter, built by **composition** over a backend and an
encoder. It is NOT a subclass of `PendulumSim`.

Why composition, not inheritance:
- `step`/`reset` have incompatible return types between the two classes (gym 5-tuple
  vs `SensorReading`), so inheritance would violate Liskov and collide.
- `PendulumEnv` already must inherit `gym.Env`.
- Composition over the Protocol is what enables swapping `PendulumSim` for a
  `HardwareBackend` (the sim-to-real path).

Key points:
- Constructor accepts an optional `backend` (defaults to a fresh `PendulumSim`),
  plus `sim_config` and `seed` that are forwarded to the default backend.
- The action stays as voltage `[-12, 12]`. `step()` converts it to PWM and routes it
  through `backend.step(pwm)`, so the action passes through the same 10-bit
  quantization the ESP32 DAC has. The volt->PWM->volt round trip is intentional; it
  models the actuator resolution.
- `reset(seed)` forwards the seed to the backend (reproducible per-episode noise) and
  seeds the encoder from the first reading (initial velocity 0).
- `_compute_reward()` is named with a leading underscore on purpose: SB3 `check_env`
  treats a public `compute_reward` as a `GoalEnv` marker and fails. Do not rename it
  back.
- `step()` returns a Python `float` reward and a Python `bool` `terminated`. SB3
  `check_env` rejects numpy scalars, so keep the explicit `float(...)` / `bool(...)`
  casts.

At deployment there is no `PendulumEnv`: you use `HardwareBackend` +
`ObservationEncoder` + the trained network. Reward and episode logic are training-only.

## The deployment harness: `policy_controller.py`

`ModelController` is the concrete "ObservationEncoder + trained network -> PWM" path, with no
`PendulumEnv` around it. It wraps a Stable-Baselines3 model and exposes the GUI/firmware
controller contract (`reset()` + `compute(pend_enc, motor_enc, t_us) -> pwm`), so the GUI can run
a trained `.zip` against `PendulumSim` exactly like a hand-written controller. It reuses
`ObservationEncoder` (never reimplements it), which is what keeps the deployed features identical
to training. The encoder is bootstrapped on the **first `compute`** (not in `reset()`), because the
caller resets the plant after `reset()` and the first reading only arrives with the first
`compute`; that first observation carries velocity 0, matching `PendulumEnv.reset`. The
voltage->PWM conversion is the same one `PendulumEnv.step` uses, so the actuator quantization
matches too. This is the sim-side stand-in for what a real ESP32 deployment would do.

## Tests

- `test/test_pendulum_sim.py` — backend contract (imports `SensorReading` from
  `gym_envs.backend`).
- `test/test_env_smoke.py` — env contract; termination tests use a scripted
  `FakeBackend` (implements the Protocol) instead of poking MuJoCo internals.
- `test/test_pid_balance.py` — runs `controllers/pid_balance.py` against the ideal
  `PendulumSim`; guards that the equilibrium path is unchanged. Overrides the module's
  `_CONFIG_PATH` so the 1 kHz cases use `ControlConfig()` defaults and one case loads
  `configs/control_250hz.toml` to cover the zero-order-hold decimation path.
- `test/test_control_config.py` — `ControlConfig` alpha/period math and `from_toml()`.
