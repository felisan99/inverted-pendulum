# How to write a custom controller

The GUI monitor (`scripts/gui_monitor.py`) can load any Python script that follows a simple interface. This lets you test PID, LQR, bang-bang, or any other strategy without modifying the GUI itself.

---

## The interface contract

Your script must define a class named `Controller` with two methods:

```python
class Controller:

    def reset(self) -> None:
        """Called once when the user clicks Start control."""
        ...

    def compute(self, pend_enc: int, motor_enc: int, t_us: int) -> int:
        """
        Called at ~1 kHz. Must return a PWM value in [-1023, +1023].

        pend_enc  -- pendulum absolute position, 0-4095 (AS5600 12-bit encoder)
                     0    = upright (unstable equilibrium)
                     2048 = hanging straight down
        motor_enc -- arm cumulative position in Hall counts (signed, unbounded)
                     1716 counts = one full revolution at the output shaft
        t_us      -- simulation time in microseconds
        """
        ...
```

That is all. No imports from this project are required, no base class to inherit from.

---

## Scripts vs. trained models

The GUI has two separate ways to drive the pendulum:

- **External Controller** (this document): an arbitrary `.py` script with a `Controller`
  class. It lives entirely outside the repo and imports nothing from the project. You own the
  feature extraction (raw counts in, PWM out).
- **Trained Model**: an SB3 `.zip` produced by `agents/trainer.py`. Pick the algorithm
  (PPO/SAC/A2C) and click **Start model**. The GUI wraps it in `gym_envs/policy_controller.py`
  (`ModelController`), which feeds the network through the project's `ObservationEncoder`, the
  same feature extractor used during training. You do **not** write any glue code, and you must
  not reimplement the encoder; that is exactly what guarantees the network sees the observation
  it was trained on (sim-to-real consistency).

Both routes are mutually exclusive (one controller at a time) and both respect the
**Disturbance** controls, so you can hit the pendulum and watch either recover.

---

## Sensor quick-reference

| Value | Meaning |
|---|---|
| `pend_enc == 0` | Pendulum perfectly upright |
| `pend_enc == 2048` | Pendulum hanging down |
| `pend_enc` in `(0, 1024)` | Tilted toward positive side |
| `pend_enc` in `(3072, 4095)` | Tilted toward negative side (wraps around 4096) |
| `motor_enc > 0` | Arm has turned clockwise from its starting position |
| `motor_enc < 0` | Arm has turned counter-clockwise |

To convert to radians: `angle = pend_enc * 2 * pi / 4096`.
Values above `pi` are on the "negative" side: `if angle > pi: angle -= 2 * pi`.

---

## A minimal example — threshold controller

This controller does not use any math library. It reads `pend_enc` directly and decides what to do with a few `if` statements.

The idea: if the pendulum is leaning to one side, push the arm in the opposite direction to bring it back.

```python
# controllers/threshold_example.py

_DEAD_ZONE    = 50    # counts around upright where we do nothing (~4.4 deg)
_STRONG_PWM   = 600   # PWM applied when far from upright
_WEAK_PWM     = 200   # PWM applied when close but outside dead zone
_FAR_THRESHOLD = 400  # counts: above this we consider the pendulum "far"


class Controller:

    def reset(self) -> None:
        pass  # no state to initialize

    def compute(self, pend_enc: int, motor_enc: int, t_us: int) -> int:
        # Remap pend_enc so that 0 = upright and negative values are on the
        # left side. Values 0-2047 are rightward, values 2048-4095 are leftward
        # but we remap those to negative numbers.
        if pend_enc > 2048:
            error = pend_enc - 4096   # e.g. 3900 becomes -196
        else:
            error = pend_enc          # e.g. 200 stays 200

        # Dead zone: close enough to upright, do nothing.
        if abs(error) < _DEAD_ZONE:
            return 0

        # Determine direction: push against the lean.
        direction = 1 if error > 0 else -1

        # Strength depends on how far the pendulum has tilted.
        if abs(error) > _FAR_THRESHOLD:
            pwm = _STRONG_PWM
        else:
            pwm = _WEAK_PWM

        return direction * pwm
```

Save this file anywhere on your machine, then load it from the GUI.

---

## Loading your controller in the GUI

1. Run `python scripts/gui_monitor.py`.
2. In the **External Controller** section on the right panel, type the path to your script (or click `···` to browse).
3. Set the **Perturbation from upright** angle — how many degrees off-center the pendulum starts. A small value (2-5°) gives the controller a chance to recover; a large value is a harder challenge.
4. Click **Start control**. The simulation resets with the pendulum at that angle and your `compute()` takes over immediately.
5. Click **Stop control** to hand back manual control (PWM goes to 0).

The **Status** label shows:
- `Idle` — no controller active
- `Active` — your controller is running
- `Error: <message>` — `compute()` raised an exception; control stopped automatically and PWM was set to 0

---

## Tips

**Keeping state between calls**

`reset()` is the place to initialize any variables that need to persist across `compute()` calls (integrals, previous errors, filters, etc.):

```python
def reset(self) -> None:
    self._previous_error = 0
    self._sum = 0.0
```

**Computing elapsed time**

`t_us` is the simulation clock in microseconds. Use it to compute `dt` instead of assuming a fixed 1 ms:

```python
def compute(self, pend_enc, motor_enc, t_us):
    if self._last_t is not None:
        dt = (t_us - self._last_t) * 1e-6   # seconds
    else:
        dt = 0.001
    self._last_t = t_us
    ...
```

**Clamping output**

`compute()` must return an integer. Values outside `[-1023, +1023]` are clamped by the simulator, but it is cleaner to clamp explicitly:

```python
return int(max(-1023, min(1023, raw_pwm)))
```

**Crashing safely**

If `compute()` raises any exception the GUI catches it, stops the controller, and displays the error message. You do not need to add try/except inside `compute()` unless you want to handle specific cases yourself.

**Reloading after edits**

Click **Stop control**, edit your file, then click **Start control** again. The module is re-imported on every Start, so changes take effect immediately without restarting the GUI.
