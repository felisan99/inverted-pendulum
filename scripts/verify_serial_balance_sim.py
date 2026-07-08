"""
Sim verification for the serial-link balance loop (plan Task 1c).

Reproduces, in simulation, the PC<->ESP32 serial control path that balance_SG runs
over: the host does not see a clean uniform 1 kHz stream. It polls the firmware's
1 kHz snapshot once per serial round-trip (~1.6 ms, with OS/USB jitter), so the
samples that reach the controller are delayed and non-uniformly spaced. This is the
condition a fixed-dt Savitzky-Golay kernel mis-handles (mis-scaled, noisy velocity
-> PWM saturation), and that the real-timestamp least-squares slope is meant to fix.

Model per host iteration:
  - advance the 1 kHz physics by a jittered round-trip (default 1.6 +/- 0.4 ms),
    holding the last PWM (zero-order hold, as the firmware does between commands);
  - read the latest snapshot, delayed by `latency_steps` (the round-trip lag);
  - call controller.compute(); its internal ZOH decimates to the control period.

We sweep the control period (250 Hz / 4 ms vs 200 Hz / 5 ms) and the SG window N,
and report whether the pole survives the run and how hard the PWM works
(saturation fraction, sign-change rate, RMS). A clean balance = survives full
duration with low saturation and few sign changes. The 1 kHz uniform baseline
(N=21) is a backward-compatibility check on the real-dt slope.

Run:
    python scripts/verify_serial_balance_sim.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gym_envs.observation import pend_to_rad
from gym_envs.pendulum_sim import PWM_MAX, PendulumSim
from gym_envs.sim_config import SimConfig

import controllers.balance_SG as balance_SG

_FALL_RAD = math.radians(25.0)


def run_trial(
    *,
    period_ms: float,
    window: int,
    duration_s: float = 12.0,
    init_deg: float = 5.0,
    rt_mean_ms: float = 1.6,
    rt_jitter_ms: float = 0.4,
    latency_steps: int = 2,
    pend_noise_sigma: float = 0.16,
    seed: int = 0,
) -> dict:
    cfg = SimConfig(pend_noise_sigma=pend_noise_sigma, sensor_latency_steps=latency_steps)
    sim = PendulumSim(sim_config=cfg, seed=seed)
    reading = sim.reset(initial_angle_rad=math.radians(init_deg))

    ctrl = balance_SG.Controller()
    ctrl._period_us = period_ms * 1000.0
    ctrl._window = window
    ctrl.reset()

    rng = np.random.default_rng(seed + 1)
    pwm = 0
    pwms: list[int] = []
    max_phi = 0.0
    t = 0.0
    survived = duration_s

    while t < duration_s:
        pwm = int(max(-PWM_MAX, min(PWM_MAX, ctrl.compute(reading.pend_enc, reading.motor_enc, reading.t_us))))
        pwms.append(pwm)

        n_steps = max(1, int(round(rt_mean_ms + rng.normal(0.0, rt_jitter_ms))))
        for _ in range(n_steps):
            reading = sim.step(pwm)
        t += n_steps * 1e-3

        phi = abs(pend_to_rad(reading.pend_enc))
        max_phi = max(max_phi, phi)
        if phi > _FALL_RAD:
            survived = t
            break

    sim.close()

    arr = np.asarray(pwms, dtype=float)
    sat = float(np.mean(np.abs(arr) >= PWM_MAX - 3)) if arr.size else 0.0
    sign_changes = int(np.sum(np.abs(np.diff(np.sign(arr))) > 1)) if arr.size > 1 else 0
    sign_rate = sign_changes / max(survived, 1e-9)
    rms = float(np.sqrt(np.mean(arr**2))) if arr.size else 0.0

    return {
        "period_ms": period_ms,
        "window": window,
        "survived_s": survived,
        "balanced": survived >= duration_s - 1e-6,
        "sat_frac": sat,
        "sign_per_s": sign_rate,
        "rms_pwm": rms,
        "max_deg": math.degrees(max_phi),
    }


def _fmt(r: dict) -> str:
    ok = "OK " if r["balanced"] else "FALL"
    return (f"  {ok}  period={r['period_ms']:.0f}ms  N={r['window']:>2}  "
            f"survived={r['survived_s']:5.2f}s  sat={r['sat_frac']*100:4.1f}%  "
            f"sign/s={r['sign_per_s']:5.1f}  rms={r['rms_pwm']:6.1f}  maxdeg={r['max_deg']:4.1f}")


def main() -> None:
    print("Baseline: 1 kHz uniform, no jitter/latency (backward-compat of real-dt slope)")
    print(_fmt(run_trial(period_ms=1.0, window=21, rt_mean_ms=1.0, rt_jitter_ms=0.0,
                         latency_steps=0, pend_noise_sigma=0.16)))

    print("\nSerial path (round-trip 1.6+/-0.4 ms, latency 2 steps, noise 0.16 c), N sweep:")
    for period_ms in (4.0, 3.0, 5.0):
        print(f" control period {period_ms:.0f} ms  ({1000/period_ms:.0f} Hz):")
        for window in (5, 7, 9, 11, 13, 15):
            print(_fmt(run_trial(period_ms=period_ms, window=window)))

    print("\nWorst-case serial (round-trip 2.0+/-0.8 ms), 4 ms control, N sweep:")
    for window in (5, 7, 9, 11):
        print(_fmt(run_trial(period_ms=4.0, window=window, rt_mean_ms=2.0, rt_jitter_ms=0.8, latency_steps=3)))


if __name__ == "__main__":
    main()
