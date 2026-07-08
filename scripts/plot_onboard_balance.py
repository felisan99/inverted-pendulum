"""
Plot results from data/onboard_balance.csv (recorded by record_onboard_telemetry.py).

Four figures:
  1. Pendulum angle vs time
  2. PWM output and its term contributions
  3. Arm position and arm angular velocity
  4. Pendulum angle and pendulum angular velocity

Usage:
    python scripts/plot_onboard_balance.py
    python scripts/plot_onboard_balance.py --csv data/onboard_balance.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parent.parent

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        "#e8e8e8",
    "grid.linewidth":    0.6,
    "font.size":         10,
    "axes.labelsize":    10,
    "legend.fontsize":   9,
    "legend.frameon":    False,
    "lines.linewidth":   0.9,
})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, default=_ROOT / "data" / "onboard_balance.csv")
    args = p.parse_args()

    rows = []
    with args.csv.open() as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items()})

    if not rows:
        print("CSV is empty.")
        return

    # Trim to armed window.
    armed_mask = [r["armed"] > 0 for r in rows]
    if any(armed_mask):
        start = next(i for i, v in enumerate(armed_mask) if v)
        rows = rows[start:]

    t             = np.array([r["t_us"]           for r in rows]) * 1e-6
    t            -= t[0]
    phi_deg       = np.array([r["phi_deg"]         for r in rows])
    theta_arm_deg = np.array([r["theta_arm_deg"]   for r in rows])
    phi_d         = np.array([r["phi_d_rad_s"]     for r in rows])
    omega_arm     = np.array([r["omega_arm_rad_s"] for r in rows])
    pwm           = np.array([r["pwm"]             for r in rows])
    term_p        = np.array([r["term_p"]          for r in rows])
    term_d        = np.array([r["term_d"]          for r in rows])
    term_b        = np.array([r["term_b"]          for r in rows])
    # term_a is not in telemetry but can be recomputed: -KA * theta_arm_rad
    KA            = -50.0
    term_a        = -KA * (theta_arm_deg * np.pi / 180.0)

    # ── Figure 1: pendulum angle ────────────────────────────────────────────
    fig1, ax = plt.subplots(figsize=(10, 3))
    fig1.suptitle("Pendulum angle", x=0.12, ha="left", fontsize=11, fontweight="bold")
    ax.plot(t, phi_deg, color="#2979FF", lw=0.9)
    ax.axhline(0, color="#E53935", lw=0.8, ls="--", label="setpoint")
    ax.set_ylim(-15, 15)
    ax.set_ylabel("angle (deg)")
    ax.set_xlabel("time (s)")
    ax.legend()
    fig1.tight_layout()

    # ── Figure 2: PWM and contributions ────────────────────────────────────
    fig2, ax = plt.subplots(figsize=(10, 4))
    fig2.suptitle("PWM output", x=0.12, ha="left", fontsize=11, fontweight="bold")
    ax.plot(t, term_p, color="#64B5F6", lw=0.8, label="term_p  (−Kp·φ)")
    ax.plot(t, term_d, color="#FFB74D", lw=0.8, label="term_d  (−Kd·φ̇)")
    ax.plot(t, term_b, color="#81C784", lw=0.8, label="term_b  (−Kb·ω_arm)")
    ax.plot(t, term_a, color="#CE93D8", lw=0.8, label="term_a  (−Ka·θ_arm)")
    ax.plot(t, pwm,    color="#212121", lw=1.2, label="PWM total")
    sat = int(np.max(np.abs(pwm)))
    ax.text(0.99, 0.97, f"peak |PWM| {sat}  (sat ±1023)",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color="#888888")
    ax.set_ylabel("PWM (10-bit)")
    ax.set_xlabel("time (s)")
    ax.legend(ncol=2)
    fig2.tight_layout()

    # ── Figure 3: arm position and arm velocity ─────────────────────────────
    fig3, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 4.5),
                                           sharex=True, gridspec_kw={"hspace": 0.08})
    fig3.suptitle("Arm state", x=0.12, ha="left", fontsize=11, fontweight="bold")
    ax_top.plot(t, theta_arm_deg, color="#AB47BC", lw=0.9)
    ax_top.set_ylabel("position (deg)")
    ax_top.tick_params(labelbottom=False)

    ax_bot.plot(t, omega_arm, color="#EC407A", lw=0.9)
    ax_bot.axhline(0, color="#bbbbbb", lw=0.6, ls="--")
    ax_bot.set_ylabel("velocity (rad/s)")
    ax_bot.set_xlabel("time (s)")
    fig3.tight_layout()

    # ── Figure 4: pendulum angle and pendulum velocity ──────────────────────
    fig4, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 4.5),
                                           sharex=True, gridspec_kw={"hspace": 0.08})
    fig4.suptitle("Pendulum state", x=0.12, ha="left", fontsize=11, fontweight="bold")
    ax_top.plot(t, phi_deg, color="#2979FF", lw=0.9)
    ax_top.axhline(0, color="#E53935", lw=0.8, ls="--")
    ax_top.set_ylim(-15, 15)
    ax_top.set_ylabel("angle (deg)")
    ax_top.tick_params(labelbottom=False)

    ax_bot.plot(t, phi_d, color="#FF7043", lw=0.9)
    ax_bot.axhline(0, color="#bbbbbb", lw=0.6, ls="--")
    ax_bot.set_ylabel("velocity (rad/s)")
    ax_bot.set_xlabel("time (s)")
    fig4.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
