"""
Static control-loop parameters for the classical controllers.

Mirrors SimConfig: a dataclass whose defaults reproduce the original behaviour,
loaded from a TOML [control_config] section via from_toml(). The knobs are the
control sampling rate, the derivative-filter cutoff, and the Savitzky-Golay window
length; the EMA alpha is derived from rate and cutoff so the cutoff stays invariant
when the sampling rate changes.

    alpha = 1 - exp(-2*pi*fc/fs)

Defaults (1000 Hz, 26 Hz, window 21) reproduce the original per-tick behaviour. The
window only matters for the Savitzky-Golay velocity estimator (controllers/balance_SG.py);
when the control rate is lowered for the serial link, the window is shrunk so the
estimator's group delay ((N-1)/2 * period) stays bounded. On the ESP32 these become
compile-time constants; here they are read once at controller construction.
"""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ControlConfig:
    sample_freq_hz: float = 1000.0     # control rate (zero-order hold); capped by the 1 kHz physics rate
    filter_cutoff_hz: float = 26.0     # EMA derivative-filter cutoff; alpha is derived from fc and fs
    sg_window: int = 21                # Savitzky-Golay velocity window (samples); group delay = (N-1)/2 * period

    @classmethod
    def from_toml(cls, path: str | Path) -> "ControlConfig":
        with Path(path).open("rb") as f:
            data = tomllib.load(f).get("control_config", {})
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @property
    def sample_period_us(self) -> float:
        return 1_000_000.0 / self.sample_freq_hz

    @property
    def ema_alpha(self) -> float:
        return 1.0 - math.exp(-2.0 * math.pi * self.filter_cutoff_hz / self.sample_freq_hz)
