"""
Static control-loop parameters for the classical controllers.

Mirrors SimConfig: a dataclass whose defaults reproduce the original behaviour,
loaded from a TOML [control_config] section via from_toml(). The two knobs are the
control sampling rate and the derivative-filter cutoff; the EMA alpha is derived
from both so the cutoff stays invariant when the sampling rate changes.

    alpha = 1 - exp(-2*pi*fc/fs)

Defaults (1000 Hz, 26 Hz) give alpha = 0.1507 and a 1000 us period, i.e. control
every physics tick (no decimation). On the ESP32 these become compile-time
constants; here they are read once at controller construction.
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
