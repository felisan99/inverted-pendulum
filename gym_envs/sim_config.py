"""
Non-ideal simulation parameters.

Each field defaults to its ideal value (zero effect), so SimConfig() reproduces
the noise-free behaviour. Load a profile from a TOML file with a [sim_config]
section via SimConfig.from_toml().
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SimConfig:
    pend_noise_sigma: float = 0.0      # Gaussian sigma on pend_enc  [AS5600 counts]
    motor_noise_sigma: float = 0.0     # Gaussian sigma on motor_enc [Hall counts]
    sensor_latency_steps: int = 0      # pure delay steps on the sensor reading
    dt_jitter_sigma: float = 0.0       # per-interval jitter sigma on t_us [microseconds]

    @classmethod
    def from_toml(cls, path: str | Path) -> "SimConfig":
        with Path(path).open("rb") as f:
            data = tomllib.load(f).get("sim_config", {})
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
