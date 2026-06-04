"""
Backend contract for the pendulum.

A backend takes a signed 10-bit PWM command and returns a raw SensorReading,
mirroring the ESP32 firmware API. PendulumSim implements it against MuJoCo; a
future HardwareBackend would implement it against the real ESP32 link. PendulumEnv
depends only on this Protocol, so the physics source can be swapped without
touching the RL code.
"""

from __future__ import annotations

from collections import namedtuple
from typing import Protocol, runtime_checkable

SensorReading = namedtuple("SensorReading", ["t_us", "motor_enc", "pend_enc"])


@runtime_checkable
class PendulumBackend(Protocol):
    def step(self, pwm: int) -> SensorReading: ...
    def reset(self, *args, **kwargs) -> SensorReading: ...
    def close(self) -> None: ...
