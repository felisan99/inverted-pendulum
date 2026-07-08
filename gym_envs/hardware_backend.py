"""
`HardwareBackend`: the real-ESP32 twin of `PendulumSim`.

It satisfies the same `PendulumBackend` Protocol (`step`/`reset`/`close`) as the
simulator, so any code written against the backend contract (a hand controller, the
runner in `scripts/run_hardware_balance.py`, or `PendulumEnv`) can drive the real
rig with no changes. The ESP32 runs the firmware in serial-slave mode: the host
sends a PWM command, the firmware replies with a raw `SensorReading`. Control logic
(e.g. `controllers/balance_SG.py`) stays on the host; this class is pure I/O.

The transport is injectable (`transport=`) so the protocol and backend can be tested
without hardware or even without pyserial installed.
"""

from __future__ import annotations

import time

from gym_envs.backend import SensorReading
from gym_envs.serial_protocol import (
    CMD_RESET,
    CMD_STEP,
    PWM_MAX,
    encode_request,
    read_response,
)

_DEFAULT_BAUD = 921600
_BOOT_SETTLE_S = 0.3   # opening the port toggles DTR/RTS and reboots the ESP32


class HardwareBackend:
    """Serial link to the ESP32 firmware, implementing `PendulumBackend`."""

    def __init__(
        self,
        port: str | None = None,
        *,
        baudrate: int = _DEFAULT_BAUD,
        timeout: float = 0.1,
        transport=None,
    ) -> None:
        if transport is None:
            import serial  # lazy: only needed for a real link

            transport = serial.Serial(port, baudrate=baudrate, timeout=timeout)
            time.sleep(_BOOT_SETTLE_S)
            transport.reset_input_buffer()
        self._io = transport

    def reset(self, *args, **kwargs) -> SensorReading:
        """Zero the firmware clock (t_us) and motor-count baseline; return the first
        reading. Accepts and ignores sim-only kwargs (`pendulum_down`,
        `initial_angle_rad`): the physical pendulum starts wherever it physically is.
        """
        flush = getattr(self._io, "reset_input_buffer", None)
        if flush is not None:
            flush()
        return self._transact(CMD_RESET, 0)

    def step(self, pwm: int) -> SensorReading:
        pwm = int(max(-PWM_MAX, min(PWM_MAX, pwm)))
        return self._transact(CMD_STEP, pwm)

    def close(self) -> None:
        try:
            self._transact(CMD_STEP, 0)   # leave the motor stopped
        except Exception:
            pass
        self._io.close()

    def _transact(self, cmd: int, pwm: int) -> SensorReading:
        self._io.write(encode_request(cmd, pwm))
        return read_response(self._io)
