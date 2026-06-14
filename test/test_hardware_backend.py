"""
Protocol + HardwareBackend tests, exercised against an in-process firmware emulator.
No serial port or pyserial transport is opened.
"""

import pytest

from gym_envs.backend import PendulumBackend, SensorReading
from gym_envs.hardware_backend import HardwareBackend
from gym_envs.serial_protocol import (
    CMD_RESET,
    CMD_STEP,
    PWM_MAX,
    RESP_SYNC,
    ProtocolError,
    crc8,
    decode_request,
    encode_request,
    encode_response,
    read_response,
)


class FakeFirmware:
    """Serial-like object emulating the ESP32 slave: decodes each request and queues
    a reply, exactly as the real firmware would over the wire."""

    def __init__(self):
        self._rx = bytearray()
        self.requests = []
        self.flushed = 0
        self.closed = False
        self._t_us = 0
        self._motor = 0

    def write(self, data):
        cmd, pwm = decode_request(bytes(data))
        self.requests.append((cmd, pwm))
        if cmd == CMD_RESET:
            self._t_us = 0
            self._motor = 0
        else:
            self._t_us += 1000
            self._motor += pwm
        self._rx += encode_response(SensorReading(self._t_us, self._motor, 2048))
        return len(data)

    def read(self, n=1):
        chunk = bytes(self._rx[:n])
        del self._rx[:n]
        return chunk

    def reset_input_buffer(self):
        self.flushed += 1
        self._rx.clear()

    def close(self):
        self.closed = True


class BufReader:
    """Read-only serial-like wrapper over a fixed byte buffer."""

    def __init__(self, data):
        self._b = bytearray(data)

    def read(self, n=1):
        chunk = bytes(self._b[:n])
        del self._b[:n]
        return chunk


# ---------------------------------------------------------------------------
# CRC and framing
# ---------------------------------------------------------------------------

def test_crc8_known_vector():
    # CRC-8/SMBus of "123456789" is 0xF4 (standard check value).
    assert crc8(b"123456789") == 0xF4


def test_request_round_trip():
    for cmd, pwm in [(CMD_STEP, 0), (CMD_STEP, 1023), (CMD_STEP, -1023), (CMD_RESET, 0)]:
        assert decode_request(encode_request(cmd, pwm)) == (cmd, pwm)


def test_response_round_trip():
    reading = SensorReading(t_us=4949775000, motor_enc=-12345, pend_enc=4095)
    assert read_response(BufReader(encode_response(reading))) == reading


def test_t_us_survives_beyond_32bit():
    # The bug we fixed in the GUI: 32-bit overflow. The wire format is 64-bit.
    big = 5_000_000_000  # > 2**31
    reading = SensorReading(t_us=big, motor_enc=0, pend_enc=0)
    assert read_response(BufReader(encode_response(reading))).t_us == big


def test_read_response_resyncs_past_garbage():
    reading = SensorReading(t_us=1000, motor_enc=7, pend_enc=2048)
    noise = bytes([0x00, 0x12, RESP_SYNC, 0xFF])  # includes a false sync byte
    assert read_response(BufReader(noise + encode_response(reading))) == reading


def test_read_response_timeout_raises():
    with pytest.raises(ProtocolError):
        read_response(BufReader(b""))


def test_read_response_bad_crc_raises():
    frame = bytearray(encode_response(SensorReading(1000, 0, 2048)))
    frame[-1] ^= 0xFF  # corrupt the CRC
    with pytest.raises(ProtocolError):
        read_response(BufReader(bytes(frame)))


# ---------------------------------------------------------------------------
# HardwareBackend
# ---------------------------------------------------------------------------

def test_backend_satisfies_protocol():
    backend = HardwareBackend(transport=FakeFirmware())
    assert isinstance(backend, PendulumBackend)


def test_reset_flushes_and_zeroes_clock():
    fw = FakeFirmware()
    backend = HardwareBackend(transport=fw)
    reading = backend.reset()
    assert fw.flushed == 1
    assert fw.requests[-1] == (CMD_RESET, 0)
    assert reading.t_us == 0


def test_reset_ignores_sim_only_kwargs():
    backend = HardwareBackend(transport=FakeFirmware())
    reading = backend.reset(pendulum_down=True, initial_angle_rad=0.3)
    assert isinstance(reading, SensorReading)


def test_step_returns_reading_and_advances_clock():
    backend = HardwareBackend(transport=FakeFirmware())
    backend.reset()
    r1 = backend.step(100)
    r2 = backend.step(100)
    assert r1.t_us == 1000 and r2.t_us == 2000
    assert r2.motor_enc == 200


@pytest.mark.parametrize("pwm,expected", [(99999, PWM_MAX), (-99999, -PWM_MAX), (500, 500)])
def test_step_clamps_pwm(pwm, expected):
    fw = FakeFirmware()
    backend = HardwareBackend(transport=fw)
    backend.step(pwm)
    assert fw.requests[-1] == (CMD_STEP, expected)


def test_close_stops_motor_and_closes():
    fw = FakeFirmware()
    backend = HardwareBackend(transport=fw)
    backend.reset()
    backend.close()
    assert fw.requests[-1] == (CMD_STEP, 0)
    assert fw.closed
