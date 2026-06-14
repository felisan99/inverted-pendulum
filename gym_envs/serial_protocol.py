"""
Binary framing for the ESP32 serial link (host <-> firmware).

This module is the single source of truth for the wire format. The ESP32 firmware
(`src/protocol.h`) mirrors these exact byte layouts. The host is the master: it
sends a fixed 5-byte request and the firmware answers with a fixed 16-byte reply
carrying a `SensorReading`. Framing is binary (not text) for latency and to keep
the 1 kHz loop cheap; every frame is guarded by a sync byte and a CRC-8.

Request  (host -> ESP32), 5 bytes:
    [0xAA][cmd:u8][pwm:i16 LE][crc8]
        cmd : CMD_STEP (0x01) or CMD_RESET (0x02)
        pwm : signed PWM in [-1023, +1023] (ignored for CMD_RESET)
        crc8: CRC-8 over (cmd, pwm)

Reply    (ESP32 -> host), 16 bytes:
    [0x55][t_us:i64 LE][motor_enc:i32 LE][pend_enc:u16 LE][crc8]
        t_us     : microseconds, monotonic, 0 right after a reset (64-bit: never
                   overflows, unlike a 32-bit micros())
        motor_enc: cumulative signed Hall counts (1716 counts/rev)
        pend_enc : AS5600 absolute position 0-4095 (0 = upright)
        crc8     : CRC-8 over (t_us, motor_enc, pend_enc)

CRC-8 is the SMBus polynomial 0x07, init 0x00, no reflection, no final xor.
"""

from __future__ import annotations

import struct

from gym_envs.backend import SensorReading

REQ_SYNC  = 0xAA
RESP_SYNC = 0x55

CMD_STEP  = 0x01
CMD_RESET = 0x02

PWM_MAX = 1023   # signed 10-bit actuator range; equals pendulum_sim.PWM_MAX

_REQ_PAYLOAD_FMT  = "<Bh"    # cmd:u8, pwm:i16
_RESP_PAYLOAD_FMT = "<qiH"   # t_us:i64, motor_enc:i32, pend_enc:u16

REQ_FRAME_LEN  = 1 + struct.calcsize(_REQ_PAYLOAD_FMT)  + 1   # 5
RESP_FRAME_LEN = 1 + struct.calcsize(_RESP_PAYLOAD_FMT) + 1   # 16

_MAX_SCAN = 256   # max bytes to discard while hunting for a valid frame


class ProtocolError(Exception):
    """Malformed frame (bad sync or CRC) or timeout while reading a reply."""


def crc8(data: bytes) -> int:
    """CRC-8/SMBus (poly 0x07, init 0x00). Matches the firmware implementation."""
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if (crc & 0x80) else (crc << 1) & 0xFF
    return crc


def encode_request(cmd: int, pwm: int) -> bytes:
    payload = struct.pack(_REQ_PAYLOAD_FMT, cmd, pwm)
    return bytes((REQ_SYNC, *payload, crc8(payload)))


def decode_request(frame: bytes) -> tuple[int, int]:
    """Inverse of `encode_request`. Used by tests and any Python-side emulator."""
    if len(frame) != REQ_FRAME_LEN or frame[0] != REQ_SYNC:
        raise ProtocolError("bad request framing")
    payload = frame[1:-1]
    if crc8(payload) != frame[-1]:
        raise ProtocolError("request CRC mismatch")
    return struct.unpack(_REQ_PAYLOAD_FMT, payload)


def encode_response(reading: SensorReading) -> bytes:
    """Build a reply frame. The firmware does this in C; Python uses it for tests."""
    payload = struct.pack(_RESP_PAYLOAD_FMT, reading.t_us, reading.motor_enc, reading.pend_enc)
    return bytes((RESP_SYNC, *payload, crc8(payload)))


def read_response(io) -> SensorReading:
    """Read and validate one reply frame from a serial-like object.

    `io` must expose `read(n) -> bytes` (pyserial Serial qualifies). Recovers from
    leading garbage (e.g. ESP32 boot-log bytes) and false sync bytes by aligning to
    the sync byte and re-scanning whenever the CRC fails. Raises `ProtocolError` on
    timeout (short read) or if no valid frame appears within `_MAX_SCAN` bytes.
    """
    buf = bytearray()
    scanned = 0
    while scanned < _MAX_SCAN:
        while len(buf) < RESP_FRAME_LEN:
            chunk = io.read(RESP_FRAME_LEN - len(buf))
            if not chunk:
                raise ProtocolError("timeout waiting for reply")
            buf += chunk

        if buf[0] != RESP_SYNC:
            idx = buf.find(RESP_SYNC)
            if idx == -1:
                scanned += len(buf)
                buf.clear()
            else:
                scanned += idx
                del buf[:idx]
            continue

        payload = bytes(buf[1:RESP_FRAME_LEN - 1])
        if crc8(payload) == buf[RESP_FRAME_LEN - 1]:
            t_us, motor_enc, pend_enc = struct.unpack(_RESP_PAYLOAD_FMT, payload)
            return SensorReading(t_us, motor_enc, pend_enc)

        scanned += 1          # drop this false sync byte and keep hunting
        del buf[0]

    raise ProtocolError("no valid frame within scan window")
