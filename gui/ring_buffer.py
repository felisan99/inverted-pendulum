from __future__ import annotations

import numpy as np


class RingBuffer:
    """Pre-allocated circular buffer for three float64 time-series."""

    def __init__(self, capacity: int) -> None:
        self._cap  = capacity
        self._t    = np.empty(capacity)
        self._pend = np.empty(capacity)
        self._arm  = np.empty(capacity)
        self._pwm  = np.empty(capacity)
        self._head = 0
        self._size = 0

    def append(self, t: float, pend: float, arm: float, pwm: float) -> None:
        i = self._head % self._cap
        self._t[i]    = t
        self._pend[i] = pend
        self._arm[i]  = arm
        self._pwm[i]  = pwm
        self._head   += 1
        if self._size < self._cap:
            self._size += 1

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = self._size
        if n == 0:
            empty = np.empty(0)
            return empty, empty, empty, empty
        if n < self._cap:
            return self._t[:n], self._pend[:n], self._arm[:n], self._pwm[:n]
        start = self._head % self._cap
        idx   = np.arange(start, start + self._cap) % self._cap
        return self._t[idx], self._pend[idx], self._arm[idx], self._pwm[idx]

    def clear(self) -> None:
        self._head = 0
        self._size = 0
