# hive_app/buffer/buffer.py
from collections import deque
import threading


class RollingBuffer:
    """
    Thread-safe rolling buffer storing up to `capacity` items.
    Each item can be any JSON-serializable dict (e.g. a payload).
    """
    def __init__(self, capacity: int = 20):
        self.capacity = capacity
        self._dq = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def add(self, item):
        """Append new item (oldest dropped automatically when capacity exceeded)."""
        with self._lock:
            self._dq.append(item)

    def get_all(self):
        """Return a list (oldest->newest)."""
        with self._lock:
            return list(self._dq)

    def latest(self):
        """Return latest item or None."""
        with self._lock:
            return self._dq[-1] if self._dq else None

    def clear(self):
        with self._lock:
            self._dq.clear()

BUFFER = RollingBuffer(capacity=20)