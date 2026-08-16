"""
brain/circuit_breaker.py — Circuit breaker for external service calls.

Why: if Ollama becomes unresponsive mid-session (timeout, OOM kill,
network blip), every AI request would otherwise block for the full
OLLAMA_TIMEOUT before failing. After ``failure_threshold`` consecutive
failures the breaker *opens* and callers fast-fail immediately
("AI offline") instead of hanging; after ``recovery_timeout`` it goes
half-open and lets one probe through — success closes it, failure
re-opens it. The service auto-recovers the moment it comes back.

Thread-safe: all state transitions are guarded by a lock, so the warm-up
thread and the main loop can share one breaker safely.
"""

import threading
import time

from brain.exceptions import CircuitOpenError


class CircuitBreaker:
    """Tracks failure counts and exposes fast-fail state."""

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
        enabled: bool = True,
    ) -> None:
        self.failure_threshold = max(1, int(failure_threshold))
        self.recovery_timeout = max(0.0, float(recovery_timeout))
        self.enabled = bool(enabled)
        self._lock = threading.Lock()
        self._failures = 0
        self._open_until = 0.0

    @property
    def is_open(self) -> bool:
        """True when calls should fast-fail.

        False once ``recovery_timeout`` has elapsed (half-open state —
        the caller may send one probe request).
        """
        if not self.enabled or self.recovery_timeout <= 0:
            return False
        with self._lock:
            if self._open_until == 0.0:
                return False
            return time.monotonic() < self._open_until

    def record_success(self) -> None:
        """A call succeeded — reset the failure streak and close."""
        with self._lock:
            self._failures = 0
            self._open_until = 0.0

    def record_failure(self) -> None:
        """A call failed — open the breaker at the threshold."""
        with self._lock:
            self._failures += 1
            if self._failures >= self.failure_threshold:
                self._open_until = time.monotonic() + self.recovery_timeout
                self._failures = 0

    def reset(self) -> None:
        """Force the breaker closed (used by tests / explicit recovery)."""
        with self._lock:
            self._failures = 0
            self._open_until = 0.0

    def __call__(self, fn):
        """Decorator form: wrap a sync callable with breaker checks."""

        def wrapper(*args, **kwargs):
            if self.is_open:
                raise CircuitOpenError(
                    "Circuit breaker open — call fast-failed."
                )
            try:
                result = fn(*args, **kwargs)
            except Exception:
                self.record_failure()
                raise
            self.record_success()
            return result

        return wrapper
