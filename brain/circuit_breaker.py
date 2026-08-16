"""
brain/circuit_breaker.py — Circuit breaker for external service calls.
 
[FIX M7] Added half-open state tracking. If a failure occurs during
half-open, immediately re-open the breaker instead of requiring
another threshold failures.
 
Thread-safe: all state transitions are guarded by a lock.
"""
 
import threading
import time
 
from brain.exceptions import CircuitOpenError
 
__all__ = ["CircuitBreaker", "CircuitOpenError"]
 
 
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
        self._half_open = False  # [FIX M7] Track half-open state
 
    @property
    def is_open(self) -> bool:
        """True when calls should fast-fail.
 
        False once recovery_timeout has elapsed (half-open state —
        the caller may send one probe request).
        """
        if not self.enabled or self.recovery_timeout <= 0:
            return False
        with self._lock:
            if self._open_until == 0.0:
                return False
            now = time.monotonic()
            if now >= self._open_until:
                # [FIX M7] Entering half-open state
                self._half_open = True
                return False
            return True
 
    def record_success(self) -> None:
        """A call succeeded — reset the failure streak and close."""
        with self._lock:
            self._failures = 0
            self._open_until = 0.0
            self._half_open = False  # [FIX M7]
 
    def record_failure(self) -> None:
        """A call failed — open the breaker at the threshold.
        
        [FIX M7] If in half-open state, immediately re-open without
        waiting for threshold failures.
        """
        with self._lock:
            # [FIX M7] Immediate re-open during half-open probe failure
            if self._half_open:
                self._open_until = time.monotonic() + self.recovery_timeout
                self._half_open = False
                self._failures = 0
                return
            
            self._failures += 1
            if self._failures >= self.failure_threshold:
                self._open_until = time.monotonic() + self.recovery_timeout
                self._failures = 0  # Reset for next cycle
 
    def reset(self) -> None:
        """Force the breaker closed (used by tests / explicit recovery)."""
        with self._lock:
            self._failures = 0
            self._open_until = 0.0
            self._half_open = False  # [FIX M7]
 
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
 