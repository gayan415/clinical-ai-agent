"""Circuit breaker pattern — protects against cascading failures.

When the model service or LLM is down, don't keep calling it.
Three states:
  CLOSED  → normal operation, requests pass through
  OPEN    → service is down, block all requests, return fallback
  HALF_OPEN → recovery period, allow one test request

Same pattern as Netflix Hystrix or any microservice circuit breaker.
"""

import time
from enum import Enum


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker for protecting external service calls.

    Usage:
        cb = CircuitBreaker(failure_threshold=5, recovery_timeout=60)

        if cb.allow_request():
            try:
                result = call_model_service()
                cb.record_success()
            except Exception:
                cb.record_failure()
        else:
            result = fallback_response()
    """

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._failure_count = 0
        self._last_failure_time: float = 0
        self._state = CircuitState.CLOSED

    @property
    def state(self) -> CircuitState:
        """Current circuit state, with automatic OPEN → HALF_OPEN transition."""
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._last_failure_time
            if elapsed >= self._recovery_timeout:
                self._state = CircuitState.HALF_OPEN
        return self._state

    @property
    def failure_count(self) -> int:
        return self._failure_count

    def allow_request(self) -> bool:
        """Should we allow a request through?"""
        current = self.state
        if current == CircuitState.CLOSED:
            return True
        if current == CircuitState.HALF_OPEN:
            return True  # allow one test request
        return False  # OPEN — block

    def record_success(self) -> None:
        """Call succeeded — reset failures, close circuit."""
        self._failure_count = 0
        self._state = CircuitState.CLOSED

    def record_failure(self) -> None:
        """Call failed — increment counter, maybe open circuit."""
        self._failure_count += 1
        self._last_failure_time = time.monotonic()

        if self._state == CircuitState.HALF_OPEN:
            # Test request failed — back to OPEN, reset timer
            self._state = CircuitState.OPEN
        elif self._failure_count >= self._failure_threshold:
            self._state = CircuitState.OPEN
