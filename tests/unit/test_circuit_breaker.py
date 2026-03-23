"""Tests for circuit breaker — protects against cascading failures.

Same pattern you'd use for any microservice dependency.
Three states: CLOSED (normal), OPEN (stop calling), HALF_OPEN (test one request).
"""

import pytest

from sre.circuit_breaker import CircuitBreaker, CircuitState


@pytest.mark.unit
class TestCircuitBreaker:
    def test_starts_closed(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
        assert cb.state == CircuitState.CLOSED

    def test_stays_closed_below_threshold(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

    def test_opens_at_threshold(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_open_circuit_blocks_calls(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=60)
        cb.record_failure()
        assert cb.allow_request() is False

    def test_success_resets_failure_count(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb.failure_count == 0
        assert cb.state == CircuitState.CLOSED

    def test_half_open_after_timeout(self):
        """After recovery timeout, circuit moves to HALF_OPEN to test one request."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0)
        cb.record_failure()
        # With 0s timeout, immediately transitions to HALF_OPEN
        assert cb.state == CircuitState.HALF_OPEN
        assert cb.allow_request() is True  # allows one test request

    def test_half_open_success_closes(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0)
        cb.record_failure()  # → OPEN → immediately HALF_OPEN (0s timeout)
        cb.record_success()  # test request succeeded
        assert cb.state == CircuitState.CLOSED

    def test_half_open_failure_reopens(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0)
        cb.record_failure()  # → OPEN → HALF_OPEN (0s timeout)
        cb.record_failure()  # test request failed → back to OPEN
        # With 0s timeout, OPEN immediately transitions to HALF_OPEN again
        # so we check it's not CLOSED (the failure didn't reset)
        assert cb.state != CircuitState.CLOSED
        assert cb.failure_count == 2
