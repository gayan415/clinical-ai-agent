"""Tests for inference monitoring — structured logging for every prediction."""

import json

import pytest

from mlops.monitor import InferenceMonitor


@pytest.mark.unit
class TestInferenceMonitor:
    def test_log_inference(self, tmp_path):
        monitor = InferenceMonitor(log_dir=str(tmp_path))
        monitor.log_inference(
            input_features={"age": 65, "ejection_fraction": 30},
            prediction=0.82,
            confidence=0.89,
            model_version="xgboost_v1",
            latency_ms=3.2,
        )
        logs = monitor.get_logs()
        assert len(logs) == 1
        assert logs[0]["prediction"] == 0.82

    def test_logs_are_append_only(self, tmp_path):
        """New logs append, never overwrite previous ones."""
        monitor = InferenceMonitor(log_dir=str(tmp_path))
        monitor.log_inference(
            input_features={},
            prediction=0.5,
            confidence=0.5,
            model_version="v1",
            latency_ms=1.0,
        )
        monitor.log_inference(
            input_features={},
            prediction=0.9,
            confidence=0.9,
            model_version="v1",
            latency_ms=2.0,
        )
        logs = monitor.get_logs()
        assert len(logs) == 2

    def test_log_has_timestamp(self, tmp_path):
        monitor = InferenceMonitor(log_dir=str(tmp_path))
        monitor.log_inference(
            input_features={},
            prediction=0.5,
            confidence=0.5,
            model_version="v1",
            latency_ms=1.0,
        )
        assert "timestamp" in monitor.get_logs()[0]

    def test_logs_are_valid_json(self, tmp_path):
        """Each line in the log file must be valid JSON (JSONL format)."""
        monitor = InferenceMonitor(log_dir=str(tmp_path))
        monitor.log_inference(
            input_features={"age": 65},
            prediction=0.82,
            confidence=0.89,
            model_version="v1",
            latency_ms=3.0,
        )
        log_file = tmp_path / "inference.jsonl"
        for line in log_file.read_text().strip().split("\n"):
            parsed = json.loads(line)
            assert "prediction" in parsed

    def test_metrics_summary(self, tmp_path):
        """Should compute p50/p95/p99 latency from logged inferences."""
        monitor = InferenceMonitor(log_dir=str(tmp_path))
        for latency in [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]:
            monitor.log_inference(
                input_features={},
                prediction=0.5,
                confidence=0.5,
                model_version="v1",
                latency_ms=latency,
            )
        summary = monitor.get_metrics_summary()
        assert "p50_latency_ms" in summary
        assert "p95_latency_ms" in summary
        assert "p99_latency_ms" in summary
        assert "total_predictions" in summary
        assert summary["total_predictions"] == 10

    def test_empty_metrics(self, tmp_path):
        monitor = InferenceMonitor(log_dir=str(tmp_path))
        summary = monitor.get_metrics_summary()
        assert summary["total_predictions"] == 0
