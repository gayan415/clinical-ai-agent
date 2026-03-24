"""Inference monitoring — structured logging for every prediction.

Every prediction gets logged as a JSON line (JSONL format):
- Input features, prediction, confidence, model version, latency
- Append-only (immutable audit trail)
- Queryable for metrics: p50/p95/p99 latency, prediction distribution

In production, these logs pipe into CloudWatch/ELK for dashboards and alerts.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


class InferenceMonitor:
    """Logs every prediction for observability and audit."""

    def __init__(self, log_dir: str) -> None:
        self._log_path = Path(log_dir) / "inference.jsonl"
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_inference(
        self,
        input_features: dict[str, Any],
        prediction: float,
        confidence: float,
        model_version: str,
        latency_ms: float,
    ) -> None:
        """Append a prediction log entry (JSONL — one JSON object per line)."""
        entry = {
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "input_features": input_features,
            "prediction": prediction,
            "confidence": confidence,
            "model_version": model_version,
            "latency_ms": latency_ms,
        }
        with open(self._log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_logs(self) -> list[dict[str, Any]]:
        """Read all logged inferences."""
        if not self._log_path.exists():
            return []
        entries = []
        for line in self._log_path.read_text().strip().split("\n"):
            if line:
                entries.append(json.loads(line))
        return entries

    def get_metrics_summary(self) -> dict[str, Any]:
        """Compute summary metrics from logged inferences.

        Returns p50/p95/p99 latency, total predictions, and
        prediction distribution stats.
        """
        logs = self.get_logs()
        if not logs:
            return {"total_predictions": 0}

        latencies = [log["latency_ms"] for log in logs]
        predictions = [log["prediction"] for log in logs]

        return {
            "total_predictions": len(logs),
            "p50_latency_ms": float(np.percentile(latencies, 50)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
            "p99_latency_ms": float(np.percentile(latencies, 99)),
            "mean_prediction": float(np.mean(predictions)),
            "prediction_std": float(np.std(predictions)),
        }
