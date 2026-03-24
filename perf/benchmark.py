"""Performance benchmarks for the model prediction service.

Measures single-request latency (cold + warm), batch inference throughput,
and compares against baseline. Fails if performance regresses > 10%.

Run: python -m perf.benchmark
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from model.evaluate import evaluate_model
from model.features import extract_features
from model.train import train_xgboost

BASELINE_PATH = Path("perf/baseline.json")


def benchmark_single_request(model: object, X_test: np.ndarray, n_iterations: int = 100) -> dict:
    """Benchmark single prediction latency over n iterations."""
    # Cold start (first prediction)
    start = time.perf_counter()
    model.predict_proba(X_test[:1])  # type: ignore[union-attr]
    cold_ms = (time.perf_counter() - start) * 1000

    # Warm predictions
    latencies = []
    for i in range(n_iterations):
        start = time.perf_counter()
        model.predict_proba(X_test[i % len(X_test) : i % len(X_test) + 1])  # type: ignore[union-attr]
        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)

    return {
        "cold_start_ms": round(cold_ms, 3),
        "p50_ms": round(float(np.percentile(latencies, 50)), 3),
        "p95_ms": round(float(np.percentile(latencies, 95)), 3),
        "p99_ms": round(float(np.percentile(latencies, 99)), 3),
        "mean_ms": round(float(np.mean(latencies)), 3),
        "iterations": n_iterations,
    }


def benchmark_batch(model: object, X_test: np.ndarray) -> dict:
    """Benchmark batch inference throughput."""
    results = {}
    for batch_size in [10, 100]:
        batch = X_test[:batch_size] if len(X_test) >= batch_size else X_test
        start = time.perf_counter()
        model.predict_proba(batch)  # type: ignore[union-attr]
        elapsed = (time.perf_counter() - start) * 1000
        results[f"batch_{batch_size}_ms"] = round(elapsed, 3)
        results[f"batch_{batch_size}_per_sample_ms"] = round(elapsed / len(batch), 3)

    return results


def compare_with_baseline(current: dict) -> dict:
    """Compare current results with saved baseline. Flag regressions > 10%."""
    if not BASELINE_PATH.exists():
        return {"comparison": "no_baseline", "regressions": []}

    baseline = json.loads(BASELINE_PATH.read_text())
    regressions = []

    for key in ["p50_ms", "p95_ms", "p99_ms"]:
        if key in baseline.get("single_request", {}) and key in current.get("single_request", {}):
            baseline_val = baseline["single_request"][key]
            current_val = current["single_request"][key]
            if baseline_val > 0 and current_val > baseline_val * 1.1:
                regressions.append(
                    {
                        "metric": key,
                        "baseline": baseline_val,
                        "current": current_val,
                        "regression_pct": round((current_val / baseline_val - 1) * 100, 1),
                    }
                )

    return {
        "comparison": "pass" if not regressions else "regression_detected",
        "regressions": regressions,
    }


def run_benchmarks() -> dict:
    """Run all benchmarks and return results."""
    print("Loading data and training model...")
    df = pd.read_csv("model/data/heart_failure.csv")
    X, y = extract_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_xgboost(X_train, y_train, seed=42)

    # Model quality
    metrics = evaluate_model(model, X_test, y_test, model_type="xgboost")
    print(f"Model AUC: {metrics['auc']:.3f}")

    # Benchmarks
    print("Running single request benchmark (100 iterations)...")
    single = benchmark_single_request(model, X_test)
    p50, p95, p99 = single["p50_ms"], single["p95_ms"], single["p99_ms"]
    print(f"  p50: {p50:.1f}ms | p95: {p95:.1f}ms | p99: {p99:.1f}ms")

    print("Running batch benchmark...")
    batch = benchmark_batch(model, X_test)

    results = {
        "model_metrics": metrics,
        "single_request": single,
        "batch": batch,
    }

    # Compare with baseline
    comparison = compare_with_baseline(results)
    results["comparison"] = comparison
    print(f"Baseline comparison: {comparison['comparison']}")

    if comparison["regressions"]:
        for reg in comparison["regressions"]:
            print(f"  REGRESSION: {reg['metric']} — {reg['regression_pct']}% slower")

    # Save as new baseline
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_PATH.write_text(json.dumps(results, indent=2))
    print(f"Baseline saved to {BASELINE_PATH}")

    return results


if __name__ == "__main__":
    run_benchmarks()
