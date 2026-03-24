"""Tests for model registry — version tracking, promotion, rollback.

Same concept as a container registry but for ML models.
Champion = active (serving), Challenger = candidate (backup).
"""

import pytest

from mlops.registry import ModelRegistry


@pytest.mark.unit
class TestModelRegistry:
    def test_register_model(self, tmp_path):
        registry = ModelRegistry(registry_dir=str(tmp_path))
        registry.register(
            name="xgboost_hf_risk",
            version="v1",
            path="models/v1/xgboost.pkl",
            metrics={"auc": 0.87, "accuracy": 0.82},
        )
        models = registry.list_models()
        assert len(models) == 1
        assert models[0]["name"] == "xgboost_hf_risk"

    def test_first_model_becomes_active(self, tmp_path):
        """First registered model should automatically become champion."""
        registry = ModelRegistry(registry_dir=str(tmp_path))
        registry.register(
            name="xgboost_hf_risk",
            version="v1",
            path="models/v1/xgboost.pkl",
            metrics={"auc": 0.87},
        )
        active = registry.get_active_model()
        assert active is not None
        assert active["version"] == "v1"

    def test_second_model_becomes_candidate(self, tmp_path):
        """Second model should be candidate, not override the active."""
        registry = ModelRegistry(registry_dir=str(tmp_path))
        registry.register(
            name="xgboost",
            version="v1",
            path="models/v1/xgb.pkl",
            metrics={"auc": 0.87},
        )
        registry.register(
            name="pytorch",
            version="v1",
            path="models/v1/pt.pkl",
            metrics={"auc": 0.80},
        )
        models = registry.list_models()
        assert len(models) == 2
        active = registry.get_active_model()
        assert active["name"] == "xgboost"

    def test_promote_model(self, tmp_path):
        """Promote swaps candidate to active."""
        registry = ModelRegistry(registry_dir=str(tmp_path))
        registry.register(
            name="xgboost",
            version="v1",
            path="models/v1/xgb.pkl",
            metrics={"auc": 0.85},
        )
        registry.register(
            name="pytorch",
            version="v1",
            path="models/v1/pt.pkl",
            metrics={"auc": 0.90},
        )
        registry.promote("pytorch", "v1")
        active = registry.get_active_model()
        assert active["name"] == "pytorch"

    def test_rollback(self, tmp_path):
        """Rollback restores the previous active model."""
        registry = ModelRegistry(registry_dir=str(tmp_path))
        registry.register(
            name="xgboost",
            version="v1",
            path="models/v1/xgb.pkl",
            metrics={"auc": 0.85},
        )
        registry.register(
            name="pytorch",
            version="v1",
            path="models/v1/pt.pkl",
            metrics={"auc": 0.90},
        )
        registry.promote("pytorch", "v1")
        registry.rollback()
        active = registry.get_active_model()
        assert active["name"] == "xgboost"

    def test_registry_persists_to_disk(self, tmp_path):
        """Registry data must survive process restarts."""
        registry1 = ModelRegistry(registry_dir=str(tmp_path))
        registry1.register(
            name="xgboost",
            version="v1",
            path="models/v1/xgb.pkl",
            metrics={"auc": 0.87},
        )
        # Create new instance from same directory
        registry2 = ModelRegistry(registry_dir=str(tmp_path))
        models = registry2.list_models()
        assert len(models) == 1

    def test_empty_registry(self, tmp_path):
        registry = ModelRegistry(registry_dir=str(tmp_path))
        assert registry.list_models() == []
        assert registry.get_active_model() is None
