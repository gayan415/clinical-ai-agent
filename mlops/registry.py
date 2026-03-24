"""File-based model registry — version tracking, promotion, rollback.

Tracks trained models with their metrics, manages champion/challenger
pattern, and persists to disk (JSON file). In production, this would
be MLflow or SageMaker Model Registry. We built our own to understand
the fundamentals.

Key concepts:
- active: the champion model currently serving predictions
- candidate: the challenger model ready for rollback
- promote: swap candidate to active
- rollback: restore previous active model
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class ModelRegistry:
    """File-based model registry with champion/challenger support."""

    def __init__(self, registry_dir: str) -> None:
        self._registry_path = Path(registry_dir) / "registry.json"
        self._data = self._load()

    def _load(self) -> dict[str, Any]:
        """Load registry from disk, or create empty."""
        if self._registry_path.exists():
            return json.loads(self._registry_path.read_text())  # type: ignore[no-any-return]
        return {"models": [], "active_key": None, "previous_active_key": None}

    def _save(self) -> None:
        """Persist registry to disk."""
        self._registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._registry_path.write_text(json.dumps(self._data, indent=2))

    def _model_key(self, name: str, version: str) -> str:
        return f"{name}:{version}"

    def register(
        self,
        name: str,
        version: str,
        path: str,
        metrics: dict[str, float],
    ) -> None:
        """Register a new model version.

        First model registered automatically becomes active (champion).
        Subsequent models become candidates.
        """
        entry = {
            "name": name,
            "version": version,
            "path": path,
            "metrics": metrics,
            "registered_at": datetime.now(tz=UTC).isoformat(),
            "status": "candidate",
        }

        # First model becomes active
        if not self._data["models"]:
            entry["status"] = "active"
            self._data["active_key"] = self._model_key(name, version)

        self._data["models"].append(entry)
        self._save()

    def list_models(self) -> list[dict[str, Any]]:
        """List all registered models."""
        return list(self._data["models"])

    def get_active_model(self) -> dict[str, Any] | None:
        """Get the currently active (champion) model."""
        for model in self._data["models"]:
            if model["status"] == "active":
                return dict(model)
        return None

    def promote(self, name: str, version: str) -> None:
        """Promote a model to active (champion). Previous active becomes candidate."""
        new_key = self._model_key(name, version)

        # Demote current active
        for model in self._data["models"]:
            if model["status"] == "active":
                model["status"] = "candidate"
                self._data["previous_active_key"] = self._model_key(model["name"], model["version"])

        # Promote new model
        for model in self._data["models"]:
            if self._model_key(model["name"], model["version"]) == new_key:
                model["status"] = "active"
                self._data["active_key"] = new_key

        self._save()

    def rollback(self) -> None:
        """Rollback to previous active model."""
        previous_key = self._data.get("previous_active_key")
        if not previous_key:
            return

        # Demote current active
        for model in self._data["models"]:
            if model["status"] == "active":
                model["status"] = "candidate"

        # Restore previous
        for model in self._data["models"]:
            if self._model_key(model["name"], model["version"]) == previous_key:
                model["status"] = "active"
                self._data["active_key"] = previous_key

        self._data["previous_active_key"] = None
        self._save()
