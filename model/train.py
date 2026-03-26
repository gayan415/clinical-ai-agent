"""Model training — XGBoost and PyTorch for heart failure risk prediction.

XGBoost: Gradient-boosted decision trees. Best algorithm for tabular data.
  Builds 100 trees, each correcting the previous tree's mistakes.

PyTorch: Simple feedforward neural network (12 → 64 → 32 → 1).
  Demonstrates deep learning knowledge. Overkill for 299 rows,
  but checks the JD requirement.

Both models take 12 patient features and output a death probability.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int = 42,
) -> XGBClassifier:
    """Train an XGBoost classifier for heart failure risk prediction.

    XGBoost = eXtreme Gradient Boosting. It builds decision trees sequentially,
    where each new tree focuses on the cases the previous trees got wrong.
    After 100 trees, the ensemble votes on each prediction.

    Args:
        X_train: Feature matrix (n_patients, 12)
        y_train: Target vector (n_patients,) — 0=survived, 1=died
        seed: Random seed for reproducibility

    Returns:
        Trained XGBClassifier
    """
    # Calculate class weight to handle imbalance.
    # ~32% of patients died — without this, the model leans toward "survived"
    # and misses sick patients. scale_pos_weight tells XGBoost to penalize
    # missing a death case more than missing a survival case.
    n_negative = np.sum(y_train == 0)
    n_positive = np.sum(y_train == 1)
    scale_pos_weight = n_negative / n_positive if n_positive > 0 else 1.0

    model = XGBClassifier(
        n_estimators=100,  # Build 100 trees
        max_depth=4,  # Each tree can be 4 levels deep
        learning_rate=0.1,  # How much each tree contributes (smaller = more conservative)
        scale_pos_weight=scale_pos_weight,  # Handle class imbalance
        random_state=seed,
        eval_metric="logloss",  # Optimize for log loss (good for binary classification)
    )
    model.fit(X_train, y_train)
    return model


class HeartFailureNet(nn.Module):
    """PyTorch neural network for heart failure risk prediction.

    Architecture: Input(12) → Linear(64) → ReLU → Dropout →
    Linear(32) → ReLU → Dropout → Linear(1) → Sigmoid

    - Linear layers: multiply inputs by learned weights + bias
    - ReLU: max(0, x) — introduces nonlinearity (without it, stacking layers does nothing)
    - Dropout: randomly zeros 30% of neurons during training — prevents memorization (overfitting)
    - Sigmoid: squashes output to 0-1 range (probability of death)
    """

    def __init__(self, input_dim: int = 12) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = self.network(x)
        return result


def train_pytorch(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int = 42,
    epochs: int = 100,
    learning_rate: float = 0.001,
) -> dict:
    """Train a PyTorch neural network for heart failure risk prediction.

    Training loop:
    1. Forward pass: feed data through network, get predictions
    2. Compute loss: how wrong were the predictions? (binary cross-entropy)
    3. Backward pass: compute gradients (which weights caused the error)
    4. Update weights: adjust weights in the direction that reduces error
    5. Repeat for 100 epochs (passes through the full dataset)

    Returns a dict with the model, scaler, and metadata — not just the model,
    because PyTorch needs the scaler to transform new inputs at inference time.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Standardize features: subtract mean, divide by std deviation.
    # Neural networks train better when all features are on the same scale.
    # XGBoost doesn't need this (trees don't care about scale), but NNs do.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # Convert numpy arrays to PyTorch tensors
    X_tensor = torch.FloatTensor(X_scaled)
    y_tensor = torch.FloatTensor(y_train).unsqueeze(1)  # Shape: (n, 1)

    # Initialize model, loss function, optimizer
    model = HeartFailureNet(input_dim=X_train.shape[1])
    criterion = nn.BCELoss()  # Binary Cross-Entropy — standard for binary classification
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()
    for _epoch in range(epochs):
        # Forward pass
        predictions = model(X_tensor)
        loss = criterion(predictions, y_tensor)

        # Backward pass + weight update
        optimizer.zero_grad()  # Clear previous gradients
        loss.backward()  # Compute gradients
        optimizer.step()  # Update weights

    # Switch to evaluation mode (disables dropout)
    # This is PyTorch's model.eval(), not Python's eval() — it's safe
    model.eval()  # noqa: S307

    return {
        "model": model,
        "scaler": scaler,
        "input_dim": X_train.shape[1],
    }


def train_and_save(
    data_path: str = "model/data/heart_failure.csv",
    output_dir: str = "models",
    registry_dir: str = "models",
    seed: int = 42,
) -> None:
    """Full training pipeline: load → split → train both → evaluate → register → save.

    This is what `make train` calls. It:
    1. Loads the UCI dataset
    2. Splits 80/20 with a fixed seed (reproducible)
    3. Trains both XGBoost and PyTorch
    4. Evaluates both on the test set
    5. Registers both in the model registry with metrics
    6. Saves the champion (best AUC) to disk for the FastAPI service
    """
    import hashlib
    import os

    import joblib
    import pandas as pd
    from sklearn.model_selection import train_test_split

    from mlops.registry import ModelRegistry
    from model.evaluate import evaluate_model
    from model.features import extract_features

    # Load and split data
    df = pd.read_csv(data_path)
    data_hash = hashlib.sha256(df.to_csv().encode()).hexdigest()[:12]
    print(f"Dataset: {len(df)} patients, data hash: {data_hash}")

    X, y = extract_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    print(f"Split: {len(X_train)} train, {len(X_test)} test")

    # Train XGBoost
    print("\nTraining XGBoost...")
    xgb_model = train_xgboost(X_train, y_train, seed=seed)
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test, model_type="xgboost")
    print(f"  AUC: {xgb_metrics['auc']:.3f} | Accuracy: {xgb_metrics['accuracy']:.3f}")

    # Train PyTorch
    print("\nTraining PyTorch...")
    pt_model = train_pytorch(X_train, y_train, seed=seed)
    pt_metrics = evaluate_model(pt_model, X_test, y_test, model_type="pytorch")
    print(f"  AUC: {pt_metrics['auc']:.3f} | Accuracy: {pt_metrics['accuracy']:.3f}")

    # Register both in model registry
    os.makedirs(output_dir, exist_ok=True)
    registry = ModelRegistry(registry_dir=registry_dir)

    registry.register(
        name="xgboost_hf_risk",
        version=f"v1_{data_hash}",
        path=f"{output_dir}/xgboost_hf_risk.pkl",
        metrics={k: round(v, 4) for k, v in xgb_metrics.items()},
    )
    registry.register(
        name="pytorch_hf_risk",
        version=f"v1_{data_hash}",
        path=f"{output_dir}/pytorch_hf_risk.pt",
        metrics={k: round(v, 4) for k, v in pt_metrics.items()},
    )

    # Promote champion (best AUC)
    if xgb_metrics["auc"] >= pt_metrics["auc"]:
        champion, challenger = "XGBoost", "PyTorch"
    else:
        registry.promote("pytorch_hf_risk", f"v1_{data_hash}")
        champion, challenger = "PyTorch", "XGBoost"

    print(f"\nChampion: {champion} (AUC {max(xgb_metrics['auc'], pt_metrics['auc']):.3f})")
    print(f"Challenger: {challenger}")

    # Save champion model to disk for FastAPI service
    joblib.dump(xgb_model, f"{output_dir}/xgboost_hf_risk.pkl")
    torch.save(pt_model, f"{output_dir}/pytorch_hf_risk.pt")
    print(f"\nModels saved to {output_dir}/")
    print(f"Registry saved to {registry_dir}/registry.json")


if __name__ == "__main__":
    train_and_save()
