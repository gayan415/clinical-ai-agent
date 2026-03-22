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
        return self.network(x)


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
