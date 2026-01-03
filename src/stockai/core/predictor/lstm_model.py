"""LSTM Sequence Model for StockAI Direction Prediction.

Captures sequential dependencies in price patterns using PyTorch.
"""

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from stockai.core.predictor.features import FeatureEngineer, create_target, ALL_FEATURES

logger = logging.getLogger(__name__)


class LSTMNetwork(nn.Module):
    """LSTM neural network for sequence classification."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Use last time step
        last_output = lstm_out[:, -1, :]
        return self.fc(last_output).squeeze(-1)

    def set_inference_mode(self):
        """Set model to inference/evaluation mode."""
        self.train(False)


class LSTMPredictor:
    """LSTM-based direction classifier.

    Features:
    - Sequence input: 20 trading days
    - 2-layer LSTM architecture
    - Early stopping with patience
    - GPU acceleration if available
    - Model saved in PyTorch format
    """

    def __init__(
        self,
        sequence_length: int = 20,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        model_path: Path | None = None,
    ):
        """Initialize LSTM predictor.

        Args:
            sequence_length: Number of time steps for input
            hidden_size: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            learning_rate: Initial learning rate
            model_path: Path to save/load model
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required. Install with: pip install torch")

        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.model_path = model_path or Path("data/models/lstm_v1.pt")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: LSTMNetwork | None = None
        self.feature_engineer = FeatureEngineer(normalize=True)
        self.feature_names: list[str] = []
        self.training_metrics: dict[str, Any] = {}

        logger.info(f"LSTM using device: {self.device}")

    def _create_sequences(
        self,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create sequence data for LSTM.

        Args:
            features: Feature DataFrame
            target: Target Series

        Returns:
            Tuple of (X sequences, y targets)
        """
        X_list = []
        y_list = []

        feature_values = features.values
        target_values = target.values

        for i in range(len(features) - self.sequence_length):
            X_list.append(feature_values[i:i + self.sequence_length])
            # Target is for the day after the sequence
            if i + self.sequence_length < len(target_values):
                y_list.append(target_values[i + self.sequence_length])

        return np.array(X_list), np.array(y_list)

    def train(
        self,
        train_df: pd.DataFrame,
        horizon: int = 3,
        epochs: int = 100,
        batch_size: int = 32,
        patience: int = 10,
        validation_split: float = 0.2,
    ) -> dict[str, float]:
        """Train the LSTM model.

        Args:
            train_df: OHLCV DataFrame
            horizon: Days ahead for prediction
            epochs: Maximum training epochs
            batch_size: Training batch size
            patience: Early stopping patience
            validation_split: Validation data fraction

        Returns:
            Training metrics
        """
        logger.info(f"Training LSTM model with {len(train_df)} samples")

        # Generate features
        features = self.feature_engineer.generate_features(train_df)
        self.feature_names = list(features.columns)
        n_features = len(self.feature_names)

        # Create target
        target = create_target(train_df.loc[features.index], horizon=horizon)

        # Align and create sequences
        valid_idx = target.dropna().index.intersection(features.index)
        features = features.loc[valid_idx]
        target = target.loc[valid_idx]

        X, y = self._create_sequences(features, target)

        if len(X) < 100:
            raise ValueError(f"Insufficient training data: {len(X)} sequences")

        # Train/val split
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).to(self.device)

        # Create data loaders
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Initialize model
        self.model = LSTMNetwork(
            input_size=n_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5
        )

        # Training loop with early stopping
        best_val_loss = float("inf")
        patience_counter = 0
        history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

        for epoch in range(epochs):
            # Training mode
            self.model.train(True)
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Inference mode for validation
            self.model.set_inference_mode()
            with torch.no_grad():
                val_outputs = self.model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t).item()
                val_preds = (val_outputs > 0.5).float()
                val_accuracy = (val_preds == y_val_t).float().mean().item()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_accuracy)

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model state
                best_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, "
                    f"val_loss={val_loss:.4f}, val_acc={val_accuracy:.4f}"
                )

        # Restore best model
        self.model.load_state_dict(best_state)

        # Final metrics
        self.model.set_inference_mode()
        with torch.no_grad():
            train_outputs = self.model(X_train_t)
            train_preds = (train_outputs > 0.5).float()
            train_accuracy = (train_preds == y_train_t).float().mean().item()

            val_outputs = self.model(X_val_t)
            val_preds = (val_outputs > 0.5).float()
            val_accuracy = (val_preds == y_val_t).float().mean().item()

        self.training_metrics = {
            "train_accuracy": float(train_accuracy),
            "val_accuracy": float(val_accuracy),
            "best_val_loss": float(best_val_loss),
            "epochs_trained": len(history["train_loss"]),
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_features": n_features,
            "sequence_length": self.sequence_length,
        }

        logger.info(f"Training complete. Val accuracy: {val_accuracy:.4f}")
        return self.training_metrics

    def predict(
        self,
        df: pd.DataFrame,
    ) -> dict[str, Any]:
        """Predict direction for given data.

        Args:
            df: OHLCV DataFrame (needs at least sequence_length rows)

        Returns:
            Prediction with probability and confidence
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() or load() first.")

        if len(df) < self.sequence_length:
            raise ValueError(
                f"Need at least {self.sequence_length} rows for prediction"
            )

        start_time = time.time()

        # Generate features
        features = self.feature_engineer.generate_features(df)

        if len(features) < self.sequence_length:
            raise ValueError("Insufficient data after feature generation")

        # Use last sequence
        sequence = features.iloc[-self.sequence_length:].values
        X = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)

        # Predict in inference mode
        self.model.set_inference_mode()
        with torch.no_grad():
            prob = float(self.model(X).item())

        direction = "UP" if prob > 0.5 else "DOWN"
        confidence = abs(prob - 0.5) * 2

        inference_time = (time.time() - start_time) * 1000

        return {
            "direction": direction,
            "probability": prob,
            "confidence": confidence,
            "model": "lstm",
            "inference_time_ms": inference_time,
        }

    def predict_proba(self, df: pd.DataFrame) -> float:
        """Get prediction probability (for ensemble).

        Args:
            df: OHLCV DataFrame

        Returns:
            Probability of UP direction (0-1)
        """
        result = self.predict(df)
        return result["probability"]

    def save(self, path: Path | None = None) -> bool:
        """Save model to file.

        Args:
            path: Save path (defaults to model_path)

        Returns:
            True if successful
        """
        if self.model is None:
            raise ValueError("No model to save")

        save_path = path or self.model_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            checkpoint = {
                "model_state_dict": self.model.state_dict(),
                "feature_names": self.feature_names,
                "training_metrics": self.training_metrics,
                "config": {
                    "sequence_length": self.sequence_length,
                    "hidden_size": self.hidden_size,
                    "num_layers": self.num_layers,
                    "dropout": self.dropout,
                    "input_size": len(self.feature_names),
                },
            }
            torch.save(checkpoint, save_path)
            logger.info(f"Model saved to {save_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False

    def load(self, path: Path | None = None) -> bool:
        """Load model from file.

        Args:
            path: Load path (defaults to model_path)

        Returns:
            True if successful
        """
        load_path = path or self.model_path

        if not load_path.exists():
            logger.warning(f"Model file not found: {load_path}")
            return False

        try:
            checkpoint = torch.load(load_path, map_location=self.device)

            config = checkpoint["config"]
            self.sequence_length = config["sequence_length"]
            self.hidden_size = config["hidden_size"]
            self.num_layers = config["num_layers"]
            self.dropout = config["dropout"]

            self.model = LSTMNetwork(
                input_size=config["input_size"],
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
            ).to(self.device)

            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.feature_names = checkpoint["feature_names"]
            self.training_metrics = checkpoint["training_metrics"]

            self.model.set_inference_mode()
            logger.info(f"Model loaded from {load_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
