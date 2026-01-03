"""E2E Tests for Prediction Models.

Tests the ML prediction pipeline including:
- Feature engineering
- XGBoost classifier
- LSTM sequence model
- Ensemble predictor
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# Skip all tests if ML dependencies not available
try:
    import xgboost
    import torch
    HAS_ML_DEPS = True
except ImportError:
    HAS_ML_DEPS = False

pytestmark = pytest.mark.skipif(
    not HAS_ML_DEPS,
    reason="ML dependencies (xgboost, torch) not installed"
)


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    n_days = 300  # Need enough data for LSTM sequences

    dates = pd.date_range(end=datetime.now(), periods=n_days, freq="B")

    # Generate realistic price movement
    base_price = 10000
    returns = np.random.normal(0.0002, 0.02, n_days)
    close_prices = base_price * np.cumprod(1 + returns)

    # Generate OHLCV
    high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.01, n_days)))
    low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.01, n_days)))
    open_prices = (high_prices + low_prices) / 2 + np.random.normal(0, 50, n_days)
    volumes = np.random.randint(1000000, 10000000, n_days)

    df = pd.DataFrame({
        "date": dates,
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "close": close_prices,
        "volume": volumes,
    })

    return df


@pytest.fixture
def small_ohlcv_data():
    """Create minimal OHLCV data for quick tests."""
    np.random.seed(42)
    n_days = 60  # Minimum for feature generation

    dates = pd.date_range(end=datetime.now(), periods=n_days, freq="B")
    base_price = 10000
    returns = np.random.normal(0.0002, 0.02, n_days)
    close_prices = base_price * np.cumprod(1 + returns)

    df = pd.DataFrame({
        "date": dates,
        "open": close_prices * 0.995,
        "high": close_prices * 1.01,
        "low": close_prices * 0.99,
        "close": close_prices,
        "volume": np.random.randint(1000000, 10000000, n_days),
    })

    return df


class TestFeatureEngineering:
    """Tests for feature engineering pipeline."""

    def test_generate_features_basic(self, small_ohlcv_data):
        """Test basic feature generation."""
        from stockai.core.predictor.features import FeatureEngineer

        engineer = FeatureEngineer(normalize=False)
        features = engineer.generate_features(small_ohlcv_data)

        # Should have features
        assert not features.empty
        assert len(features.columns) > 30

        # Should have price features
        assert "return_1d" in features.columns
        assert "volatility_20d" in features.columns

        # Should have technical features
        assert "rsi_14" in features.columns
        assert "macd" in features.columns
        assert "bb_position" in features.columns

        # Should have volume features
        assert "volume_sma5_ratio" in features.columns
        assert "obv" in features.columns

    def test_generate_features_normalized(self, small_ohlcv_data):
        """Test normalized feature generation."""
        from stockai.core.predictor.features import FeatureEngineer

        engineer = FeatureEngineer(normalize=True)
        features = engineer.generate_features(small_ohlcv_data)

        # Normalized features should have mean near 0, std near 1
        for col in ["return_1d", "rsi_14", "macd"]:
            if col in features.columns:
                mean = features[col].mean()
                std = features[col].std()
                assert abs(mean) < 0.5, f"{col} mean should be near 0"
                assert 0.5 < std < 2.0, f"{col} std should be near 1"

    def test_create_target(self, small_ohlcv_data):
        """Test target variable creation."""
        from stockai.core.predictor.features import create_target

        target = create_target(small_ohlcv_data, horizon=3)

        # Target should be binary (0 or 1)
        unique_values = target.dropna().unique()
        assert set(unique_values).issubset({0, 1})

        # Target should have same length as input
        assert len(target) == len(small_ohlcv_data)

        # With horizon=3, roughly half should be UP (1) and half DOWN (0)
        # This is a probabilistic check - not too extreme either way
        up_ratio = (target == 1).sum() / len(target)
        assert 0.2 < up_ratio < 0.8, "Target should be roughly balanced"

    def test_feature_importance(self, small_ohlcv_data):
        """Test feature importance extraction."""
        from stockai.core.predictor.features import FeatureEngineer

        engineer = FeatureEngineer()
        features = engineer.generate_features(small_ohlcv_data)

        # Create mock importance
        importance_dict = {col: 0.01 for col in features.columns}
        importance_df = engineer.get_feature_importance(importance_dict)

        # Should have feature, group, importance columns
        assert "feature" in importance_df.columns
        assert "group" in importance_df.columns
        assert "importance" in importance_df.columns

    def test_missing_columns_error(self):
        """Test error on missing required columns."""
        from stockai.core.predictor.features import FeatureEngineer

        engineer = FeatureEngineer()
        bad_df = pd.DataFrame({"price": [100, 101, 102]})

        with pytest.raises(ValueError, match="must have columns"):
            engineer.generate_features(bad_df)

    def test_empty_dataframe(self):
        """Test handling of empty dataframe."""
        from stockai.core.predictor.features import FeatureEngineer

        engineer = FeatureEngineer()
        empty_df = pd.DataFrame()

        features = engineer.generate_features(empty_df)
        assert features.empty


class TestXGBoostPredictor:
    """Tests for XGBoost prediction model."""

    def test_xgboost_train(self, sample_ohlcv_data):
        """Test XGBoost model training."""
        from stockai.core.predictor.xgboost_model import XGBoostPredictor

        predictor = XGBoostPredictor()
        metrics = predictor.train(sample_ohlcv_data, horizon=3)

        # Should have training metrics
        assert "train_accuracy" in metrics
        assert "val_accuracy" in metrics
        assert 0 <= metrics["train_accuracy"] <= 1
        assert 0 <= metrics["val_accuracy"] <= 1

    def test_xgboost_predict(self, sample_ohlcv_data):
        """Test XGBoost prediction."""
        from stockai.core.predictor.xgboost_model import XGBoostPredictor

        predictor = XGBoostPredictor()
        predictor.train(sample_ohlcv_data, horizon=3)

        # Predict on same data (for testing)
        result = predictor.predict(sample_ohlcv_data)

        # Should have required fields
        assert "direction" in result
        assert result["direction"] in ["UP", "DOWN"]
        assert "probability" in result
        assert 0 <= result["probability"] <= 1
        assert "confidence" in result
        assert result["model"] == "xgboost"

    def test_xgboost_predict_proba(self, sample_ohlcv_data):
        """Test probability prediction interface."""
        from stockai.core.predictor.xgboost_model import XGBoostPredictor

        predictor = XGBoostPredictor()
        predictor.train(sample_ohlcv_data, horizon=3)

        prob = predictor.predict_proba(sample_ohlcv_data)
        assert 0 <= prob <= 1

    def test_xgboost_feature_importance(self, sample_ohlcv_data):
        """Test feature importance extraction."""
        from stockai.core.predictor.xgboost_model import XGBoostPredictor

        predictor = XGBoostPredictor()
        predictor.train(sample_ohlcv_data, horizon=3)

        importance = predictor.get_feature_importance()

        # Should be a DataFrame with features
        assert isinstance(importance, pd.DataFrame)
        assert "feature" in importance.columns
        assert "importance" in importance.columns

    def test_xgboost_save_load(self, sample_ohlcv_data, tmp_path):
        """Test model serialization."""
        from stockai.core.predictor.xgboost_model import XGBoostPredictor

        model_path = tmp_path / "xgboost_test.json"

        # Train and save
        predictor = XGBoostPredictor(model_path=model_path)
        predictor.train(sample_ohlcv_data, horizon=3)
        original_result = predictor.predict(sample_ohlcv_data)
        assert predictor.save()

        # Load in new predictor
        new_predictor = XGBoostPredictor(model_path=model_path)
        assert new_predictor.load()

        # Should produce same results
        loaded_result = new_predictor.predict(sample_ohlcv_data)
        assert loaded_result["direction"] == original_result["direction"]
        assert abs(loaded_result["probability"] - original_result["probability"]) < 0.01

    def test_xgboost_walkforward(self, sample_ohlcv_data):
        """Test walk-forward cross-validation."""
        from stockai.core.predictor.xgboost_model import XGBoostPredictor

        predictor = XGBoostPredictor()
        cv_metrics = predictor.train_walkforward(
            sample_ohlcv_data,
            horizon=3,
            n_splits=3,
            test_size_months=1,
        )

        # Should have CV metrics
        assert "mean_accuracy" in cv_metrics
        assert "std_accuracy" in cv_metrics
        assert "folds" in cv_metrics
        assert len(cv_metrics["folds"]) == 3

    def test_xgboost_insufficient_data(self):
        """Test error on insufficient training data."""
        from stockai.core.predictor.xgboost_model import XGBoostPredictor

        predictor = XGBoostPredictor()

        # Create minimal data
        small_df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=30, freq="B"),
            "open": [100] * 30,
            "high": [101] * 30,
            "low": [99] * 30,
            "close": [100] * 30,
            "volume": [1000000] * 30,
        })

        with pytest.raises(ValueError, match="Insufficient"):
            predictor.train(small_df)


@pytest.mark.skip(reason="LSTM tests cause segfaults on some PyTorch installations")
class TestLSTMPredictor:
    """Tests for LSTM sequence model.

    Note: These tests are skipped by default due to PyTorch/MPS compatibility
    issues that can cause segmentation faults on macOS with M-series chips.
    Run with: pytest -v --run-lstm to execute these tests.
    """

    def test_lstm_train(self, sample_ohlcv_data):
        """Test LSTM model training."""
        from stockai.core.predictor.lstm_model import LSTMPredictor

        predictor = LSTMPredictor(
            sequence_length=10,  # Shorter for faster testing
            hidden_size=32,
            num_layers=1,
        )
        metrics = predictor.train(
            sample_ohlcv_data,
            horizon=3,
            epochs=5,  # Minimal epochs for testing
            patience=3,
        )

        # Should have training metrics
        assert "train_accuracy" in metrics
        assert "val_accuracy" in metrics
        assert "epochs_trained" in metrics
        assert metrics["epochs_trained"] > 0

    def test_lstm_predict(self, sample_ohlcv_data):
        """Test LSTM prediction."""
        from stockai.core.predictor.lstm_model import LSTMPredictor

        predictor = LSTMPredictor(sequence_length=10, hidden_size=32, num_layers=1)
        predictor.train(sample_ohlcv_data, horizon=3, epochs=3)

        result = predictor.predict(sample_ohlcv_data)

        # Should have required fields
        assert "direction" in result
        assert result["direction"] in ["UP", "DOWN"]
        assert "probability" in result
        assert 0 <= result["probability"] <= 1
        assert result["model"] == "lstm"

    def test_lstm_predict_proba(self, sample_ohlcv_data):
        """Test probability prediction interface."""
        from stockai.core.predictor.lstm_model import LSTMPredictor

        predictor = LSTMPredictor(sequence_length=10, hidden_size=32, num_layers=1)
        predictor.train(sample_ohlcv_data, horizon=3, epochs=3)

        prob = predictor.predict_proba(sample_ohlcv_data)
        assert 0 <= prob <= 1

    def test_lstm_save_load(self, sample_ohlcv_data, tmp_path):
        """Test model serialization."""
        from stockai.core.predictor.lstm_model import LSTMPredictor

        model_path = tmp_path / "lstm_test.pt"

        # Train and save
        predictor = LSTMPredictor(
            sequence_length=10,
            hidden_size=32,
            num_layers=1,
            model_path=model_path,
        )
        predictor.train(sample_ohlcv_data, horizon=3, epochs=3)
        original_result = predictor.predict(sample_ohlcv_data)
        assert predictor.save()

        # Load in new predictor
        new_predictor = LSTMPredictor(model_path=model_path)
        assert new_predictor.load()

        # Should produce same results
        loaded_result = new_predictor.predict(sample_ohlcv_data)
        assert loaded_result["direction"] == original_result["direction"]
        # Allow small floating point differences
        assert abs(loaded_result["probability"] - original_result["probability"]) < 0.01

    def test_lstm_insufficient_data(self):
        """Test error on insufficient sequence data."""
        from stockai.core.predictor.lstm_model import LSTMPredictor

        predictor = LSTMPredictor(sequence_length=20)

        # Data shorter than sequence length
        short_df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=10, freq="B"),
            "open": [100] * 10,
            "high": [101] * 10,
            "low": [99] * 10,
            "close": [100] * 10,
            "volume": [1000000] * 10,
        })

        # Train with too little data - should handle gracefully or error
        with pytest.raises(ValueError):
            predictor.train(short_df)


class TestEnsemblePredictor:
    """Tests for ensemble prediction model.

    Note: Tests that involve LSTM training are skipped due to PyTorch/MPS
    compatibility issues. XGBoost-only tests are run.
    """

    def test_ensemble_init(self):
        """Test ensemble initialization."""
        from stockai.core.predictor.ensemble import EnsemblePredictor

        ensemble = EnsemblePredictor()

        # Should have default weights
        assert ensemble.weights["xgboost"] == 0.4
        assert ensemble.weights["lstm"] == 0.4
        assert ensemble.weights["sentiment"] == 0.2

    def test_ensemble_custom_weights(self):
        """Test ensemble with custom weights."""
        from stockai.core.predictor.ensemble import EnsemblePredictor

        custom_weights = {"xgboost": 0.5, "lstm": 0.3, "sentiment": 0.2}
        ensemble = EnsemblePredictor(weights=custom_weights)

        assert ensemble.weights == custom_weights

    def test_ensemble_weight_normalization(self):
        """Test that weights are normalized if they don't sum to 1."""
        from stockai.core.predictor.ensemble import EnsemblePredictor

        bad_weights = {"xgboost": 1.0, "lstm": 1.0, "sentiment": 0.5}
        ensemble = EnsemblePredictor(weights=bad_weights)

        # Should normalize
        total = sum(ensemble.weights.values())
        assert abs(total - 1.0) < 0.01

    def test_ensemble_xgboost_only(self, sample_ohlcv_data, tmp_path):
        """Test ensemble with XGBoost only (LSTM skipped due to PyTorch issues)."""
        from stockai.core.predictor.ensemble import EnsemblePredictor
        from stockai.core.predictor.xgboost_model import XGBoostPredictor

        xgb_path = tmp_path / "xgb.json"
        lstm_path = tmp_path / "lstm.pt"  # Won't exist

        # Train XGBoost separately
        xgb_predictor = XGBoostPredictor(model_path=xgb_path)
        xgb_predictor.train(sample_ohlcv_data, horizon=3)
        xgb_predictor.save()

        # Create ensemble and load
        ensemble = EnsemblePredictor(
            xgboost_path=xgb_path,
            lstm_path=lstm_path,
        )

        loaded = ensemble.load_models()
        assert loaded["xgboost"]
        assert not loaded["lstm"]

        # Should still produce valid predictions with just XGBoost
        result = ensemble.predict(sample_ohlcv_data)

        assert "direction" in result
        assert result["direction"] in ["UP", "DOWN"]
        assert "probability" in result
        assert "confidence" in result
        assert result["model"] == "ensemble"
        assert result["active_models"] == 1

    def test_ensemble_predict_with_sentiment_xgboost_only(self, sample_ohlcv_data, tmp_path):
        """Test prediction with sentiment modifier (XGBoost only)."""
        from stockai.core.predictor.ensemble import EnsemblePredictor
        from stockai.core.predictor.xgboost_model import XGBoostPredictor

        xgb_path = tmp_path / "xgb.json"
        lstm_path = tmp_path / "lstm.pt"

        # Train XGBoost separately
        xgb_predictor = XGBoostPredictor(model_path=xgb_path)
        xgb_predictor.train(sample_ohlcv_data, horizon=3)
        xgb_predictor.save()

        # Create ensemble
        ensemble = EnsemblePredictor(
            xgboost_path=xgb_path,
            lstm_path=lstm_path,
        )
        ensemble.load_models()

        # Positive sentiment should push toward UP
        result_pos = ensemble.predict(sample_ohlcv_data, sentiment_score=0.8)
        result_neg = ensemble.predict(sample_ohlcv_data, sentiment_score=-0.8)

        # With same underlying data, positive sentiment should give higher probability
        assert result_pos["probability"] > result_neg["probability"]

        # Should include sentiment contribution
        assert "sentiment" in result_pos["contributions"]
        assert result_pos["contributions"]["sentiment"]["score"] == 0.8

    def test_ensemble_confidence_xgboost_only(self, sample_ohlcv_data, tmp_path):
        """Test confidence calibration (XGBoost only)."""
        from stockai.core.predictor.ensemble import EnsemblePredictor
        from stockai.core.predictor.xgboost_model import XGBoostPredictor

        xgb_path = tmp_path / "xgb.json"

        # Train XGBoost
        xgb_predictor = XGBoostPredictor(model_path=xgb_path)
        xgb_predictor.train(sample_ohlcv_data, horizon=3)
        xgb_predictor.save()

        # Create ensemble
        ensemble = EnsemblePredictor(xgboost_path=xgb_path)
        ensemble.load_models()

        result = ensemble.predict(sample_ohlcv_data)

        # Confidence should be between 0 and 1
        assert 0 <= result["confidence"] <= 1

        # Confidence level should be valid
        assert result["confidence_level"] in ["HIGH", "MEDIUM", "LOW"]

    def test_ensemble_no_models_loaded(self, small_ohlcv_data, tmp_path):
        """Test prediction when no models are loaded."""
        from stockai.core.predictor.ensemble import EnsemblePredictor

        ensemble = EnsemblePredictor(
            xgboost_path=tmp_path / "nonexistent.json",
            lstm_path=tmp_path / "nonexistent.pt",
        )

        loaded = ensemble.load_models()
        assert not loaded["xgboost"]
        assert not loaded["lstm"]

        # Prediction should still work (with 0 active models)
        result = ensemble.predict(small_ohlcv_data)
        assert result["active_models"] == 0

    def test_ensemble_model_summary_xgboost_only(self, sample_ohlcv_data, tmp_path):
        """Test model summary extraction (XGBoost only)."""
        from stockai.core.predictor.ensemble import EnsemblePredictor
        from stockai.core.predictor.xgboost_model import XGBoostPredictor

        xgb_path = tmp_path / "xgb.json"

        # Train XGBoost
        xgb_predictor = XGBoostPredictor(model_path=xgb_path)
        xgb_predictor.train(sample_ohlcv_data, horizon=3)
        xgb_predictor.save()

        # Create ensemble and load
        ensemble = EnsemblePredictor(xgboost_path=xgb_path)
        ensemble.load_models()

        summary = ensemble.get_model_summary()

        assert "models_loaded" in summary
        assert "weights" in summary
        assert "xgboost_metrics" in summary
        assert "lstm_metrics" in summary

    @pytest.mark.skip(reason="LSTM training causes segfaults on some PyTorch installations")
    def test_ensemble_train_all(self, sample_ohlcv_data, tmp_path):
        """Test training all models."""
        from stockai.core.predictor.ensemble import EnsemblePredictor

        xgb_path = tmp_path / "xgb.json"
        lstm_path = tmp_path / "lstm.pt"

        ensemble = EnsemblePredictor(
            xgboost_path=xgb_path,
            lstm_path=lstm_path,
        )

        # Override LSTM params for faster training
        results = ensemble.train_all(
            sample_ohlcv_data,
            horizon=3,
            lstm_params={"epochs": 3, "patience": 2},
        )

        # Should have results for both models
        assert "xgboost" in results
        assert "lstm" in results

        # Models should be marked as loaded
        assert ensemble.models_loaded["xgboost"]
        assert ensemble.models_loaded["lstm"]

    @pytest.mark.skip(reason="LSTM training causes segfaults on some PyTorch installations")
    def test_ensemble_save_load_all(self, sample_ohlcv_data, tmp_path):
        """Test saving and loading all models."""
        from stockai.core.predictor.ensemble import EnsemblePredictor

        xgb_path = tmp_path / "xgb.json"
        lstm_path = tmp_path / "lstm.pt"

        # Train and save
        ensemble = EnsemblePredictor(
            xgboost_path=xgb_path,
            lstm_path=lstm_path,
        )

        ensemble.train_all(
            sample_ohlcv_data,
            horizon=3,
            lstm_params={"epochs": 3},
        )

        original_result = ensemble.predict(sample_ohlcv_data)
        save_results = ensemble.save_all()

        assert save_results.get("xgboost", False)
        assert save_results.get("lstm", False)

        # Load in new ensemble
        new_ensemble = EnsemblePredictor(
            xgboost_path=xgb_path,
            lstm_path=lstm_path,
        )

        loaded = new_ensemble.load_models()
        assert loaded["xgboost"]
        assert loaded["lstm"]

        # Should produce similar results
        loaded_result = new_ensemble.predict(sample_ohlcv_data)
        assert loaded_result["direction"] == original_result["direction"]


class TestPredictCLI:
    """Tests for CLI predict command."""

    def test_predict_command_help(self):
        """Test predict command help text."""
        from typer.testing import CliRunner
        from stockai.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["predict", "--help"])

        assert result.exit_code == 0
        assert "stock predict" in result.output or "Predict" in result.output
        assert "--horizon" in result.output

    def test_predict_no_models_warning(self, tmp_path, monkeypatch):
        """Test predict shows warning when no models exist."""
        from typer.testing import CliRunner
        from stockai.cli.main import app
        from stockai import config

        # Mock settings to use temp path
        def mock_settings():
            settings = config.Settings()
            settings._project_root = tmp_path
            return settings

        monkeypatch.setattr("stockai.cli.main.get_settings", mock_settings)

        runner = CliRunner()
        result = runner.invoke(app, ["predict", "BBCA"])

        # Should warn about missing models or show placeholder
        assert "Models Not Trained" in result.output or "Warning" in result.output or "UNKNOWN" in result.output


class TestTrainCLI:
    """Tests for CLI train command."""

    def test_train_command_help(self):
        """Test train command help text."""
        from typer.testing import CliRunner
        from stockai.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["train", "--help"])

        assert result.exit_code == 0
        assert "Train" in result.output
        assert "--symbol" in result.output
        assert "--horizon" in result.output
        assert "--force" in result.output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
