"""Feature Engineering Pipeline for StockAI Prediction Models.

Generates ~105 features from raw price data for ML models:
- Price features: returns, volatility, range
- Technical features: RSI, MACD, BB, Stochastic, ATR
- Volume features: volume ratio, OBV, accumulation
- Market features: IHSG correlation, sector performance
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from stockai.data.sectors import get_sector_relative_strength, get_stock_sector

logger = logging.getLogger(__name__)

# Feature groups
PRICE_FEATURES = [
    "return_1d", "return_5d", "return_10d", "return_20d",
    "volatility_5d", "volatility_10d", "volatility_20d",
    "high_low_range", "open_close_range",
    "price_sma5_ratio", "price_sma10_ratio", "price_sma20_ratio",
    "sma5_sma20_ratio", "price_momentum_5d", "price_momentum_10d",
]

TECHNICAL_FEATURES = [
    "rsi_14", "rsi_7", "rsi_21",
    "macd", "macd_signal", "macd_histogram",
    "bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_position",
    "stoch_k", "stoch_d",
    "atr_14", "atr_7",
    "cci_14", "cci_20",
    "williams_r",
    "adx_14", "di_plus", "di_minus",
]

VOLUME_FEATURES = [
    "volume_sma5_ratio", "volume_sma10_ratio", "volume_sma20_ratio",
    "obv", "obv_change", "obv_sma5_ratio",
    "accumulation_distribution", "ad_change",
    "volume_price_trend", "mfi_14",
    "force_index", "ease_of_movement",
]

MARKET_FEATURES = [
    "ihsg_correlation_20d", "ihsg_beta_20d",
    "sector_relative_strength", "market_regime",
]

ALL_FEATURES = PRICE_FEATURES + TECHNICAL_FEATURES + VOLUME_FEATURES + MARKET_FEATURES


class FeatureEngineer:
    """Feature engineering pipeline for stock prediction.

    Generates comprehensive feature set from OHLCV data:
    - Price-based features (returns, volatility, ranges)
    - Technical indicators (RSI, MACD, Bollinger Bands, etc.)
    - Volume-based features (OBV, A/D, MFI, etc.)
    - Market context features (IHSG correlation, sector strength)
    """

    def __init__(
        self,
        include_market_features: bool = True,
        normalize: bool = True,
        fill_method: str = "ffill",
    ):
        """Initialize feature engineer.

        Args:
            include_market_features: Whether to include market-level features
            normalize: Whether to normalize features
            fill_method: Method for handling missing values ('ffill', 'drop')
        """
        self.include_market_features = include_market_features
        self.normalize = normalize
        self.fill_method = fill_method
        self.feature_stats: dict[str, dict[str, float]] = {}

    def generate_features(
        self,
        df: pd.DataFrame,
        ihsg_df: pd.DataFrame | None = None,
        symbol: str | None = None,
    ) -> pd.DataFrame:
        """Generate all features from OHLCV data.

        Args:
            df: DataFrame with columns: open, high, low, close, volume
            ihsg_df: Optional IHSG index data for market features
            symbol: Optional stock symbol for sector-based features

        Returns:
            DataFrame with all generated features
        """
        if df.empty:
            logger.warning("Empty DataFrame provided")
            return pd.DataFrame()

        # Ensure required columns
        required = ["open", "high", "low", "close", "volume"]
        if not all(col in df.columns for col in required):
            raise ValueError(f"DataFrame must have columns: {required}")

        # Start with copy
        features = df.copy()

        # Generate feature groups
        features = self._add_price_features(features)
        features = self._add_technical_features(features)
        features = self._add_volume_features(features)

        if self.include_market_features and ihsg_df is not None:
            features = self._add_market_features(features, ihsg_df, symbol)
        else:
            # Add placeholder market features
            for feat in MARKET_FEATURES:
                features[feat] = 0.0

        # Handle missing values
        features = self._handle_missing(features)

        # Normalize if requested
        if self.normalize:
            features = self._normalize_features(features)

        # Select only feature columns
        feature_cols = [c for c in ALL_FEATURES if c in features.columns]
        return features[feature_cols]

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features."""
        close = df["close"]
        high = df["high"]
        low = df["low"]
        open_ = df["open"]

        # Returns
        df["return_1d"] = close.pct_change(1)
        df["return_5d"] = close.pct_change(5)
        df["return_10d"] = close.pct_change(10)
        df["return_20d"] = close.pct_change(20)

        # Volatility (rolling std of returns)
        returns = close.pct_change()
        df["volatility_5d"] = returns.rolling(5).std()
        df["volatility_10d"] = returns.rolling(10).std()
        df["volatility_20d"] = returns.rolling(20).std()

        # Ranges
        df["high_low_range"] = (high - low) / close
        df["open_close_range"] = (close - open_) / open_

        # Price relative to SMAs
        sma5 = close.rolling(5).mean()
        sma10 = close.rolling(10).mean()
        sma20 = close.rolling(20).mean()

        df["price_sma5_ratio"] = close / sma5 - 1
        df["price_sma10_ratio"] = close / sma10 - 1
        df["price_sma20_ratio"] = close / sma20 - 1
        df["sma5_sma20_ratio"] = sma5 / sma20 - 1

        # Momentum
        df["price_momentum_5d"] = close - close.shift(5)
        df["price_momentum_10d"] = close - close.shift(10)

        return df

    def _add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicator features."""
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # RSI
        df["rsi_14"] = self._calculate_rsi(close, 14)
        df["rsi_7"] = self._calculate_rsi(close, 7)
        df["rsi_21"] = self._calculate_rsi(close, 21)

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]

        # Bollinger Bands
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        df["bb_upper"] = sma20 + 2 * std20
        df["bb_middle"] = sma20
        df["bb_lower"] = sma20 - 2 * std20
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
        df["bb_position"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

        # Stochastic
        low14 = low.rolling(14).min()
        high14 = high.rolling(14).max()
        df["stoch_k"] = 100 * (close - low14) / (high14 - low14)
        df["stoch_d"] = df["stoch_k"].rolling(3).mean()

        # ATR (Average True Range)
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        df["atr_14"] = tr.rolling(14).mean()
        df["atr_7"] = tr.rolling(7).mean()

        # CCI (Commodity Channel Index)
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(14).mean()
        mad = typical_price.rolling(14).apply(lambda x: np.abs(x - x.mean()).mean())
        df["cci_14"] = (typical_price - sma_tp) / (0.015 * mad)
        sma_tp20 = typical_price.rolling(20).mean()
        mad20 = typical_price.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
        df["cci_20"] = (typical_price - sma_tp20) / (0.015 * mad20)

        # Williams %R
        df["williams_r"] = -100 * (high14 - close) / (high14 - low14)

        # ADX (Average Directional Index)
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        atr14 = tr.rolling(14).mean()
        df["di_plus"] = 100 * (plus_dm.rolling(14).mean() / atr14)
        df["di_minus"] = 100 * (minus_dm.rolling(14).mean() / atr14)
        dx = 100 * (df["di_plus"] - df["di_minus"]).abs() / (df["di_plus"] + df["di_minus"])
        df["adx_14"] = dx.rolling(14).mean()

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # Volume relative to SMAs
        vol_sma5 = volume.rolling(5).mean()
        vol_sma10 = volume.rolling(10).mean()
        vol_sma20 = volume.rolling(20).mean()

        df["volume_sma5_ratio"] = volume / vol_sma5
        df["volume_sma10_ratio"] = volume / vol_sma10
        df["volume_sma20_ratio"] = volume / vol_sma20

        # OBV (On-Balance Volume)
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        df["obv"] = obv
        df["obv_change"] = obv.pct_change(5)
        df["obv_sma5_ratio"] = obv / obv.rolling(5).mean()

        # Accumulation/Distribution
        clv = ((close - low) - (high - close)) / (high - low)
        clv = clv.fillna(0)
        ad = (clv * volume).cumsum()
        df["accumulation_distribution"] = ad
        df["ad_change"] = ad.diff(5)

        # Volume Price Trend
        vpt = (close.pct_change() * volume).cumsum()
        df["volume_price_trend"] = vpt

        # Money Flow Index
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        pos_flow = money_flow.where(typical_price > typical_price.shift(), 0)
        neg_flow = money_flow.where(typical_price < typical_price.shift(), 0)
        pos_sum = pos_flow.rolling(14).sum()
        neg_sum = neg_flow.rolling(14).sum()
        mfi = 100 - (100 / (1 + pos_sum / neg_sum.replace(0, 1)))
        df["mfi_14"] = mfi

        # Force Index
        df["force_index"] = close.diff() * volume

        # Ease of Movement
        distance = ((high + low) / 2) - ((high.shift() + low.shift()) / 2)
        box_ratio = (volume / 1e6) / (high - low)
        df["ease_of_movement"] = distance / box_ratio.replace(0, 1)

        return df

    def _add_market_features(
        self,
        df: pd.DataFrame,
        ihsg_df: pd.DataFrame,
        symbol: str | None = None,
    ) -> pd.DataFrame:
        """Add market-level features.

        Args:
            df: Stock OHLCV DataFrame
            ihsg_df: IHSG index DataFrame
            symbol: Stock symbol for sector-based features
        """
        # Ensure IHSG data is aligned
        ihsg_close = ihsg_df["close"].reindex(df.index, method="ffill")
        stock_returns = df["close"].pct_change()
        ihsg_returns = ihsg_close.pct_change()

        # Correlation with IHSG
        correlation = stock_returns.rolling(20).corr(ihsg_returns)
        df["ihsg_correlation_20d"] = correlation

        # Beta
        covariance = stock_returns.rolling(20).cov(ihsg_returns)
        variance = ihsg_returns.rolling(20).var()
        df["ihsg_beta_20d"] = covariance / variance.replace(0, 1)

        # Sector Relative Strength (real implementation)
        if symbol:
            try:
                sector_rs = get_sector_relative_strength(df, symbol, period=20)
                df["sector_relative_strength"] = sector_rs
                sector = get_stock_sector(symbol)
                if sector:
                    logger.debug(f"Calculated sector RS for {symbol} ({sector})")
            except Exception as e:
                logger.warning(f"Could not calculate sector RS for {symbol}: {e}")
                df["sector_relative_strength"] = 0.0
        else:
            df["sector_relative_strength"] = 0.0

        # Market regime (simplified: based on IHSG trend)
        ihsg_sma50 = ihsg_close.rolling(50).mean()
        df["market_regime"] = (ihsg_close > ihsg_sma50).astype(float)

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()

        rs = avg_gain / avg_loss.replace(0, 1)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features."""
        if self.fill_method == "ffill":
            df = df.ffill()
            # Drop remaining NaN rows (from beginning due to rolling)
            df = df.dropna()
        elif self.fill_method == "drop":
            df = df.dropna()

        # Replace infinities with NaN and forward fill
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.ffill().bfill()

        return df

    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize features using z-score normalization."""
        for col in df.columns:
            if col in ALL_FEATURES:
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    df[col] = (df[col] - mean) / std
                    self.feature_stats[col] = {"mean": mean, "std": std}
                else:
                    df[col] = 0.0
                    self.feature_stats[col] = {"mean": mean, "std": 1.0}
        return df

    def get_feature_importance(
        self,
        model_importance: dict[str, float] | None = None,
    ) -> pd.DataFrame:
        """Get feature importance summary.

        Args:
            model_importance: Optional dict of feature importances from model

        Returns:
            DataFrame with feature names, groups, and importance scores
        """
        data = []
        for feat in ALL_FEATURES:
            if feat in PRICE_FEATURES:
                group = "price"
            elif feat in TECHNICAL_FEATURES:
                group = "technical"
            elif feat in VOLUME_FEATURES:
                group = "volume"
            else:
                group = "market"

            importance = model_importance.get(feat, 0.0) if model_importance else 0.0
            data.append({"feature": feat, "group": group, "importance": importance})

        return pd.DataFrame(data).sort_values("importance", ascending=False)


def generate_features(
    ohlcv_data: pd.DataFrame,
    ihsg_data: pd.DataFrame | None = None,
    normalize: bool = True,
) -> pd.DataFrame:
    """Convenience function to generate features.

    Args:
        ohlcv_data: DataFrame with OHLCV columns
        ihsg_data: Optional IHSG index data
        normalize: Whether to normalize features

    Returns:
        DataFrame with all generated features
    """
    engineer = FeatureEngineer(
        include_market_features=ihsg_data is not None,
        normalize=normalize,
    )
    return engineer.generate_features(ohlcv_data, ihsg_data)


def create_target(
    df: pd.DataFrame,
    horizon: int = 3,
    threshold: float = 0.0,
) -> pd.Series:
    """Create binary target for prediction.

    Args:
        df: DataFrame with 'close' column
        horizon: Days ahead to predict
        threshold: Minimum return for UP classification

    Returns:
        Series with binary target (1=UP, 0=DOWN)
    """
    future_return = df["close"].shift(-horizon) / df["close"] - 1
    target = (future_return > threshold).astype(int)
    return target
