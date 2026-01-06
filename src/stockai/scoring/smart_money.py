"""Smart Money Score Module.

Analyzes institutional accumulation/distribution patterns through
volume-price relationship analysis, OBV trends, and MFI signals.

Score range: -2.0 (heavy distribution) to 5.0 (strong accumulation)
"""

from dataclasses import dataclass

import pandas as pd
import ta


@dataclass
class SmartMoneyResult:
    """Result of smart money analysis."""

    score: float  # -2.0 to 5.0
    accumulation_days: int
    distribution_days: int
    net_accumulation: int
    obv_trend: str  # BULLISH, NEUTRAL, BEARISH
    mfi: float
    mfi_signal: str  # OVERBOUGHT, NEUTRAL, OVERSOLD
    unusual_volume: str  # HIGH, NORMAL, LOW
    interpretation: str  # ACCUMULATION, NEUTRAL, DISTRIBUTION


def calculate_smart_money_score(
    df: pd.DataFrame,
    volume_threshold: float = 1.2,
    lookback: int = 20,
) -> SmartMoneyResult:
    """Calculate Smart Money Score from OHLCV data.

    Args:
        df: DataFrame with columns: open, high, low, close, volume
        volume_threshold: Multiplier for average volume to detect unusual activity
        lookback: Number of days for analysis

    Returns:
        SmartMoneyResult with score and component analysis
    """
    if len(df) < lookback:
        lookback = len(df)

    recent = df.tail(lookback).copy()

    # Calculate daily returns
    recent["return"] = recent["close"].pct_change()

    # Calculate average volume
    avg_volume = recent["volume"].mean()

    # Volume ratio (current vs average)
    recent["volume_ratio"] = recent["volume"] / avg_volume

    # Accumulation days: price up + volume > threshold
    accumulation_mask = (recent["return"] > 0) & (
        recent["volume_ratio"] > volume_threshold
    )
    accumulation_days = accumulation_mask.sum()

    # Distribution days: price down + volume > threshold
    distribution_mask = (recent["return"] < 0) & (
        recent["volume_ratio"] > volume_threshold
    )
    distribution_days = distribution_mask.sum()

    net_accumulation = accumulation_days - distribution_days

    # Calculate OBV using ta library
    obv_indicator = ta.volume.OnBalanceVolumeIndicator(recent["close"], recent["volume"])
    obv_values = obv_indicator.on_balance_volume()

    if obv_values is not None and len(obv_values) >= 5:
        obv_sma = obv_values.rolling(5).mean()
        obv_current = obv_values.iloc[-1]
        obv_sma_current = obv_sma.iloc[-1]

        if pd.notna(obv_current) and pd.notna(obv_sma_current):
            if obv_current > obv_sma_current * 1.05:
                obv_trend = "BULLISH"
            elif obv_current < obv_sma_current * 0.95:
                obv_trend = "BEARISH"
            else:
                obv_trend = "NEUTRAL"
        else:
            obv_trend = "NEUTRAL"
    else:
        obv_trend = "NEUTRAL"

    # Calculate MFI using ta library
    mfi_indicator = ta.volume.MFIIndicator(
        recent["high"], recent["low"], recent["close"], recent["volume"], window=14
    )
    mfi_series = mfi_indicator.money_flow_index()

    if mfi_series is not None and len(mfi_series) > 0:
        mfi_value = mfi_series.iloc[-1]
        if pd.isna(mfi_value):
            mfi_value = 50.0
    else:
        mfi_value = 50.0

    # MFI signal interpretation
    if mfi_value >= 80:
        mfi_signal = "OVERBOUGHT"
    elif mfi_value <= 20:
        mfi_signal = "OVERSOLD"
    else:
        mfi_signal = "NEUTRAL"

    # Detect unusual volume (latest bar)
    latest_volume_ratio = recent["volume_ratio"].iloc[-1] if len(recent) > 0 else 1.0
    if pd.isna(latest_volume_ratio):
        latest_volume_ratio = 1.0

    if latest_volume_ratio >= 2.0:
        unusual_volume = "HIGH"
    elif latest_volume_ratio <= 0.5:
        unusual_volume = "LOW"
    else:
        unusual_volume = "NORMAL"

    # Calculate composite score (-2.0 to 5.0)
    score = 0.0

    # Net accumulation contribution (-1 to 2)
    if net_accumulation > 3:
        score += 2.0
    elif net_accumulation > 0:
        score += 1.0
    elif net_accumulation < -3:
        score -= 1.0
    elif net_accumulation < 0:
        score -= 0.5

    # OBV trend contribution (-0.5 to 1.5)
    if obv_trend == "BULLISH":
        score += 1.5
    elif obv_trend == "BEARISH":
        score -= 0.5

    # MFI contribution (-0.5 to 1.0)
    if mfi_signal == "OVERSOLD":
        score += 1.0  # Buying opportunity
    elif mfi_signal == "OVERBOUGHT":
        score -= 0.5  # Caution

    # Unusual volume bonus (0 to 0.5)
    if unusual_volume == "HIGH" and net_accumulation > 0:
        score += 0.5

    # Clamp to valid range
    score = max(-2.0, min(5.0, score))

    # Determine interpretation
    if score >= 3.0:
        interpretation = "ACCUMULATION"
    elif score <= 0:
        interpretation = "DISTRIBUTION"
    else:
        interpretation = "NEUTRAL"

    return SmartMoneyResult(
        score=score,
        accumulation_days=int(accumulation_days),
        distribution_days=int(distribution_days),
        net_accumulation=int(net_accumulation),
        obv_trend=obv_trend,
        mfi=float(mfi_value),
        mfi_signal=mfi_signal,
        unusual_volume=unusual_volume,
        interpretation=interpretation,
    )
