"""Support and Resistance Detection Module.

Detects key price levels using pivot point analysis with scipy.signal.argrelextrema.
Identifies nearest support/resistance and calculates distance for gate validation.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema


@dataclass
class SupportResistanceResult:
    """Result of support/resistance analysis."""

    current_price: float
    supports: list[float]  # Up to 3 levels
    resistances: list[float]  # Up to 3 levels
    nearest_support: float | None
    nearest_resistance: float | None
    distance_to_support_pct: float | None
    is_near_support: bool
    suggested_stop_loss: float


def find_support_resistance(
    df: pd.DataFrame,
    lookback: int = 60,
    order: int = 5,
    max_levels: int = 3,
    near_support_threshold: float = 5.0,
    stop_loss_pct: float = 3.0,
    max_stop_loss_pct: float = 8.0,
) -> SupportResistanceResult:
    """Find support and resistance levels from price data.

    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        lookback: Number of days to analyze
        order: Number of points on each side for pivot detection
        max_levels: Maximum number of support/resistance levels to return
        near_support_threshold: Percentage threshold to consider "near support"
        stop_loss_pct: Percentage below support for stop loss
        max_stop_loss_pct: Maximum stop loss percentage from current price

    Returns:
        SupportResistanceResult with detected levels and analysis
    """
    if len(df) < lookback:
        lookback = len(df)

    recent = df.tail(lookback).copy()
    current_price = recent["close"].iloc[-1]

    # Find pivot highs (resistance candidates)
    highs = recent["high"].values
    pivot_high_idx = argrelextrema(highs, np.greater, order=order)[0]
    resistance_candidates = [highs[i] for i in pivot_high_idx]

    # Find pivot lows (support candidates)
    lows = recent["low"].values
    pivot_low_idx = argrelextrema(lows, np.less, order=order)[0]
    support_candidates = [lows[i] for i in pivot_low_idx]

    # Filter levels within 20% of current price
    price_range_pct = 20.0
    lower_bound = current_price * (1 - price_range_pct / 100)
    upper_bound = current_price * (1 + price_range_pct / 100)

    # Filter and sort supports (below current price, closest first)
    supports = sorted(
        [s for s in support_candidates if lower_bound <= s < current_price],
        reverse=True,  # Closest to price first
    )[:max_levels]

    # Filter and sort resistances (above current price, closest first)
    resistances = sorted(
        [r for r in resistance_candidates if current_price < r <= upper_bound],
    )[:max_levels]

    # Nearest support and resistance
    nearest_support = supports[0] if supports else None
    nearest_resistance = resistances[0] if resistances else None

    # Calculate distance to support
    if nearest_support is not None:
        distance_to_support_pct = (
            (current_price - nearest_support) / current_price
        ) * 100
    else:
        distance_to_support_pct = None

    # Check if near support
    is_near_support = (
        distance_to_support_pct is not None
        and distance_to_support_pct <= near_support_threshold
    )

    # Calculate suggested stop loss
    if nearest_support is not None:
        # 3% below the nearest support
        sl_from_support = nearest_support * (1 - stop_loss_pct / 100)
        # But cap at 8% below current price
        sl_max = current_price * (1 - max_stop_loss_pct / 100)
        suggested_stop_loss = max(sl_from_support, sl_max)
    else:
        # No support found, use 8% below current price
        suggested_stop_loss = current_price * (1 - max_stop_loss_pct / 100)

    return SupportResistanceResult(
        current_price=float(current_price),
        supports=supports,
        resistances=resistances,
        nearest_support=nearest_support,
        nearest_resistance=nearest_resistance,
        distance_to_support_pct=distance_to_support_pct,
        is_near_support=is_near_support,
        suggested_stop_loss=float(suggested_stop_loss),
    )
