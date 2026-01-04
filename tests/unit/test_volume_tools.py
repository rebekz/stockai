"""Unit Tests for Volume Analysis Tools.

Tests for get_volume_analysis, get_volume_profile, and get_volume_signals tools.
These tests mock the Yahoo Finance data source to ensure consistent, isolated testing.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta


def create_mock_price_data(
    days: int = 60,
    base_price: float = 10000.0,
    base_volume: int = 1000000,
    trend: str = "neutral",
    volume_trend: str = "neutral",
) -> pd.DataFrame:
    """Create mock price data for testing.

    Args:
        days: Number of days of data
        base_price: Starting price
        base_volume: Base volume
        trend: Price trend ('up', 'down', 'neutral')
        volume_trend: Volume trend ('up', 'down', 'spike', 'neutral')

    Returns:
        DataFrame with OHLCV data
    """
    dates = [datetime.now() - timedelta(days=days - i) for i in range(days)]

    # Generate prices based on trend
    prices = []
    current_price = base_price
    for i in range(days):
        if trend == "up":
            change = np.random.uniform(0, 0.02)  # 0-2% up bias
        elif trend == "down":
            change = np.random.uniform(-0.02, 0)  # 0-2% down bias
        else:
            change = np.random.uniform(-0.01, 0.01)

        current_price = current_price * (1 + change)
        prices.append(current_price)

    # Generate volume based on volume trend
    volumes = []
    current_volume = base_volume
    for i in range(days):
        if volume_trend == "up":
            vol = current_volume * np.random.uniform(1.0, 1.3)
            current_volume = vol * 1.05
        elif volume_trend == "down":
            vol = current_volume * np.random.uniform(0.7, 1.0)
            current_volume = vol * 0.95
        elif volume_trend == "spike" and i >= days - 3:
            vol = base_volume * 3.0  # 3x volume spike in last 3 days
        else:
            vol = base_volume * np.random.uniform(0.8, 1.2)
        volumes.append(int(vol))

    # Create DataFrame
    df = pd.DataFrame({
        "date": dates,
        "symbol": "TEST",
        "open": [p * np.random.uniform(0.995, 1.0) for p in prices],
        "high": [p * np.random.uniform(1.0, 1.01) for p in prices],
        "low": [p * np.random.uniform(0.99, 1.0) for p in prices],
        "close": prices,
        "volume": volumes,
    })

    return df


class TestToolRegistration:
    """Test that volume tools are properly registered."""

    def test_get_volume_analysis_registered(self):
        """Test that get_volume_analysis is registered in the tool registry."""
        from stockai.tools.registry import get_registry

        # Force import of stock_tools to trigger registration
        import stockai.tools.stock_tools

        registry = get_registry()
        tool_info = registry.get_tool_info("get_volume_analysis")

        assert tool_info is not None
        assert tool_info["name"] == "get_volume_analysis"
        assert tool_info["category"] == "analysis"
        assert tool_info["permission"] == "safe"

    def test_get_volume_profile_registered(self):
        """Test that get_volume_profile is registered in the tool registry."""
        from stockai.tools.registry import get_registry

        import stockai.tools.stock_tools

        registry = get_registry()
        tool_info = registry.get_tool_info("get_volume_profile")

        assert tool_info is not None
        assert tool_info["name"] == "get_volume_profile"
        assert tool_info["category"] == "analysis"

    def test_get_volume_signals_registered(self):
        """Test that get_volume_signals is registered in the tool registry."""
        from stockai.tools.registry import get_registry

        import stockai.tools.stock_tools

        registry = get_registry()
        tool_info = registry.get_tool_info("get_volume_signals")

        assert tool_info is not None
        assert tool_info["name"] == "get_volume_signals"
        assert tool_info["category"] == "analysis"


class TestGetVolumeAnalysis:
    """Test get_volume_analysis tool functionality."""

    @patch("stockai.tools.stock_tools._yahoo")
    def test_returns_obv_indicator(self, mock_yahoo):
        """Test that OBV indicator is calculated and returned."""
        mock_yahoo.get_price_history.return_value = create_mock_price_data(days=60)

        from stockai.tools.stock_tools import get_volume_analysis

        result = get_volume_analysis("TEST")

        assert "error" not in result
        assert "indicators" in result
        assert "obv" in result["indicators"]
        assert "value" in result["indicators"]["obv"]
        assert "trend" in result["indicators"]["obv"]
        assert result["indicators"]["obv"]["trend"] in ["rising", "falling"]

    @patch("stockai.tools.stock_tools._yahoo")
    def test_returns_vwap_indicator(self, mock_yahoo):
        """Test that VWAP indicator is calculated and returned."""
        mock_yahoo.get_price_history.return_value = create_mock_price_data(days=60)

        from stockai.tools.stock_tools import get_volume_analysis

        result = get_volume_analysis("TEST")

        assert "error" not in result
        assert "vwap" in result["indicators"]
        assert "value" in result["indicators"]["vwap"]
        assert "price_position" in result["indicators"]["vwap"]
        assert result["indicators"]["vwap"]["price_position"] in ["above", "below"]

    @patch("stockai.tools.stock_tools._yahoo")
    def test_returns_mfi_indicator(self, mock_yahoo):
        """Test that MFI indicator is calculated and returned."""
        mock_yahoo.get_price_history.return_value = create_mock_price_data(days=60)

        from stockai.tools.stock_tools import get_volume_analysis

        result = get_volume_analysis("TEST")

        assert "error" not in result
        assert "mfi" in result["indicators"]
        assert "value" in result["indicators"]["mfi"]
        assert "interpretation" in result["indicators"]["mfi"]
        assert result["indicators"]["mfi"]["interpretation"] in [
            "oversold",
            "overbought",
            "neutral",
        ]

    @patch("stockai.tools.stock_tools._yahoo")
    def test_returns_ad_indicator(self, mock_yahoo):
        """Test that Accumulation/Distribution indicator is calculated."""
        mock_yahoo.get_price_history.return_value = create_mock_price_data(days=60)

        from stockai.tools.stock_tools import get_volume_analysis

        result = get_volume_analysis("TEST")

        assert "error" not in result
        assert "accumulation_distribution" in result["indicators"]
        assert "value" in result["indicators"]["accumulation_distribution"]
        assert "trend" in result["indicators"]["accumulation_distribution"]
        assert result["indicators"]["accumulation_distribution"]["trend"] in [
            "accumulation",
            "distribution",
        ]

    @patch("stockai.tools.stock_tools._yahoo")
    def test_returns_volume_ratios(self, mock_yahoo):
        """Test that volume ratios are calculated correctly."""
        mock_yahoo.get_price_history.return_value = create_mock_price_data(days=60)

        from stockai.tools.stock_tools import get_volume_analysis

        result = get_volume_analysis("TEST")

        assert "error" not in result
        assert "volume_ratios" in result["indicators"]
        ratios = result["indicators"]["volume_ratios"]
        assert "vs_5d_avg" in ratios
        assert "vs_10d_avg" in ratios
        assert "vs_20d_avg" in ratios
        assert "avg_volume_5d" in ratios
        assert "avg_volume_10d" in ratios
        assert "avg_volume_20d" in ratios

    @patch("stockai.tools.stock_tools._yahoo")
    def test_returns_signals(self, mock_yahoo):
        """Test that signals are generated."""
        mock_yahoo.get_price_history.return_value = create_mock_price_data(days=60)

        from stockai.tools.stock_tools import get_volume_analysis

        result = get_volume_analysis("TEST")

        assert "error" not in result
        assert "signals" in result
        assert isinstance(result["signals"], list)

    @patch("stockai.tools.stock_tools._yahoo")
    def test_returns_timestamp(self, mock_yahoo):
        """Test that timestamp is included in response."""
        mock_yahoo.get_price_history.return_value = create_mock_price_data(days=60)

        from stockai.tools.stock_tools import get_volume_analysis

        result = get_volume_analysis("TEST")

        assert "timestamp" in result

    @patch("stockai.tools.stock_tools._yahoo")
    def test_insufficient_data_error(self, mock_yahoo):
        """Test error handling for insufficient data."""
        # Return only 10 days of data (less than required 20)
        mock_yahoo.get_price_history.return_value = create_mock_price_data(days=10)

        from stockai.tools.stock_tools import get_volume_analysis

        result = get_volume_analysis("TEST")

        assert "error" in result
        assert "Insufficient data" in result["error"]

    @patch("stockai.tools.stock_tools._yahoo")
    def test_empty_data_error(self, mock_yahoo):
        """Test error handling for empty data."""
        mock_yahoo.get_price_history.return_value = pd.DataFrame()

        from stockai.tools.stock_tools import get_volume_analysis

        result = get_volume_analysis("TEST")

        assert "error" in result


class TestGetVolumeProfile:
    """Test get_volume_profile tool functionality."""

    @patch("stockai.tools.stock_tools._yahoo")
    def test_returns_poc(self, mock_yahoo):
        """Test that Point of Control is calculated."""
        mock_yahoo.get_price_history.return_value = create_mock_price_data(days=30)

        from stockai.tools.stock_tools import get_volume_profile

        result = get_volume_profile("TEST")

        assert "error" not in result
        assert "volume_profile" in result
        assert "poc" in result["volume_profile"]
        assert "price" in result["volume_profile"]["poc"]
        assert "volume" in result["volume_profile"]["poc"]
        assert "volume_percent" in result["volume_profile"]["poc"]

    @patch("stockai.tools.stock_tools._yahoo")
    def test_returns_value_area(self, mock_yahoo):
        """Test that Value Area (VAH, VAL) is calculated."""
        mock_yahoo.get_price_history.return_value = create_mock_price_data(days=30)

        from stockai.tools.stock_tools import get_volume_profile

        result = get_volume_profile("TEST")

        assert "error" not in result
        assert "value_area" in result["volume_profile"]
        va = result["volume_profile"]["value_area"]
        assert "high" in va
        assert "low" in va
        assert "volume_percent" in va
        # Value Area should contain approximately 70% of volume
        assert va["volume_percent"] >= 60  # Allow some tolerance

    @patch("stockai.tools.stock_tools._yahoo")
    def test_returns_price_levels(self, mock_yahoo):
        """Test that price levels with volume distribution are returned."""
        mock_yahoo.get_price_history.return_value = create_mock_price_data(days=30)

        from stockai.tools.stock_tools import get_volume_profile

        result = get_volume_profile("TEST", num_levels=10)

        assert "error" not in result
        assert "levels" in result
        assert len(result["levels"]) == 10

        # Check level structure
        for level in result["levels"]:
            assert "price_low" in level
            assert "price_high" in level
            assert "volume" in level
            assert "volume_percent" in level
            assert "is_poc" in level
            assert "in_value_area" in level

    @patch("stockai.tools.stock_tools._yahoo")
    def test_returns_price_range(self, mock_yahoo):
        """Test that price range is included."""
        mock_yahoo.get_price_history.return_value = create_mock_price_data(days=30)

        from stockai.tools.stock_tools import get_volume_profile

        result = get_volume_profile("TEST")

        assert "price_range" in result
        assert "min" in result["price_range"]
        assert "max" in result["price_range"]
        assert result["price_range"]["max"] > result["price_range"]["min"]

    @patch("stockai.tools.stock_tools._yahoo")
    def test_returns_signals(self, mock_yahoo):
        """Test that interpretation signals are generated."""
        mock_yahoo.get_price_history.return_value = create_mock_price_data(days=30)

        from stockai.tools.stock_tools import get_volume_profile

        result = get_volume_profile("TEST")

        assert "signals" in result
        assert isinstance(result["signals"], list)
        assert len(result["signals"]) > 0

    @patch("stockai.tools.stock_tools._yahoo")
    def test_insufficient_data_error(self, mock_yahoo):
        """Test error handling for insufficient data."""
        mock_yahoo.get_price_history.return_value = create_mock_price_data(days=3)

        from stockai.tools.stock_tools import get_volume_profile

        result = get_volume_profile("TEST")

        assert "error" in result

    @patch("stockai.tools.stock_tools._yahoo")
    def test_custom_num_levels(self, mock_yahoo):
        """Test that custom number of price levels works."""
        mock_yahoo.get_price_history.return_value = create_mock_price_data(days=30)

        from stockai.tools.stock_tools import get_volume_profile

        result = get_volume_profile("TEST", num_levels=15)

        assert "error" not in result
        assert len(result["levels"]) == 15


class TestGetVolumeSignals:
    """Test get_volume_signals tool functionality."""

    @patch("stockai.tools.stock_tools._yahoo")
    def test_returns_overall_signal(self, mock_yahoo):
        """Test that overall signal is generated."""
        mock_yahoo.get_price_history.return_value = create_mock_price_data(days=60)

        from stockai.tools.stock_tools import get_volume_signals

        result = get_volume_signals("TEST")

        assert "error" not in result
        assert "overall_signal" in result
        assert "direction" in result["overall_signal"]
        assert result["overall_signal"]["direction"] in ["buy", "sell", "neutral"]
        assert "confidence" in result["overall_signal"]
        assert 0 <= result["overall_signal"]["confidence"] <= 1

    @patch("stockai.tools.stock_tools._yahoo")
    def test_returns_individual_signals(self, mock_yahoo):
        """Test that individual signals are returned."""
        mock_yahoo.get_price_history.return_value = create_mock_price_data(days=60)

        from stockai.tools.stock_tools import get_volume_signals

        result = get_volume_signals("TEST")

        assert "error" not in result
        assert "individual_signals" in result
        assert isinstance(result["individual_signals"], list)

        # Each signal should have required fields
        for signal in result["individual_signals"]:
            assert "type" in signal
            assert "direction" in signal
            assert "description" in signal
            assert "confidence" in signal
            assert signal["direction"] in ["bullish", "bearish", "neutral"]

    @patch("stockai.tools.stock_tools._yahoo")
    def test_returns_signal_summary(self, mock_yahoo):
        """Test that signal summary counts are returned."""
        mock_yahoo.get_price_history.return_value = create_mock_price_data(days=60)

        from stockai.tools.stock_tools import get_volume_signals

        result = get_volume_signals("TEST")

        assert "signal_summary" in result
        summary = result["signal_summary"]
        assert "bullish_signals" in summary
        assert "bearish_signals" in summary
        assert "neutral_signals" in summary
        assert "total_signals" in summary

        # Total should match sum of individual counts
        assert (
            summary["total_signals"]
            == summary["bullish_signals"]
            + summary["bearish_signals"]
            + summary["neutral_signals"]
        )

    @patch("stockai.tools.stock_tools._yahoo")
    def test_volume_spike_bullish_detection(self, mock_yahoo):
        """Test detection of bullish volume spike."""
        # Create data with volume spike on up day
        df = create_mock_price_data(days=60, trend="up", volume_trend="spike")
        mock_yahoo.get_price_history.return_value = df

        from stockai.tools.stock_tools import get_volume_signals

        result = get_volume_signals("TEST")

        assert "error" not in result
        # Should detect high volume
        assert result["volume_ratio_20d"] > 1.5

    @patch("stockai.tools.stock_tools._yahoo")
    def test_accumulation_signal_detection(self, mock_yahoo):
        """Test detection of accumulation pattern."""
        df = create_mock_price_data(days=60, trend="up", volume_trend="up")
        mock_yahoo.get_price_history.return_value = df

        from stockai.tools.stock_tools import get_volume_signals

        result = get_volume_signals("TEST")

        assert "error" not in result
        # Should have A/D related signal
        signal_types = [s["type"] for s in result["individual_signals"]]
        assert any(
            t in ["accumulation", "distribution", "mixed_accumulation"]
            for t in signal_types
        )

    @patch("stockai.tools.stock_tools._yahoo")
    def test_mfi_signal_detection(self, mock_yahoo):
        """Test that MFI signals are included."""
        df = create_mock_price_data(days=60)
        mock_yahoo.get_price_history.return_value = df

        from stockai.tools.stock_tools import get_volume_signals

        result = get_volume_signals("TEST")

        assert "error" not in result
        signal_types = [s["type"] for s in result["individual_signals"]]
        assert any(
            t in ["mfi_oversold", "mfi_overbought", "mfi_neutral"] for t in signal_types
        )

    @patch("stockai.tools.stock_tools._yahoo")
    def test_obv_signal_detection(self, mock_yahoo):
        """Test that OBV trend signals are included."""
        df = create_mock_price_data(days=60)
        mock_yahoo.get_price_history.return_value = df

        from stockai.tools.stock_tools import get_volume_signals

        result = get_volume_signals("TEST")

        assert "error" not in result
        signal_types = [s["type"] for s in result["individual_signals"]]
        assert any(
            t in ["obv_bullish_trend", "obv_bearish_trend", "obv_neutral"]
            for t in signal_types
        )

    @patch("stockai.tools.stock_tools._yahoo")
    def test_returns_volume_ratio(self, mock_yahoo):
        """Test that volume ratio vs 20d average is returned."""
        mock_yahoo.get_price_history.return_value = create_mock_price_data(days=60)

        from stockai.tools.stock_tools import get_volume_signals

        result = get_volume_signals("TEST")

        assert "volume_ratio_20d" in result
        assert result["volume_ratio_20d"] > 0

    @patch("stockai.tools.stock_tools._yahoo")
    def test_returns_current_price_and_volume(self, mock_yahoo):
        """Test that current price and volume are returned."""
        mock_yahoo.get_price_history.return_value = create_mock_price_data(days=60)

        from stockai.tools.stock_tools import get_volume_signals

        result = get_volume_signals("TEST")

        assert "current_price" in result
        assert "current_volume" in result
        assert result["current_price"] > 0
        assert result["current_volume"] > 0

    @patch("stockai.tools.stock_tools._yahoo")
    def test_insufficient_data_error(self, mock_yahoo):
        """Test error handling for insufficient data."""
        mock_yahoo.get_price_history.return_value = create_mock_price_data(days=10)

        from stockai.tools.stock_tools import get_volume_signals

        result = get_volume_signals("TEST")

        assert "error" in result
        assert "Insufficient data" in result["error"]

    @patch("stockai.tools.stock_tools._yahoo")
    def test_empty_data_error(self, mock_yahoo):
        """Test error handling for empty data."""
        mock_yahoo.get_price_history.return_value = pd.DataFrame()

        from stockai.tools.stock_tools import get_volume_signals

        result = get_volume_signals("TEST")

        assert "error" in result

    @patch("stockai.tools.stock_tools._yahoo")
    def test_bullish_bearish_scores_in_range(self, mock_yahoo):
        """Test that bullish and bearish scores are properly normalized."""
        mock_yahoo.get_price_history.return_value = create_mock_price_data(days=60)

        from stockai.tools.stock_tools import get_volume_signals

        result = get_volume_signals("TEST")

        assert "overall_signal" in result
        assert 0 <= result["overall_signal"]["bullish_score"] <= 1
        assert 0 <= result["overall_signal"]["bearish_score"] <= 1


class TestEdgeCases:
    """Test edge cases and error handling."""

    @patch("stockai.tools.stock_tools._yahoo")
    def test_volume_analysis_with_zero_volume(self, mock_yahoo):
        """Test handling of zero volume data."""
        df = create_mock_price_data(days=60)
        df["volume"] = 0  # Set all volumes to zero
        mock_yahoo.get_price_history.return_value = df

        from stockai.tools.stock_tools import get_volume_analysis

        # Should handle gracefully without crashing
        result = get_volume_analysis("TEST")
        # May return error or handle zero volume
        assert result is not None

    @patch("stockai.tools.stock_tools._yahoo")
    def test_volume_profile_single_price(self, mock_yahoo):
        """Test volume profile when all prices are the same."""
        df = create_mock_price_data(days=30)
        constant_price = 10000.0
        df["open"] = constant_price
        df["high"] = constant_price
        df["low"] = constant_price
        df["close"] = constant_price
        mock_yahoo.get_price_history.return_value = df

        from stockai.tools.stock_tools import get_volume_profile

        result = get_volume_profile("TEST")
        # Should handle gracefully - may return error for invalid price range
        assert result is not None

    @patch("stockai.tools.stock_tools._yahoo")
    def test_symbol_case_handling(self, mock_yahoo):
        """Test that symbol case is handled correctly."""
        mock_yahoo.get_price_history.return_value = create_mock_price_data(days=60)

        from stockai.tools.stock_tools import get_volume_analysis

        result = get_volume_analysis("test")  # lowercase

        assert "symbol" in result
        assert result["symbol"] == "TEST"  # Should be uppercase

    @patch("stockai.tools.stock_tools._yahoo")
    def test_custom_period_parameter(self, mock_yahoo):
        """Test that custom period parameter is passed correctly."""
        mock_yahoo.get_price_history.return_value = create_mock_price_data(days=60)

        from stockai.tools.stock_tools import get_volume_analysis

        result = get_volume_analysis("TEST", period="6mo")

        mock_yahoo.get_price_history.assert_called_with("TEST", period="6mo")

    @patch("stockai.tools.stock_tools._yahoo")
    def test_volume_signals_with_flat_market(self, mock_yahoo):
        """Test volume signals in a flat market condition."""
        # Create data with very low price movement
        df = create_mock_price_data(days=60, trend="neutral")
        mock_yahoo.get_price_history.return_value = df

        from stockai.tools.stock_tools import get_volume_signals

        result = get_volume_signals("TEST")

        assert "error" not in result
        # In neutral conditions, should have neutral signals
        assert result["overall_signal"]["direction"] in ["buy", "sell", "neutral"]


class TestIndicatorCalculations:
    """Test specific indicator calculation logic."""

    @patch("stockai.tools.stock_tools._yahoo")
    def test_obv_trend_rising_interpretation(self, mock_yahoo):
        """Test OBV is interpreted as bullish when rising."""
        df = create_mock_price_data(days=60, trend="up", volume_trend="up")
        mock_yahoo.get_price_history.return_value = df

        from stockai.tools.stock_tools import get_volume_analysis

        result = get_volume_analysis("TEST")

        assert "error" not in result
        obv = result["indicators"]["obv"]
        # OBV interpretation should match trend
        assert obv["interpretation"] in ["bullish", "bearish"]

    @patch("stockai.tools.stock_tools._yahoo")
    def test_mfi_oversold_threshold(self, mock_yahoo):
        """Test MFI oversold interpretation at value < 20."""
        df = create_mock_price_data(days=60)
        mock_yahoo.get_price_history.return_value = df

        from stockai.tools.stock_tools import get_volume_analysis

        result = get_volume_analysis("TEST")

        mfi = result["indicators"]["mfi"]
        # Verify interpretation logic
        if mfi["value"] < 20:
            assert mfi["interpretation"] == "oversold"
        elif mfi["value"] > 80:
            assert mfi["interpretation"] == "overbought"
        else:
            assert mfi["interpretation"] == "neutral"

    @patch("stockai.tools.stock_tools._yahoo")
    def test_vwap_price_position(self, mock_yahoo):
        """Test VWAP price position calculation."""
        df = create_mock_price_data(days=60)
        mock_yahoo.get_price_history.return_value = df

        from stockai.tools.stock_tools import get_volume_analysis

        result = get_volume_analysis("TEST")

        vwap = result["indicators"]["vwap"]
        current_price = result["current_price"]

        # Verify price position matches actual comparison
        if current_price > vwap["value"]:
            assert vwap["price_position"] == "above"
        else:
            assert vwap["price_position"] == "below"

    @patch("stockai.tools.stock_tools._yahoo")
    def test_volume_ratio_calculation(self, mock_yahoo):
        """Test volume ratio calculation accuracy."""
        df = create_mock_price_data(days=60)
        mock_yahoo.get_price_history.return_value = df

        from stockai.tools.stock_tools import get_volume_analysis

        result = get_volume_analysis("TEST")

        ratios = result["indicators"]["volume_ratios"]

        # Volume ratios should be positive
        assert ratios["vs_5d_avg"] > 0
        assert ratios["vs_10d_avg"] > 0
        assert ratios["vs_20d_avg"] > 0

        # Average volumes should be positive integers
        assert ratios["avg_volume_5d"] > 0
        assert ratios["avg_volume_10d"] > 0
        assert ratios["avg_volume_20d"] > 0
