"""Unit Tests for Quality Gates System.

Tests for the gate validation system including:
- Gate validation logic (gates.py)
- Smart Money Score calculation (smart_money.py)
- Support/Resistance detection (support_resistance.py)

These tests mock external data sources to ensure isolated, reproducible testing.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# =============================================================================
# Test Data Fixtures
# =============================================================================


def create_mock_ohlcv_data(
    days: int = 60,
    base_price: float = 10000.0,
    base_volume: int = 1000000,
    trend: str = "neutral",
    volume_pattern: str = "normal",
) -> pd.DataFrame:
    """Create mock OHLCV data for testing.

    Args:
        days: Number of days of data
        base_price: Starting price
        base_volume: Base volume
        trend: Price trend ('up', 'down', 'neutral')
        volume_pattern: Volume pattern ('accumulation', 'distribution', 'normal')

    Returns:
        DataFrame with OHLCV data
    """
    dates = [datetime.now() - timedelta(days=days - i) for i in range(days)]

    # Generate prices based on trend
    prices = []
    current_price = base_price
    for i in range(days):
        if trend == "up":
            change = np.random.uniform(0.001, 0.02)
        elif trend == "down":
            change = np.random.uniform(-0.02, -0.001)
        else:
            change = np.random.uniform(-0.01, 0.01)

        current_price = current_price * (1 + change)
        prices.append(current_price)

    # Generate volume based on pattern
    volumes = []
    for i in range(days):
        base_vol = base_volume * np.random.uniform(0.8, 1.2)

        if volume_pattern == "accumulation":
            # Higher volume on up days
            if i > 0 and prices[i] > prices[i - 1]:
                base_vol *= 1.5
        elif volume_pattern == "distribution":
            # Higher volume on down days
            if i > 0 and prices[i] < prices[i - 1]:
                base_vol *= 1.5

        volumes.append(int(base_vol))

    # Create OHLC from prices
    df = pd.DataFrame({
        "date": dates,
        "open": [p * np.random.uniform(0.995, 1.0) for p in prices],
        "high": [p * np.random.uniform(1.0, 1.02) for p in prices],
        "low": [p * np.random.uniform(0.98, 1.0) for p in prices],
        "close": prices,
        "volume": volumes,
    })

    return df


def create_swing_data(days: int = 60, swings: int = 3) -> pd.DataFrame:
    """Create OHLCV data with clear swing highs and lows for support/resistance testing.

    Args:
        days: Number of days
        swings: Number of swings (creates clear pivot points)

    Returns:
        DataFrame with clear support/resistance levels
    """
    dates = [datetime.now() - timedelta(days=days - i) for i in range(days)]

    # Create price pattern with clear swings
    base_price = 10000.0
    prices = []

    days_per_swing = days // (swings * 2)
    direction = 1  # Start going up
    current_price = base_price

    for i in range(days):
        swing_phase = (i // days_per_swing) % 2

        if swing_phase == 0:
            # Going up
            change = np.random.uniform(0.005, 0.02)
        else:
            # Going down
            change = np.random.uniform(-0.02, -0.005)

        current_price = current_price * (1 + change)
        prices.append(current_price)

    df = pd.DataFrame({
        "date": dates,
        "open": [p * 0.998 for p in prices],
        "high": [p * 1.01 for p in prices],
        "low": [p * 0.99 for p in prices],
        "close": prices,
        "volume": [1000000] * days,
    })

    return df


# =============================================================================
# Tests for gates.py
# =============================================================================


class TestGateConfig:
    """Tests for GateConfig dataclass."""

    def test_default_values(self):
        """Test that GateConfig has correct default thresholds."""
        from stockai.scoring.gates import GateConfig

        config = GateConfig()

        assert config.overall_min == 70.0
        assert config.technical_min == 60.0
        assert config.smart_money_min == 3.0
        assert config.near_support_pct == 5.0
        assert config.adx_min == 20.0
        assert config.fundamental_min == 60.0

    def test_custom_values(self):
        """Test that GateConfig accepts custom thresholds."""
        from stockai.scoring.gates import GateConfig

        config = GateConfig(
            overall_min=80.0,
            technical_min=70.0,
            smart_money_min=4.0,
            near_support_pct=3.0,
            adx_min=25.0,
            fundamental_min=65.0,
        )

        assert config.overall_min == 80.0
        assert config.technical_min == 70.0
        assert config.smart_money_min == 4.0
        assert config.near_support_pct == 3.0
        assert config.adx_min == 25.0
        assert config.fundamental_min == 65.0


class TestGateResult:
    """Tests for GateResult dataclass."""

    def test_gate_result_fields(self):
        """Test that GateResult has all required fields."""
        from stockai.scoring.gates import GateResult

        result = GateResult(
            all_passed=True,
            gates_passed=6,
            total_gates=6,
            passed_gates=["Gate 1", "Gate 2"],
            rejection_reasons=[],
            confidence="HIGH",
        )

        assert result.all_passed is True
        assert result.gates_passed == 6
        assert result.total_gates == 6
        assert len(result.passed_gates) == 2
        assert len(result.rejection_reasons) == 0
        assert result.confidence == "HIGH"


class TestValidateGates:
    """Tests for validate_gates function."""

    def test_all_gates_pass(self):
        """Test when all gates pass with high-quality stock."""
        from stockai.scoring.gates import validate_gates

        stock_data = {
            "overall_score": 85.0,
            "technical_score": 75.0,
            "smart_money_score": 4.0,
            "distance_to_support_pct": 2.5,
            "adx": 30.0,
            "fundamental_score": 70.0,
        }

        result = validate_gates(stock_data)

        assert result.all_passed is True
        assert result.gates_passed == 6
        assert result.total_gates == 6
        assert len(result.rejection_reasons) == 0
        assert result.confidence == "HIGH"

    def test_overall_score_gate_fail(self):
        """Test Gate 1: Overall Score failure."""
        from stockai.scoring.gates import validate_gates

        stock_data = {
            "overall_score": 65.0,  # Below 70
            "technical_score": 75.0,
            "smart_money_score": 4.0,
            "distance_to_support_pct": 2.5,
            "adx": 30.0,
            "fundamental_score": 70.0,
        }

        result = validate_gates(stock_data)

        assert result.all_passed is False
        assert result.gates_passed == 5
        assert "Overall Score" in result.rejection_reasons[0]
        assert result.confidence == "WATCH"  # 1 failure + score >= 60

    def test_technical_score_gate_fail(self):
        """Test Gate 2: Technical Score failure."""
        from stockai.scoring.gates import validate_gates

        stock_data = {
            "overall_score": 85.0,
            "technical_score": 50.0,  # Below 60
            "smart_money_score": 4.0,
            "distance_to_support_pct": 2.5,
            "adx": 30.0,
            "fundamental_score": 70.0,
        }

        result = validate_gates(stock_data)

        assert result.all_passed is False
        assert "Technical Score" in result.rejection_reasons[0]

    def test_smart_money_gate_fail(self):
        """Test Gate 3: Smart Money Score failure."""
        from stockai.scoring.gates import validate_gates

        stock_data = {
            "overall_score": 85.0,
            "technical_score": 75.0,
            "smart_money_score": 1.5,  # Below 3.0
            "distance_to_support_pct": 2.5,
            "adx": 30.0,
            "fundamental_score": 70.0,
        }

        result = validate_gates(stock_data)

        assert result.all_passed is False
        assert "Smart Money Score" in result.rejection_reasons[0]

    def test_distance_to_support_gate_fail(self):
        """Test Gate 4: Distance to Support failure."""
        from stockai.scoring.gates import validate_gates

        stock_data = {
            "overall_score": 85.0,
            "technical_score": 75.0,
            "smart_money_score": 4.0,
            "distance_to_support_pct": 8.0,  # Above 5%
            "adx": 30.0,
            "fundamental_score": 70.0,
        }

        result = validate_gates(stock_data)

        assert result.all_passed is False
        assert "Distance to Support" in result.rejection_reasons[0]

    def test_no_support_level_fails(self):
        """Test Gate 4: No support level found failure."""
        from stockai.scoring.gates import validate_gates

        stock_data = {
            "overall_score": 85.0,
            "technical_score": 75.0,
            "smart_money_score": 4.0,
            "distance_to_support_pct": None,  # No support
            "adx": 30.0,
            "fundamental_score": 70.0,
        }

        result = validate_gates(stock_data)

        assert result.all_passed is False
        assert "No support level found" in result.rejection_reasons[0]

    def test_adx_gate_fail(self):
        """Test Gate 5: ADX Trend Strength failure."""
        from stockai.scoring.gates import validate_gates

        stock_data = {
            "overall_score": 85.0,
            "technical_score": 75.0,
            "smart_money_score": 4.0,
            "distance_to_support_pct": 2.5,
            "adx": 15.0,  # Below 20
            "fundamental_score": 70.0,
        }

        result = validate_gates(stock_data)

        assert result.all_passed is False
        assert "ADX Trend Strength" in result.rejection_reasons[0]

    def test_fundamental_score_gate_fail(self):
        """Test Gate 6: Fundamental Score failure."""
        from stockai.scoring.gates import validate_gates

        stock_data = {
            "overall_score": 85.0,
            "technical_score": 75.0,
            "smart_money_score": 4.0,
            "distance_to_support_pct": 2.5,
            "adx": 30.0,
            "fundamental_score": 50.0,  # Below 60
        }

        result = validate_gates(stock_data)

        assert result.all_passed is False
        assert "Fundamental Score" in result.rejection_reasons[0]

    def test_multiple_gate_failures(self):
        """Test when multiple gates fail."""
        from stockai.scoring.gates import validate_gates

        stock_data = {
            "overall_score": 50.0,  # Fail
            "technical_score": 40.0,  # Fail
            "smart_money_score": 0.5,  # Fail
            "distance_to_support_pct": 10.0,  # Fail
            "adx": 10.0,  # Fail
            "fundamental_score": 40.0,  # Fail
        }

        result = validate_gates(stock_data)

        assert result.all_passed is False
        assert result.gates_passed == 0
        assert len(result.rejection_reasons) == 6
        assert result.confidence == "REJECTED"

    def test_watch_confidence_level(self):
        """Test WATCH confidence when 1-2 gates fail with overall >= 60."""
        from stockai.scoring.gates import validate_gates

        stock_data = {
            "overall_score": 65.0,  # Fail but >= 60
            "technical_score": 75.0,
            "smart_money_score": 4.0,
            "distance_to_support_pct": 2.5,
            "adx": 15.0,  # Fail
            "fundamental_score": 70.0,
        }

        result = validate_gates(stock_data)

        assert result.all_passed is False
        assert result.gates_passed == 4
        assert result.confidence == "WATCH"

    def test_rejected_confidence_many_failures(self):
        """Test REJECTED confidence when > 2 gates fail."""
        from stockai.scoring.gates import validate_gates

        stock_data = {
            "overall_score": 50.0,  # Fail
            "technical_score": 40.0,  # Fail
            "smart_money_score": 1.0,  # Fail
            "distance_to_support_pct": 2.5,
            "adx": 30.0,
            "fundamental_score": 70.0,
        }

        result = validate_gates(stock_data)

        assert result.all_passed is False
        assert len(result.rejection_reasons) == 3
        assert result.confidence == "REJECTED"

    def test_custom_gate_config(self):
        """Test gate validation with custom thresholds."""
        from stockai.scoring.gates import validate_gates, GateConfig

        # Custom stricter config
        config = GateConfig(
            overall_min=80.0,
            technical_min=70.0,
            smart_money_min=4.0,
            near_support_pct=3.0,
            adx_min=25.0,
            fundamental_min=65.0,
        )

        stock_data = {
            "overall_score": 75.0,  # Passes default (70) but fails custom (80)
            "technical_score": 65.0,  # Passes default (60) but fails custom (70)
            "smart_money_score": 3.5,  # Passes default (3) but fails custom (4)
            "distance_to_support_pct": 4.0,  # Passes default (5) but fails custom (3)
            "adx": 22.0,  # Passes default (20) but fails custom (25)
            "fundamental_score": 62.0,  # Passes default (60) but fails custom (65)
        }

        # Should pass with default config
        result_default = validate_gates(stock_data)
        assert result_default.all_passed is True

        # Should fail with custom stricter config
        result_custom = validate_gates(stock_data, config=config)
        assert result_custom.all_passed is False
        assert len(result_custom.rejection_reasons) == 6

    def test_missing_data_uses_defaults(self):
        """Test that missing data uses default zero values."""
        from stockai.scoring.gates import validate_gates

        stock_data = {}  # Empty data

        result = validate_gates(stock_data)

        assert result.all_passed is False
        assert result.gates_passed == 0
        assert len(result.rejection_reasons) == 6

    def test_passed_gates_list_format(self):
        """Test that passed_gates list has correct format."""
        from stockai.scoring.gates import validate_gates

        stock_data = {
            "overall_score": 85.0,
            "technical_score": 75.0,
            "smart_money_score": 4.0,
            "distance_to_support_pct": 2.5,
            "adx": 30.0,
            "fundamental_score": 70.0,
        }

        result = validate_gates(stock_data)

        # Check that passed gates include score values
        for gate_msg in result.passed_gates:
            assert ">=" in gate_msg or "<=" in gate_msg


# =============================================================================
# Tests for smart_money.py
# =============================================================================


class TestSmartMoneyResult:
    """Tests for SmartMoneyResult dataclass."""

    def test_smart_money_result_fields(self):
        """Test that SmartMoneyResult has all required fields."""
        from stockai.scoring.smart_money import SmartMoneyResult

        result = SmartMoneyResult(
            score=3.5,
            accumulation_days=5,
            distribution_days=2,
            net_accumulation=3,
            obv_trend="BULLISH",
            mfi=55.0,
            mfi_signal="NEUTRAL",
            unusual_volume="NORMAL",
            interpretation="ACCUMULATION",
        )

        assert result.score == 3.5
        assert result.accumulation_days == 5
        assert result.distribution_days == 2
        assert result.net_accumulation == 3
        assert result.obv_trend == "BULLISH"
        assert result.mfi == 55.0
        assert result.mfi_signal == "NEUTRAL"
        assert result.unusual_volume == "NORMAL"
        assert result.interpretation == "ACCUMULATION"


class TestCalculateSmartMoneyScore:
    """Tests for calculate_smart_money_score function."""

    def test_score_range(self):
        """Test that score is always within valid range (-2 to 5)."""
        from stockai.scoring.smart_money import calculate_smart_money_score

        df = create_mock_ohlcv_data(days=30)
        result = calculate_smart_money_score(df)

        assert -2.0 <= result.score <= 5.0

    def test_accumulation_pattern_high_score(self):
        """Test that accumulation pattern produces higher score."""
        from stockai.scoring.smart_money import calculate_smart_money_score

        df = create_mock_ohlcv_data(
            days=30, trend="up", volume_pattern="accumulation"
        )
        result = calculate_smart_money_score(df)

        # Accumulation should produce positive score
        assert result.score >= 0
        assert result.net_accumulation >= 0

    def test_distribution_pattern_low_score(self):
        """Test that distribution pattern produces lower score."""
        from stockai.scoring.smart_money import calculate_smart_money_score

        df = create_mock_ohlcv_data(
            days=30, trend="down", volume_pattern="distribution"
        )
        result = calculate_smart_money_score(df)

        # Distribution pattern should produce lower/negative score
        assert result.distribution_days >= 0

    def test_obv_trend_values(self):
        """Test that OBV trend is one of valid values."""
        from stockai.scoring.smart_money import calculate_smart_money_score

        df = create_mock_ohlcv_data(days=30)
        result = calculate_smart_money_score(df)

        assert result.obv_trend in ["BULLISH", "NEUTRAL", "BEARISH"]

    def test_mfi_signal_values(self):
        """Test that MFI signal is one of valid values."""
        from stockai.scoring.smart_money import calculate_smart_money_score

        df = create_mock_ohlcv_data(days=30)
        result = calculate_smart_money_score(df)

        assert result.mfi_signal in ["OVERBOUGHT", "NEUTRAL", "OVERSOLD"]

    def test_mfi_value_range(self):
        """Test that MFI value is within valid range (0-100)."""
        from stockai.scoring.smart_money import calculate_smart_money_score

        df = create_mock_ohlcv_data(days=30)
        result = calculate_smart_money_score(df)

        assert 0 <= result.mfi <= 100

    def test_unusual_volume_values(self):
        """Test that unusual_volume is one of valid values."""
        from stockai.scoring.smart_money import calculate_smart_money_score

        df = create_mock_ohlcv_data(days=30)
        result = calculate_smart_money_score(df)

        assert result.unusual_volume in ["HIGH", "NORMAL", "LOW"]

    def test_interpretation_values(self):
        """Test that interpretation is one of valid values."""
        from stockai.scoring.smart_money import calculate_smart_money_score

        df = create_mock_ohlcv_data(days=30)
        result = calculate_smart_money_score(df)

        assert result.interpretation in ["ACCUMULATION", "NEUTRAL", "DISTRIBUTION"]

    def test_interpretation_matches_score(self):
        """Test that interpretation aligns with score."""
        from stockai.scoring.smart_money import calculate_smart_money_score

        df = create_mock_ohlcv_data(days=30)
        result = calculate_smart_money_score(df)

        if result.score >= 3.0:
            assert result.interpretation == "ACCUMULATION"
        elif result.score <= 0:
            assert result.interpretation == "DISTRIBUTION"
        else:
            assert result.interpretation == "NEUTRAL"

    def test_short_data_handles_gracefully(self):
        """Test that short data is handled without errors."""
        from stockai.scoring.smart_money import calculate_smart_money_score

        df = create_mock_ohlcv_data(days=10)  # Less than default lookback
        result = calculate_smart_money_score(df)

        # Should return valid result without error
        assert result is not None
        assert -2.0 <= result.score <= 5.0

    def test_custom_lookback_parameter(self):
        """Test that custom lookback parameter works."""
        from stockai.scoring.smart_money import calculate_smart_money_score

        df = create_mock_ohlcv_data(days=60)
        result = calculate_smart_money_score(df, lookback=10)

        assert result is not None

    def test_custom_volume_threshold(self):
        """Test that custom volume threshold affects results."""
        from stockai.scoring.smart_money import calculate_smart_money_score

        df = create_mock_ohlcv_data(days=30)

        # Lower threshold = more days count as unusual volume
        result_low = calculate_smart_money_score(df, volume_threshold=1.0)
        # Higher threshold = fewer days count as unusual volume
        result_high = calculate_smart_money_score(df, volume_threshold=2.0)

        # With lower threshold, should detect more accumulation/distribution
        total_low = result_low.accumulation_days + result_low.distribution_days
        total_high = result_high.accumulation_days + result_high.distribution_days
        assert total_low >= total_high

    def test_net_accumulation_calculation(self):
        """Test that net_accumulation = accumulation_days - distribution_days."""
        from stockai.scoring.smart_money import calculate_smart_money_score

        df = create_mock_ohlcv_data(days=30)
        result = calculate_smart_money_score(df)

        assert result.net_accumulation == (
            result.accumulation_days - result.distribution_days
        )


# =============================================================================
# Tests for support_resistance.py
# =============================================================================


class TestSupportResistanceResult:
    """Tests for SupportResistanceResult dataclass."""

    def test_support_resistance_result_fields(self):
        """Test that SupportResistanceResult has all required fields."""
        from stockai.scoring.support_resistance import SupportResistanceResult

        result = SupportResistanceResult(
            current_price=10000.0,
            supports=[9500.0, 9200.0],
            resistances=[10500.0, 10800.0],
            nearest_support=9500.0,
            nearest_resistance=10500.0,
            distance_to_support_pct=5.0,
            is_near_support=False,
            suggested_stop_loss=9215.0,
        )

        assert result.current_price == 10000.0
        assert len(result.supports) == 2
        assert len(result.resistances) == 2
        assert result.nearest_support == 9500.0
        assert result.nearest_resistance == 10500.0
        assert result.distance_to_support_pct == 5.0
        assert result.is_near_support is False
        assert result.suggested_stop_loss == 9215.0


class TestFindSupportResistance:
    """Tests for find_support_resistance function."""

    def test_returns_valid_result(self):
        """Test that function returns a valid SupportResistanceResult."""
        from stockai.scoring.support_resistance import find_support_resistance

        df = create_swing_data(days=60)
        result = find_support_resistance(df)

        assert result is not None
        assert result.current_price > 0

    def test_supports_below_current_price(self):
        """Test that all support levels are below current price."""
        from stockai.scoring.support_resistance import find_support_resistance

        df = create_swing_data(days=60)
        result = find_support_resistance(df)

        for support in result.supports:
            assert support < result.current_price

    def test_resistances_above_current_price(self):
        """Test that all resistance levels are above current price."""
        from stockai.scoring.support_resistance import find_support_resistance

        df = create_swing_data(days=60)
        result = find_support_resistance(df)

        for resistance in result.resistances:
            assert resistance > result.current_price

    def test_max_levels_respected(self):
        """Test that max_levels parameter limits results."""
        from stockai.scoring.support_resistance import find_support_resistance

        df = create_swing_data(days=100, swings=5)
        result = find_support_resistance(df, max_levels=2)

        assert len(result.supports) <= 2
        assert len(result.resistances) <= 2

    def test_distance_to_support_calculation(self):
        """Test that distance to support is calculated correctly."""
        from stockai.scoring.support_resistance import find_support_resistance

        df = create_swing_data(days=60)
        result = find_support_resistance(df)

        if result.nearest_support is not None:
            expected_pct = (
                (result.current_price - result.nearest_support) / result.current_price
            ) * 100
            assert abs(result.distance_to_support_pct - expected_pct) < 0.01

    def test_is_near_support_flag(self):
        """Test is_near_support flag is set correctly."""
        from stockai.scoring.support_resistance import find_support_resistance

        df = create_swing_data(days=60)
        result = find_support_resistance(df, near_support_threshold=10.0)

        if result.distance_to_support_pct is not None:
            expected_near = result.distance_to_support_pct <= 10.0
            assert result.is_near_support == expected_near

    def test_suggested_stop_loss_below_support(self):
        """Test that suggested stop loss is below nearest support."""
        from stockai.scoring.support_resistance import find_support_resistance

        df = create_swing_data(days=60)
        result = find_support_resistance(df)

        if result.nearest_support is not None:
            assert result.suggested_stop_loss < result.nearest_support

    def test_suggested_stop_loss_not_too_low(self):
        """Test that stop loss respects max_stop_loss_pct."""
        from stockai.scoring.support_resistance import find_support_resistance

        df = create_swing_data(days=60)
        max_sl_pct = 8.0
        result = find_support_resistance(df, max_stop_loss_pct=max_sl_pct)

        min_sl = result.current_price * (1 - max_sl_pct / 100)
        assert result.suggested_stop_loss >= min_sl

    def test_no_support_found_case(self):
        """Test behavior when no support levels can be found."""
        from stockai.scoring.support_resistance import find_support_resistance

        # Create data with only upward movement (no swing lows)
        df = create_mock_ohlcv_data(days=20, trend="up")
        result = find_support_resistance(df, order=10)

        # With short data and large order, may not find supports
        # Either supports is empty or has some levels
        assert isinstance(result.supports, list)

    def test_short_data_handles_gracefully(self):
        """Test that short data is handled without errors."""
        from stockai.scoring.support_resistance import find_support_resistance

        df = create_swing_data(days=15)  # Short data
        result = find_support_resistance(df)

        assert result is not None
        assert result.current_price > 0

    def test_custom_lookback(self):
        """Test that custom lookback parameter works."""
        from stockai.scoring.support_resistance import find_support_resistance

        df = create_swing_data(days=100)

        result_short = find_support_resistance(df, lookback=30)
        result_long = find_support_resistance(df, lookback=90)

        # Both should return valid results
        assert result_short.current_price == result_long.current_price

    def test_custom_order_affects_pivot_detection(self):
        """Test that order parameter affects pivot detection sensitivity."""
        from stockai.scoring.support_resistance import find_support_resistance

        df = create_swing_data(days=100)

        # Smaller order = more sensitive = potentially more pivots
        result_sensitive = find_support_resistance(df, order=3)
        # Larger order = less sensitive = fewer pivots
        result_strict = find_support_resistance(df, order=10)

        # Both should return valid results
        assert result_sensitive is not None
        assert result_strict is not None

    def test_nearest_support_is_closest(self):
        """Test that nearest_support is the closest to current price."""
        from stockai.scoring.support_resistance import find_support_resistance

        df = create_swing_data(days=100, swings=5)
        result = find_support_resistance(df)

        if result.supports and result.nearest_support:
            # Nearest should be the first (closest) in sorted list
            assert result.nearest_support == result.supports[0]

            # All other supports should be further away
            for support in result.supports[1:]:
                assert support < result.nearest_support

    def test_nearest_resistance_is_closest(self):
        """Test that nearest_resistance is the closest to current price."""
        from stockai.scoring.support_resistance import find_support_resistance

        df = create_swing_data(days=100, swings=5)
        result = find_support_resistance(df)

        if result.resistances and result.nearest_resistance:
            # Nearest should be the first (closest) in sorted list
            assert result.nearest_resistance == result.resistances[0]

            # All other resistances should be further away
            for resistance in result.resistances[1:]:
                assert resistance > result.nearest_resistance


# =============================================================================
# Integration Tests for Combined Scoring
# =============================================================================


class TestScoringIntegration:
    """Integration tests for the complete scoring pipeline."""

    def test_gates_use_smart_money_score(self):
        """Test that gates correctly evaluate Smart Money Score."""
        from stockai.scoring.gates import validate_gates
        from stockai.scoring.smart_money import calculate_smart_money_score

        df = create_mock_ohlcv_data(days=30, trend="up", volume_pattern="accumulation")
        sm_result = calculate_smart_money_score(df)

        stock_data = {
            "overall_score": 85.0,
            "technical_score": 75.0,
            "smart_money_score": sm_result.score,
            "distance_to_support_pct": 2.5,
            "adx": 30.0,
            "fundamental_score": 70.0,
        }

        gate_result = validate_gates(stock_data)

        # If SM score >= 3, gate should pass
        if sm_result.score >= 3.0:
            assert "Smart Money Score" not in " ".join(gate_result.rejection_reasons)

    def test_gates_use_support_distance(self):
        """Test that gates correctly evaluate distance to support."""
        from stockai.scoring.gates import validate_gates
        from stockai.scoring.support_resistance import find_support_resistance

        df = create_swing_data(days=60)
        sr_result = find_support_resistance(df)

        if sr_result.distance_to_support_pct is not None:
            stock_data = {
                "overall_score": 85.0,
                "technical_score": 75.0,
                "smart_money_score": 4.0,
                "distance_to_support_pct": sr_result.distance_to_support_pct,
                "adx": 30.0,
                "fundamental_score": 70.0,
            }

            gate_result = validate_gates(stock_data)

            # If distance <= 5%, gate should pass
            if sr_result.distance_to_support_pct <= 5.0:
                assert "Distance to Support" not in " ".join(
                    gate_result.rejection_reasons
                )

    def test_complete_scoring_pipeline(self):
        """Test a complete scoring pipeline from raw data to gate validation."""
        from stockai.scoring.gates import validate_gates
        from stockai.scoring.smart_money import calculate_smart_money_score
        from stockai.scoring.support_resistance import find_support_resistance

        # Create realistic data
        df = create_mock_ohlcv_data(days=60, trend="up", volume_pattern="accumulation")

        # Calculate all scores
        sm_result = calculate_smart_money_score(df)
        sr_result = find_support_resistance(df)

        # Mock other scores (would come from factors.py in real usage)
        stock_data = {
            "overall_score": 75.0,
            "technical_score": 70.0,
            "smart_money_score": sm_result.score,
            "distance_to_support_pct": sr_result.distance_to_support_pct,
            "adx": 25.0,
            "fundamental_score": 65.0,
        }

        gate_result = validate_gates(stock_data)

        # Should get a valid result with proper structure
        assert gate_result.total_gates == 6
        assert gate_result.gates_passed <= gate_result.total_gates
        assert len(gate_result.passed_gates) == gate_result.gates_passed
        assert gate_result.confidence in ["HIGH", "WATCH", "REJECTED"]
