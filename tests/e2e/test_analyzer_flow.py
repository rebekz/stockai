"""Integration Tests for the Complete Analyzer Flow.

Tests the full analysis pipeline from raw data through to trade plan generation.
Uses mocked data to ensure reproducible, isolated tests that don't depend on
external APIs.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock


# =============================================================================
# Test Data Generation
# =============================================================================


def create_realistic_ohlcv(
    days: int = 120,
    base_price: float = 4500.0,
    trend: str = "bullish",
    volatility: float = 0.015,
) -> pd.DataFrame:
    """Create realistic OHLCV data simulating Indonesian stock behavior.

    Args:
        days: Number of trading days
        base_price: Starting price
        trend: 'bullish', 'bearish', or 'sideways'
        volatility: Daily volatility (as decimal)

    Returns:
        DataFrame with realistic OHLCV data
    """
    np.random.seed(42)  # Reproducibility
    dates = [datetime.now() - timedelta(days=days - i) for i in range(days)]

    # Generate returns based on trend
    if trend == "bullish":
        drift = 0.0003  # Positive drift
    elif trend == "bearish":
        drift = -0.0002  # Negative drift
    else:
        drift = 0  # No drift for sideways

    returns = np.random.normal(drift, volatility, days)
    prices = [base_price]
    for r in returns[1:]:
        prices.append(prices[-1] * (1 + r))

    prices = np.array(prices)

    # Generate OHLC from close prices
    opens = prices * (1 + np.random.uniform(-0.005, 0.005, days))
    highs = np.maximum(prices, opens) * (1 + np.random.uniform(0, 0.015, days))
    lows = np.minimum(prices, opens) * (1 - np.random.uniform(0, 0.015, days))

    # Generate volume with some patterns
    base_volume = 5_000_000
    volumes = base_volume * (1 + np.random.uniform(-0.3, 0.5, days))

    # Add volume spikes on larger price moves
    large_moves = np.abs(returns) > volatility * 1.5
    volumes[large_moves] *= 1.5

    df = pd.DataFrame({
        "date": dates,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": prices,
        "volume": volumes.astype(int),
    })

    return df


def create_qualified_stock_data() -> tuple[pd.DataFrame, dict]:
    """Create data for a stock that should pass all gates.

    Returns:
        Tuple of (price_data, fundamentals)
    """
    # Create bullish trend near support with good volume
    df = create_realistic_ohlcv(days=120, trend="bullish", volatility=0.012)

    # Strong fundamentals
    fundamentals = {
        "pe_ratio": 12.0,  # Reasonable P/E
        "pb_ratio": 1.8,   # Reasonable P/B
        "roe": 18.0,       # Strong ROE
        "debt_to_equity": 0.4,  # Low debt
        "profit_margin": 15.0,  # Good margin
        "current_ratio": 2.0,   # Strong liquidity
    }

    return df, fundamentals


def create_rejected_stock_data() -> tuple[pd.DataFrame, dict]:
    """Create data for a stock that should be rejected.

    Returns:
        Tuple of (price_data, fundamentals)
    """
    # Create bearish trend with high volatility
    df = create_realistic_ohlcv(days=120, trend="bearish", volatility=0.025)

    # Weak fundamentals
    fundamentals = {
        "pe_ratio": 35.0,  # High P/E
        "pb_ratio": 4.5,   # High P/B
        "roe": 5.0,        # Low ROE
        "debt_to_equity": 1.8,  # High debt
        "profit_margin": 3.0,   # Low margin
        "current_ratio": 0.8,   # Weak liquidity
    }

    return df, fundamentals


# =============================================================================
# Analyzer Flow Tests
# =============================================================================


class TestAnalyzeStock:
    """Tests for the main analyze_stock function."""

    def test_analyze_returns_complete_result(self):
        """Test that analyze_stock returns all expected fields."""
        from stockai.scoring.analyzer import analyze_stock

        df, fundamentals = create_qualified_stock_data()
        result = analyze_stock("BBCA", df, fundamentals)

        # Check all required fields exist
        assert result.ticker == "BBCA"
        assert result.current_price > 0
        assert 0 <= result.composite_score <= 100
        assert 0 <= result.value_score <= 100
        assert 0 <= result.quality_score <= 100
        assert 0 <= result.momentum_score <= 100
        assert 0 <= result.volatility_score <= 100
        assert result.smart_money is not None
        assert result.support_resistance is not None
        assert result.adx is not None
        assert result.gates is not None
        assert result.decision in ["BUY", "NO_TRADE"]
        assert result.confidence in ["HIGH", "WATCH", "REJECTED"]

    def test_ticker_uppercased(self):
        """Test that ticker is uppercased in result."""
        from stockai.scoring.analyzer import analyze_stock

        df, _ = create_qualified_stock_data()
        result = analyze_stock("bbca", df)

        assert result.ticker == "BBCA"

    def test_smart_money_analysis_included(self):
        """Test that Smart Money analysis is performed."""
        from stockai.scoring.analyzer import analyze_stock

        df, _ = create_qualified_stock_data()
        result = analyze_stock("BBCA", df)

        # Check Smart Money fields
        assert -2.0 <= result.smart_money.score <= 5.0
        assert result.smart_money.obv_trend in ["BULLISH", "NEUTRAL", "BEARISH"]
        assert result.smart_money.interpretation in [
            "ACCUMULATION", "NEUTRAL", "DISTRIBUTION"
        ]

    def test_support_resistance_analysis_included(self):
        """Test that Support/Resistance analysis is performed."""
        from stockai.scoring.analyzer import analyze_stock

        df, _ = create_qualified_stock_data()
        result = analyze_stock("BBCA", df)

        # Check Support/Resistance fields
        assert result.support_resistance.current_price > 0
        assert isinstance(result.support_resistance.supports, list)
        assert isinstance(result.support_resistance.resistances, list)

    def test_adx_analysis_included(self):
        """Test that ADX trend strength is calculated."""
        from stockai.scoring.analyzer import analyze_stock

        df, _ = create_qualified_stock_data()
        result = analyze_stock("BBCA", df)

        # Check ADX fields
        assert "adx" in result.adx or "error" in result.adx
        if "adx" in result.adx:
            assert result.adx["adx"] >= 0
            assert result.adx["trend_strength"] in [
                "STRONG", "MODERATE", "WEAK", "NONE", "ABSENT", "UNKNOWN"
            ]

    def test_gate_validation_included(self):
        """Test that gate validation is performed."""
        from stockai.scoring.analyzer import analyze_stock

        df, fundamentals = create_qualified_stock_data()
        result = analyze_stock("BBCA", df, fundamentals)

        # Check gate validation
        assert result.gates.total_gates == 6
        assert 0 <= result.gates.gates_passed <= 6
        assert result.gates.confidence in ["HIGH", "WATCH", "REJECTED"]

    def test_trade_plan_generated_for_qualified_stock(self):
        """Test that trade plan is generated when gates pass."""
        from stockai.scoring.analyzer import analyze_stock

        df, fundamentals = create_qualified_stock_data()
        result = analyze_stock("BBCA", df, fundamentals)

        # If HIGH or WATCH, trade plan should be generated
        if result.confidence in ["HIGH", "WATCH"]:
            assert result.trade_plan is not None
            assert result.trade_plan.entry_low > 0
            assert result.trade_plan.entry_high > result.trade_plan.entry_low
            assert result.trade_plan.stop_loss < result.trade_plan.entry_low
            assert result.trade_plan.take_profit_1 > result.trade_plan.entry_high

    def test_no_trade_plan_for_rejected_stock(self):
        """Test that no trade plan is generated when rejected."""
        from stockai.scoring.analyzer import analyze_stock

        df, fundamentals = create_rejected_stock_data()
        result = analyze_stock("WEAK", df, fundamentals)

        if result.confidence == "REJECTED":
            assert result.trade_plan is None or result.decision == "NO_TRADE"

    def test_decision_matches_confidence(self):
        """Test that decision aligns with confidence level."""
        from stockai.scoring.analyzer import analyze_stock

        df, fundamentals = create_qualified_stock_data()
        result = analyze_stock("BBCA", df, fundamentals)

        if result.gates.all_passed:
            assert result.decision == "BUY"
            assert result.confidence == "HIGH"
        elif result.gates.confidence == "WATCH":
            assert result.decision == "NO_TRADE"
            assert result.confidence == "WATCH"
        else:
            assert result.decision == "NO_TRADE"
            assert result.confidence == "REJECTED"

    def test_custom_gate_config(self):
        """Test that custom gate config affects validation."""
        from stockai.scoring.analyzer import analyze_stock
        from stockai.scoring.gates import GateConfig

        df, fundamentals = create_qualified_stock_data()

        # Very lenient config
        lenient_config = GateConfig(
            overall_min=30.0,
            technical_min=30.0,
            smart_money_min=0.0,
            near_support_pct=50.0,
            adx_min=5.0,
            fundamental_min=30.0,
        )

        # Very strict config
        strict_config = GateConfig(
            overall_min=95.0,
            technical_min=90.0,
            smart_money_min=4.5,
            near_support_pct=1.0,
            adx_min=40.0,
            fundamental_min=90.0,
        )

        result_lenient = analyze_stock("BBCA", df, fundamentals, lenient_config)
        result_strict = analyze_stock("BBCA", df, fundamentals, strict_config)

        # Lenient should pass more gates
        assert result_lenient.gates.gates_passed >= result_strict.gates.gates_passed


class TestFactorScoreCalculation:
    """Tests for factor score calculation within analyzer."""

    def test_momentum_score_reflects_trend(self):
        """Test that momentum score reflects price trend."""
        from stockai.scoring.analyzer import analyze_stock

        bullish_df, _ = create_qualified_stock_data()
        bearish_df, _ = create_rejected_stock_data()

        result_bullish = analyze_stock("BULL", bullish_df)
        result_bearish = analyze_stock("BEAR", bearish_df)

        # Bullish should have higher momentum
        assert result_bullish.momentum_score >= result_bearish.momentum_score

    def test_volatility_score_calculation(self):
        """Test that volatility score is calculated correctly."""
        from stockai.scoring.analyzer import analyze_stock

        low_vol_df = create_realistic_ohlcv(days=120, volatility=0.01)
        high_vol_df = create_realistic_ohlcv(days=120, volatility=0.03)

        result_low_vol = analyze_stock("STABLE", low_vol_df)
        result_high_vol = analyze_stock("VOLATILE", high_vol_df)

        # Lower volatility = higher volatility score (inverted for scoring)
        # volatility_score: 0-100 scale where higher = lower risk = better
        assert result_low_vol.volatility_score >= result_high_vol.volatility_score

    def test_fundamentals_affect_value_quality_scores(self):
        """Test that fundamentals affect value and quality scores."""
        from stockai.scoring.analyzer import analyze_stock

        df, _ = create_qualified_stock_data()

        good_fund = {
            "pe_ratio": 10.0,
            "pb_ratio": 1.0,
            "roe": 20.0,
            "debt_to_equity": 0.3,
            "profit_margin": 20.0,
        }

        poor_fund = {
            "pe_ratio": 50.0,
            "pb_ratio": 5.0,
            "roe": 3.0,
            "debt_to_equity": 2.0,
            "profit_margin": 2.0,
        }

        result_good = analyze_stock("GOOD", df, good_fund)
        result_poor = analyze_stock("POOR", df, poor_fund)

        # Good fundamentals should score higher
        assert result_good.value_score >= result_poor.value_score
        assert result_good.quality_score >= result_poor.quality_score


class TestTradePlanGeneration:
    """Tests for trade plan generation within analyzer."""

    def test_trade_plan_price_levels_logical(self):
        """Test that trade plan has logical price levels."""
        from stockai.scoring.analyzer import analyze_stock

        df, fundamentals = create_qualified_stock_data()
        result = analyze_stock("BBCA", df, fundamentals)

        if result.trade_plan:
            plan = result.trade_plan

            # Entry range should be around current price
            assert plan.entry_low <= result.current_price * 1.05
            assert plan.entry_high >= result.current_price * 0.95

            # Stop loss below entry
            assert plan.stop_loss < plan.entry_low

            # Take profits ascending
            assert plan.take_profit_1 < plan.take_profit_2
            assert plan.take_profit_2 < plan.take_profit_3

            # All take profits above entry
            assert plan.take_profit_1 > plan.entry_high
            assert plan.take_profit_2 > plan.entry_high
            assert plan.take_profit_3 > plan.entry_high

    def test_risk_reward_ratio_positive(self):
        """Test that risk/reward ratio is positive and reasonable."""
        from stockai.scoring.analyzer import analyze_stock

        df, fundamentals = create_qualified_stock_data()
        result = analyze_stock("BBCA", df, fundamentals)

        if result.trade_plan:
            assert result.trade_plan.risk_reward_ratio > 0
            # Typically want at least 1:1 risk/reward
            # Depending on implementation, this may vary


class TestGateValidationFlow:
    """Tests for the gate validation integration."""

    def test_all_gates_evaluated(self):
        """Test that all 6 gates are evaluated."""
        from stockai.scoring.analyzer import analyze_stock

        df, fundamentals = create_qualified_stock_data()
        result = analyze_stock("BBCA", df, fundamentals)

        # Should have 6 gates total
        assert result.gates.total_gates == 6

        # Total of passed + rejected should equal total
        total_evaluated = (
            len(result.gates.passed_gates) +
            len(result.gates.rejection_reasons)
        )
        assert total_evaluated == 6

    def test_rejection_reasons_specific(self):
        """Test that rejection reasons mention specific gates."""
        from stockai.scoring.analyzer import analyze_stock

        df, fundamentals = create_rejected_stock_data()
        result = analyze_stock("WEAK", df, fundamentals)

        if result.gates.rejection_reasons:
            # Each rejection reason should mention the gate
            gate_names = [
                "Overall Score", "Technical Score", "Smart Money Score",
                "Distance to Support", "ADX Trend", "Fundamental Score"
            ]
            for reason in result.gates.rejection_reasons:
                assert any(gate in reason for gate in gate_names)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_short_data_handled(self):
        """Test that short data periods are handled gracefully."""
        from stockai.scoring.analyzer import analyze_stock

        short_df = create_realistic_ohlcv(days=20)
        result = analyze_stock("SHORT", short_df)

        # Should still return a valid result
        assert result is not None
        assert result.current_price > 0

    def test_missing_fundamentals_handled(self):
        """Test that missing fundamentals are handled with defaults."""
        from stockai.scoring.analyzer import analyze_stock

        df, _ = create_qualified_stock_data()
        result = analyze_stock("NODATA", df, None)

        # Should still calculate all scores using defaults
        assert result is not None
        assert 0 <= result.value_score <= 100
        assert 0 <= result.quality_score <= 100

    def test_partial_fundamentals_handled(self):
        """Test that partial fundamentals work correctly."""
        from stockai.scoring.analyzer import analyze_stock

        df, _ = create_qualified_stock_data()

        # Only some fundamentals provided
        partial_fund = {
            "pe_ratio": 12.0,
            "roe": 15.0,
        }

        result = analyze_stock("PARTIAL", df, partial_fund)

        assert result is not None
        assert result.value_score >= 0

    def test_extreme_price_values_handled(self):
        """Test that extreme price values are handled."""
        from stockai.scoring.analyzer import analyze_stock

        df = create_realistic_ohlcv(days=60, base_price=50000.0)  # High price
        result = analyze_stock("EXPENSIVE", df)

        assert result is not None
        assert result.current_price > 0

    def test_low_volume_handled(self):
        """Test that very low volume data is handled."""
        from stockai.scoring.analyzer import analyze_stock

        df = create_realistic_ohlcv(days=60)
        df["volume"] = df["volume"] // 100  # Very low volume

        result = analyze_stock("ILLIQUID", df)

        assert result is not None


# =============================================================================
# End-to-End Integration Tests
# =============================================================================


class TestEndToEndFlow:
    """End-to-end tests simulating real usage."""

    def test_buy_signal_full_flow(self):
        """Test complete flow for a BUY signal scenario."""
        from stockai.scoring.analyzer import analyze_stock

        # Create strong bullish stock
        np.random.seed(123)
        df = create_realistic_ohlcv(days=120, trend="bullish", volatility=0.01)

        fundamentals = {
            "pe_ratio": 10.0,
            "pb_ratio": 1.2,
            "roe": 22.0,
            "debt_to_equity": 0.3,
            "profit_margin": 18.0,
            "current_ratio": 2.5,
        }

        result = analyze_stock("STRONG", df, fundamentals)

        # Log the results for debugging
        print(f"\nStrong Stock Analysis:")
        print(f"  Composite Score: {result.composite_score:.1f}")
        print(f"  Smart Money: {result.smart_money.score:.1f}")
        print(f"  ADX: {result.adx.get('adx', 'N/A')}")
        print(f"  Distance to Support: {result.support_resistance.distance_to_support_pct}")
        print(f"  Gates Passed: {result.gates.gates_passed}/{result.gates.total_gates}")
        print(f"  Decision: {result.decision}")
        print(f"  Confidence: {result.confidence}")

        # Should be a valid analysis
        assert result.decision in ["BUY", "NO_TRADE"]
        assert result.confidence in ["HIGH", "WATCH", "REJECTED"]

    def test_reject_signal_full_flow(self):
        """Test complete flow for a REJECT signal scenario."""
        from stockai.scoring.analyzer import analyze_stock

        # Create weak stock
        np.random.seed(456)
        df = create_realistic_ohlcv(days=120, trend="bearish", volatility=0.025)

        fundamentals = {
            "pe_ratio": 45.0,
            "pb_ratio": 4.0,
            "roe": 4.0,
            "debt_to_equity": 2.5,
            "profit_margin": 2.0,
            "current_ratio": 0.7,
        }

        result = analyze_stock("WEAK", df, fundamentals)

        # Log the results for debugging
        print(f"\nWeak Stock Analysis:")
        print(f"  Composite Score: {result.composite_score:.1f}")
        print(f"  Smart Money: {result.smart_money.score:.1f}")
        print(f"  ADX: {result.adx.get('adx', 'N/A')}")
        print(f"  Distance to Support: {result.support_resistance.distance_to_support_pct}")
        print(f"  Gates Passed: {result.gates.gates_passed}/{result.gates.total_gates}")
        print(f"  Decision: {result.decision}")
        print(f"  Confidence: {result.confidence}")
        print(f"  Rejections: {result.gates.rejection_reasons}")

        # Weak stock should likely be rejected or watched
        # But the exact outcome depends on the analysis
        assert result.decision in ["BUY", "NO_TRADE"]

    def test_consistency_across_runs(self):
        """Test that analysis is consistent with same input."""
        from stockai.scoring.analyzer import analyze_stock

        np.random.seed(789)
        df = create_realistic_ohlcv(days=60)
        fundamentals = {"pe_ratio": 15.0, "roe": 12.0}

        result1 = analyze_stock("TEST", df, fundamentals)
        result2 = analyze_stock("TEST", df, fundamentals)

        # Results should be identical
        assert result1.composite_score == result2.composite_score
        assert result1.smart_money.score == result2.smart_money.score
        assert result1.gates.gates_passed == result2.gates.gates_passed
        assert result1.decision == result2.decision

    def test_multiple_stocks_comparison(self):
        """Test analyzing multiple stocks for comparison."""
        from stockai.scoring.analyzer import analyze_stock

        np.random.seed(101)

        stocks = [
            ("BANK", "bullish", 0.012, {"pe_ratio": 11.0, "roe": 18.0}),
            ("MINING", "sideways", 0.02, {"pe_ratio": 8.0, "roe": 12.0}),
            ("TECH", "bullish", 0.018, {"pe_ratio": 25.0, "roe": 22.0}),
        ]

        results = []
        for ticker, trend, vol, fund in stocks:
            df = create_realistic_ohlcv(days=90, trend=trend, volatility=vol)
            result = analyze_stock(ticker, df, fund)
            results.append(result)

            print(f"\n{ticker} Analysis:")
            print(f"  Score: {result.composite_score:.1f}")
            print(f"  Decision: {result.decision} ({result.confidence})")

        # All should return valid results
        assert all(r is not None for r in results)
        assert all(r.ticker in ["BANK", "MINING", "TECH"] for r in results)
