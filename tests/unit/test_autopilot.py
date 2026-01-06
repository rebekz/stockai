"""Unit Tests for Autopilot Trading System.

Tests the autopilot engine, executor, and database integration:
- Configuration validation
- Signal generation thresholds
- Position sizing calculations
- Paper trading execution
- Database persistence
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from decimal import Decimal
import pytz

TIMEZONE = pytz.timezone("Asia/Jakarta")


class TestAutopilotConfig:
    """Test AutopilotConfig dataclass."""

    def test_default_config_values(self):
        """Test default configuration values match design spec."""
        from stockai.autopilot.engine import AutopilotConfig, IndexType

        config = AutopilotConfig()

        assert config.index == IndexType.JII70
        assert config.capital == 10_000_000
        assert config.max_risk_percent == 2.0
        assert config.max_position_percent == 20.0
        assert config.max_sector_percent == 40.0
        assert config.max_positions == 10
        assert config.buy_threshold == 70.0
        assert config.sell_threshold == 50.0
        assert config.atr_multiplier == 2.0
        assert config.dry_run is False

    def test_custom_config_values(self):
        """Test custom configuration values are applied."""
        from stockai.autopilot.engine import AutopilotConfig, IndexType

        config = AutopilotConfig(
            index=IndexType.IDX30,
            capital=50_000_000,
            max_risk_percent=1.5,
            dry_run=True,
        )

        assert config.index == IndexType.IDX30
        assert config.capital == 50_000_000
        assert config.max_risk_percent == 1.5
        assert config.dry_run is True


class TestIndexType:
    """Test IndexType enum."""

    def test_all_index_types_defined(self):
        """Test all required index types are defined."""
        from stockai.autopilot.engine import IndexType

        assert IndexType.JII70.value == "JII70"
        assert IndexType.IDX30.value == "IDX30"
        assert IndexType.LQ45.value == "LQ45"
        assert IndexType.ALL.value == "ALL"


class TestTradeSignal:
    """Test TradeSignal dataclass."""

    def test_trade_signal_creation(self):
        """Test TradeSignal can be created with all fields."""
        from stockai.autopilot.engine import TradeSignal

        signal = TradeSignal(
            symbol="PWON",
            action="BUY",
            score=78.5,
            current_price=340.0,
            lots=58,
            shares=5800,
            position_value=1_972_000.0,
            stop_loss=310.0,
            target=385.0,
            reason="Score 78 > 70",
        )

        assert signal.symbol == "PWON"
        assert signal.action == "BUY"
        assert signal.score == 78.5
        assert signal.lots == 58
        assert signal.shares == 5800


class TestAutopilotEngine:
    """Test AutopilotEngine class."""

    @pytest.fixture
    def mock_yahoo_source(self):
        """Create mock Yahoo Finance data source."""
        mock = MagicMock()
        mock.get_stock_info.return_value = {
            "symbol": "PWON",
            "pe_ratio": 8.5,
            "pb_ratio": 0.8,
        }
        mock.get_price_history.return_value = MagicMock()
        mock.get_price_history.return_value.empty = False
        mock.get_current_price.return_value = {"price": 340.0}
        return mock

    @pytest.fixture
    def mock_idx_source(self):
        """Create mock IDX index source."""
        mock = MagicMock()
        mock.get_jii70_symbols.return_value = ["PWON", "TLKM", "BBRI"]
        mock.get_idx30_symbols.return_value = ["BBCA", "BBRI", "TLKM"]
        mock.get_lq45_symbols.return_value = ["BBCA", "BBRI", "TLKM", "UNVR"]
        return mock

    def test_engine_initialization_with_default_config(self):
        """Test engine initializes with default configuration."""
        from stockai.autopilot.engine import AutopilotEngine, AutopilotConfig

        engine = AutopilotEngine()

        assert engine.config is not None
        assert engine.config.index.value == "JII70"
        assert engine.config.capital == 10_000_000

    def test_engine_initialization_with_custom_config(self):
        """Test engine initializes with custom configuration."""
        from stockai.autopilot.engine import AutopilotEngine, AutopilotConfig, IndexType

        config = AutopilotConfig(index=IndexType.IDX30, capital=25_000_000)
        engine = AutopilotEngine(config=config)

        assert engine.config.index == IndexType.IDX30
        assert engine.config.capital == 25_000_000

    def test_get_index_symbols_jii70(self, mock_idx_source):
        """Test fetching JII70 index symbols."""
        from stockai.autopilot.engine import AutopilotEngine, AutopilotConfig, IndexType

        config = AutopilotConfig(index=IndexType.JII70)
        engine = AutopilotEngine(config=config)
        engine.idx_source = mock_idx_source

        symbols = engine._get_index_symbols()

        assert symbols == ["PWON", "TLKM", "BBRI"]
        mock_idx_source.get_jii70_symbols.assert_called_once()

    def test_get_index_symbols_all_combines_indices(self, mock_idx_source):
        """Test ALL index type combines all indices."""
        from stockai.autopilot.engine import AutopilotEngine, AutopilotConfig, IndexType

        config = AutopilotConfig(index=IndexType.ALL)
        engine = AutopilotEngine(config=config)
        engine.idx_source = mock_idx_source

        symbols = engine._get_index_symbols()

        # Should be unique combined symbols
        assert set(symbols) == {"PWON", "TLKM", "BBRI", "BBCA", "UNVR"}

    def test_buy_signal_generated_when_score_above_threshold(self):
        """Test BUY signal is generated when score > 70."""
        from stockai.autopilot.engine import AutopilotEngine, AutopilotConfig
        from stockai.scoring.factors import FactorScores

        engine = AutopilotEngine()
        engine.yahoo_source = MagicMock()
        engine.yahoo_source.get_current_price.return_value = {"price": 340.0}
        engine.positions = {}

        scores = [
            FactorScores(
                symbol="PWON",
                value_score=85.0,
                quality_score=80.0,
                momentum_score=75.0,
                volatility_score=70.0,
                composite_score=78.0,  # Above 70 threshold
            ),
        ]

        buy_signals, sell_signals = engine._generate_signals(scores)

        assert len(buy_signals) == 1
        assert buy_signals[0].symbol == "PWON"
        assert buy_signals[0].action == "BUY"
        assert len(sell_signals) == 0

    def test_no_buy_signal_when_already_held(self):
        """Test no BUY signal for stocks already in portfolio."""
        from stockai.autopilot.engine import AutopilotEngine, AutopilotConfig
        from stockai.scoring.factors import FactorScores

        engine = AutopilotEngine()
        engine.yahoo_source = MagicMock()
        engine.yahoo_source.get_current_price.return_value = {"price": 340.0}
        engine.positions = {"PWON": {"lots": 10, "shares": 1000}}

        scores = [
            FactorScores(
                symbol="PWON",
                value_score=85.0,
                quality_score=80.0,
                momentum_score=75.0,
                volatility_score=70.0,
                composite_score=78.0,
            ),
        ]

        buy_signals, sell_signals = engine._generate_signals(scores)

        assert len(buy_signals) == 0

    def test_sell_signal_generated_when_score_below_threshold(self):
        """Test SELL signal is generated when score < 50 for held positions."""
        from stockai.autopilot.engine import AutopilotEngine, AutopilotConfig
        from stockai.scoring.factors import FactorScores

        engine = AutopilotEngine()
        engine.yahoo_source = MagicMock()
        engine.yahoo_source.get_current_price.return_value = {"price": 280.0}
        engine.positions = {"BRPT": {"lots": 10, "shares": 1000}}

        scores = [
            FactorScores(
                symbol="BRPT",
                value_score=40.0,
                quality_score=45.0,
                momentum_score=35.0,
                volatility_score=55.0,
                composite_score=43.0,  # Below 50 threshold
            ),
        ]

        buy_signals, sell_signals = engine._generate_signals(scores)

        assert len(buy_signals) == 0
        assert len(sell_signals) == 1
        assert sell_signals[0].symbol == "BRPT"
        assert sell_signals[0].action == "SELL"

    def test_dry_run_does_not_execute_trades(self, mock_yahoo_source, mock_idx_source):
        """Test dry run mode generates signals but doesn't execute trades."""
        from stockai.autopilot.engine import AutopilotEngine, AutopilotConfig

        config = AutopilotConfig(dry_run=True)
        engine = AutopilotEngine(config=config)
        engine.yahoo_source = mock_yahoo_source
        engine.idx_source = mock_idx_source

        # Mock the internal methods
        with patch.object(engine, "_scan_stocks", return_value=[]):
            with patch.object(engine, "_generate_signals", return_value=([], [])):
                with patch.object(engine, "_save_run_to_db", return_value=1):
                    result = engine.run()

        assert len(result.executed_buys) == 0
        assert len(result.executed_sells) == 0


class TestPaperExecutor:
    """Test PaperExecutor class."""

    @pytest.fixture
    def executor(self, tmp_path):
        """Create paper executor with temp portfolio file."""
        from stockai.autopilot.executor import PaperExecutor

        portfolio_file = str(tmp_path / "test_portfolio.json")
        return PaperExecutor(portfolio_file=portfolio_file)

    def test_create_portfolio(self, executor):
        """Test creating a new paper portfolio."""
        portfolio = executor.create_portfolio(initial_capital=10_000_000)

        assert portfolio is not None
        assert portfolio.initial_capital == 10_000_000
        assert portfolio.cash == 10_000_000
        assert len(portfolio.positions) == 0

    def test_buy_success(self, executor):
        """Test successful paper buy order."""
        executor.create_portfolio(initial_capital=10_000_000)

        result = executor.buy(
            symbol="PWON",
            lots=58,
            price=340.0,
            stop_loss=310.0,
            target=385.0,
        )

        assert result is True
        assert "PWON" in executor.portfolio.positions
        assert executor.portfolio.positions["PWON"].lots == 58
        assert executor.portfolio.positions["PWON"].shares == 5800
        assert executor.portfolio.cash == 10_000_000 - (5800 * 340.0)

    def test_buy_insufficient_cash(self, executor):
        """Test buy fails when insufficient cash."""
        executor.create_portfolio(initial_capital=1_000_000)

        # Try to buy more than we can afford
        result = executor.buy(
            symbol="TLKM",
            lots=100,  # 10,000 shares
            price=3800.0,  # Would cost 38M
        )

        assert result is False
        assert "TLKM" not in executor.portfolio.positions

    def test_sell_success(self, executor):
        """Test successful paper sell order."""
        executor.create_portfolio(initial_capital=10_000_000)
        executor.buy(symbol="PWON", lots=58, price=340.0)

        # Update price and sell
        executor.portfolio.positions["PWON"].current_price = 360.0
        proceeds = executor.sell(symbol="PWON", price=360.0)

        assert proceeds == 5800 * 360.0
        assert "PWON" not in executor.portfolio.positions

    def test_sell_partial(self, executor):
        """Test partial sell order."""
        executor.create_portfolio(initial_capital=10_000_000)
        executor.buy(symbol="PWON", lots=100, price=340.0)

        proceeds = executor.sell(symbol="PWON", lots=50, price=350.0)

        assert proceeds == 5000 * 350.0
        assert "PWON" in executor.portfolio.positions
        assert executor.portfolio.positions["PWON"].lots == 50

    def test_sell_nonexistent_position(self, executor):
        """Test sell fails for non-existent position."""
        executor.create_portfolio(initial_capital=10_000_000)

        proceeds = executor.sell(symbol="NONEXISTENT")

        assert proceeds == 0

    def test_portfolio_value_calculation(self, executor):
        """Test portfolio total value calculation."""
        executor.create_portfolio(initial_capital=10_000_000)
        executor.buy(symbol="PWON", lots=58, price=340.0)

        # Update price
        executor.update_prices({"PWON": 360.0})

        # Expected: remaining cash + position value
        expected_cash = 10_000_000 - (5800 * 340.0)
        expected_position_value = 5800 * 360.0
        expected_total = expected_cash + expected_position_value

        assert executor.portfolio.total_value == pytest.approx(expected_total)

    def test_pnl_calculation(self, executor):
        """Test P&L calculation."""
        executor.create_portfolio(initial_capital=10_000_000)
        executor.buy(symbol="PWON", lots=58, price=340.0)

        # Price went up
        executor.update_prices({"PWON": 360.0})

        position = executor.portfolio.positions["PWON"]
        assert position.pnl == pytest.approx((360.0 - 340.0) * 5800)
        assert position.pnl_pct == pytest.approx(((360.0 / 340.0) - 1) * 100)

    def test_portfolio_persistence(self, executor):
        """Test portfolio saves and loads correctly."""
        from stockai.autopilot.executor import PaperExecutor

        # Create and populate portfolio
        executor.create_portfolio(initial_capital=10_000_000)
        executor.buy(symbol="PWON", lots=58, price=340.0)

        # Create new executor with same file
        executor2 = PaperExecutor(portfolio_file=executor.portfolio_file)
        portfolio2 = executor2.load_portfolio()

        assert portfolio2 is not None
        assert portfolio2.initial_capital == 10_000_000
        assert "PWON" in portfolio2.positions
        assert portfolio2.positions["PWON"].lots == 58


class TestAutopilotDatabaseModels:
    """Test autopilot database models."""

    def test_autopilot_run_model_creation(self):
        """Test AutopilotRun model can be instantiated."""
        from stockai.data.models import AutopilotRun

        run = AutopilotRun(
            run_date=datetime.now(TIMEZONE),
            index_scanned="JII70",
            stocks_scanned=70,
            signals_generated=5,
            trades_executed=3,
            initial_capital=Decimal("10000000"),
            final_value=Decimal("10250000"),
            is_dry_run=False,
        )

        assert run.index_scanned == "JII70"
        assert run.stocks_scanned == 70
        assert run.trades_executed == 3

    def test_autopilot_trade_model_creation(self):
        """Test AutopilotTrade model can be instantiated."""
        from stockai.data.models import AutopilotTrade

        trade = AutopilotTrade(
            run_id=1,
            symbol="PWON",
            action="BUY",
            lots=58,
            shares=5800,
            price=Decimal("340"),
            total_value=Decimal("1972000"),
            score=78.0,
            reason="Score 78 > 70",
            stop_loss=Decimal("310"),
            target=Decimal("385"),
        )

        assert trade.symbol == "PWON"
        assert trade.action == "BUY"
        assert trade.lots == 58


class TestFormatFunctions:
    """Test output formatting functions."""

    def test_format_autopilot_result(self):
        """Test autopilot result formatting."""
        from stockai.autopilot.engine import AutopilotResult, format_autopilot_result, TradeSignal

        result = AutopilotResult(
            run_date=datetime.now(TIMEZONE),
            index_scanned="JII70",
            capital=10_000_000,
            stocks_scanned=70,
        )
        result.buy_signals = [
            TradeSignal(
                symbol="PWON",
                action="BUY",
                score=78.0,
                current_price=340.0,
                lots=58,
                shares=5800,
                position_value=1_972_000.0,
                stop_loss=310.0,
                target=385.0,
                reason="Score 78 > 70",
            )
        ]
        result.executed_buys = result.buy_signals

        output = format_autopilot_result(result)

        assert "AUTOPILOT RUN" in output
        assert "JII70" in output
        assert "PWON" in output
        assert "58 lots" in output

    def test_format_autopilot_history(self):
        """Test autopilot history formatting."""
        from stockai.autopilot.engine import format_autopilot_history

        history = [
            {
                "id": 1,
                "run_date": datetime.now(TIMEZONE),
                "index_scanned": "JII70",
                "stocks_scanned": 70,
                "signals_generated": 5,
                "trades_executed": 2,
                "initial_capital": 10_000_000,
                "final_value": 10_100_000,
                "is_dry_run": False,
                "trades": [
                    {
                        "symbol": "PWON",
                        "action": "BUY",
                        "lots": 58,
                        "price": 340.0,
                        "total_value": 1_972_000.0,
                        "score": 78.0,
                        "reason": "Score 78 > 70",
                    }
                ],
            }
        ]

        output = format_autopilot_history(history)

        assert "AUTOPILOT HISTORY" in output
        assert "Run #1" in output
        assert "PWON" in output

    def test_format_autopilot_history_empty(self):
        """Test empty history formatting."""
        from stockai.autopilot.engine import format_autopilot_history

        output = format_autopilot_history([])

        assert "No autopilot history found" in output


class TestPaperPortfolio:
    """Test PaperPortfolio dataclass."""

    def test_portfolio_total_pnl_calculation(self):
        """Test total P&L calculation."""
        from stockai.autopilot.executor import PaperPortfolio, PaperPosition

        now = datetime.now(TIMEZONE)
        portfolio = PaperPortfolio(
            initial_capital=10_000_000,
            cash=8_000_000,
            positions={
                "PWON": PaperPosition(
                    symbol="PWON",
                    lots=58,
                    shares=5800,
                    avg_price=340.0,
                    current_price=360.0,
                    stop_loss=310.0,
                    target=385.0,
                    entry_date=now,
                )
            },
            created_at=now,
            updated_at=now,
        )

        # Total value: 8M cash + 5800 * 360 = 8M + 2.088M = 10.088M
        # P&L: 10.088M - 10M = 88,000
        assert portfolio.total_value == 8_000_000 + 5800 * 360.0
        assert portfolio.total_pnl == portfolio.total_value - 10_000_000

    def test_portfolio_pnl_percentage(self):
        """Test P&L percentage calculation."""
        from stockai.autopilot.executor import PaperPortfolio

        now = datetime.now(TIMEZONE)
        portfolio = PaperPortfolio(
            initial_capital=10_000_000,
            cash=10_500_000,  # 5% gain, no positions
            positions={},
            created_at=now,
            updated_at=now,
        )

        assert portfolio.total_pnl_pct == pytest.approx(5.0)


# ============================================================================
# AI VALIDATION TESTS
# ============================================================================


class TestAIValidatorConfig:
    """Test AIValidatorConfig dataclass."""

    def test_default_config_values(self):
        """Test default AI validator configuration values."""
        from stockai.autopilot.validator import AIValidatorConfig

        config = AIValidatorConfig()

        assert config.buy_threshold == 6.0
        assert config.sell_threshold == 5.0
        assert config.max_concurrent == 3
        assert config.per_stock_timeout == 60.0
        assert config.max_retries == 1
        assert config.retry_delay == 2.0

    def test_custom_config_values(self):
        """Test custom AI validator configuration."""
        from stockai.autopilot.validator import AIValidatorConfig

        config = AIValidatorConfig(
            buy_threshold=7.0,
            sell_threshold=4.0,
            max_concurrent=5,
        )

        assert config.buy_threshold == 7.0
        assert config.sell_threshold == 4.0
        assert config.max_concurrent == 5


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_validation_result_creation(self):
        """Test ValidationResult can be created with all fields."""
        from stockai.autopilot.validator import ValidationResult

        result = ValidationResult(
            symbol="PWON",
            signal_type="BUY",
            autopilot_score=78.0,
            ai_composite_score=7.2,
            is_approved=True,
            recommendation="STRONG_BUY",
            fundamental_score=8.0,
            technical_score=7.5,
            sentiment_score=6.5,
        )

        assert result.symbol == "PWON"
        assert result.signal_type == "BUY"
        assert result.autopilot_score == 78.0
        assert result.ai_composite_score == 7.2
        assert result.is_approved is True
        assert result.recommendation == "STRONG_BUY"
        assert result.fundamental_score == 8.0

    def test_validation_result_with_rejection(self):
        """Test ValidationResult with rejection reason."""
        from stockai.autopilot.validator import ValidationResult

        result = ValidationResult(
            symbol="TLKM",
            signal_type="BUY",
            autopilot_score=72.0,
            ai_composite_score=5.4,
            is_approved=False,
            recommendation="HOLD",
            rejection_reason="AI score 5.4 below BUY threshold 6.0",
        )

        assert result.is_approved is False
        assert result.rejection_reason is not None
        assert "below BUY threshold" in result.rejection_reason

    def test_validation_result_default_lists(self):
        """Test ValidationResult default lists are empty."""
        from stockai.autopilot.validator import ValidationResult

        result = ValidationResult(
            symbol="TEST",
            signal_type="BUY",
            autopilot_score=50.0,
            ai_composite_score=5.0,
            is_approved=False,
            recommendation="HOLD",
        )

        assert result.key_reasons == []
        assert result.risk_factors == []


class TestAIValidator:
    """Test AIValidator class."""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create mock TradingOrchestrator."""
        mock = MagicMock()
        mock.run.return_value = {
            "composite_score": 7.5,
            "recommendation": "BUY",
            "answer": "Strong momentum indicators. Solid fundamentals.",
            "fundamental_analysis": {"score": 8.0},
            "technical_analysis": {"score": 7.5},
            "sentiment_analysis": {"score": 6.5},
            "portfolio_recommendation": {"score": 7.0},
            "risk_assessment": {"score": 6.0},
        }
        return mock

    def test_validator_initialization(self):
        """Test AIValidator initializes with default config."""
        from stockai.autopilot.validator import AIValidator

        validator = AIValidator()

        assert validator.config is not None
        assert validator.config.buy_threshold == 6.0
        assert validator.config.sell_threshold == 5.0

    def test_validator_custom_config(self):
        """Test AIValidator with custom configuration."""
        from stockai.autopilot.validator import AIValidator, AIValidatorConfig

        config = AIValidatorConfig(buy_threshold=7.0)
        validator = AIValidator(config=config)

        assert validator.config.buy_threshold == 7.0

    @pytest.mark.asyncio
    async def test_buy_approved_when_ai_score_above_threshold(self, mock_orchestrator):
        """Test BUY signal is approved when AI score >= threshold."""
        from stockai.autopilot.validator import AIValidator, AIValidatorConfig

        config = AIValidatorConfig(buy_threshold=6.0)
        validator = AIValidator(config=config, orchestrator=mock_orchestrator)

        result = await validator.validate_signal("PWON", "BUY", 78.0)

        assert result.is_approved is True
        assert result.ai_composite_score == 7.5
        assert result.rejection_reason is None

    @pytest.mark.asyncio
    async def test_buy_rejected_when_ai_score_below_threshold(self, mock_orchestrator):
        """Test BUY signal is rejected when AI score < threshold."""
        from stockai.autopilot.validator import AIValidator, AIValidatorConfig

        # Mock returns low score
        mock_orchestrator.run.return_value = {
            "composite_score": 5.2,
            "recommendation": "HOLD",
        }

        config = AIValidatorConfig(buy_threshold=6.0)
        validator = AIValidator(config=config, orchestrator=mock_orchestrator)

        result = await validator.validate_signal("TLKM", "BUY", 72.0)

        assert result.is_approved is False
        assert result.ai_composite_score == 5.2
        assert "below BUY threshold" in result.rejection_reason

    @pytest.mark.asyncio
    async def test_sell_confirmed_when_ai_score_below_threshold(self, mock_orchestrator):
        """Test SELL signal is confirmed when AI score <= threshold."""
        from stockai.autopilot.validator import AIValidator, AIValidatorConfig

        mock_orchestrator.run.return_value = {
            "composite_score": 4.1,
            "recommendation": "SELL",
        }

        config = AIValidatorConfig(sell_threshold=5.0)
        validator = AIValidator(config=config, orchestrator=mock_orchestrator)

        result = await validator.validate_signal("BRPT", "SELL", 43.0)

        assert result.is_approved is True
        assert result.ai_composite_score == 4.1
        assert result.rejection_reason is None

    @pytest.mark.asyncio
    async def test_sell_rejected_when_ai_score_above_threshold(self, mock_orchestrator):
        """Test SELL signal is rejected when AI score > threshold."""
        from stockai.autopilot.validator import AIValidator, AIValidatorConfig

        mock_orchestrator.run.return_value = {
            "composite_score": 6.5,
            "recommendation": "HOLD",
        }

        config = AIValidatorConfig(sell_threshold=5.0)
        validator = AIValidator(config=config, orchestrator=mock_orchestrator)

        result = await validator.validate_signal("BBCA", "SELL", 48.0)

        assert result.is_approved is False
        assert "above SELL threshold" in result.rejection_reason

    @pytest.mark.asyncio
    async def test_score_exactly_at_threshold_is_approved(self, mock_orchestrator):
        """Test BUY at exactly threshold (6.0) is approved (>= is inclusive)."""
        from stockai.autopilot.validator import AIValidator, AIValidatorConfig

        mock_orchestrator.run.return_value = {
            "composite_score": 6.0,  # Exactly at threshold
            "recommendation": "BUY",
        }

        config = AIValidatorConfig(buy_threshold=6.0)
        validator = AIValidator(config=config, orchestrator=mock_orchestrator)

        result = await validator.validate_signal("TEST", "BUY", 70.0)

        assert result.is_approved is True

    @pytest.mark.asyncio
    async def test_api_failure_marks_signal_unavailable(self, mock_orchestrator):
        """Test API failure creates unavailable result."""
        from stockai.autopilot.validator import AIValidator

        mock_orchestrator.run.side_effect = Exception("API Error")
        validator = AIValidator(orchestrator=mock_orchestrator)

        result = await validator.validate_signal("FAIL", "BUY", 75.0)

        assert result.is_approved is False
        assert result.recommendation == "AI_UNAVAILABLE"
        assert "AI unavailable" in result.rejection_reason

    @pytest.mark.asyncio
    async def test_batch_validation(self, mock_orchestrator):
        """Test batch validation with multiple signals."""
        from stockai.autopilot.validator import AIValidator

        validator = AIValidator(orchestrator=mock_orchestrator)

        signals = [
            ("PWON", "BUY", 78.0),
            ("TLKM", "BUY", 72.0),
            ("BBCA", "BUY", 75.0),
        ]

        results = await validator.validate_signals_batch(signals)

        assert len(results) == 3
        assert all(r.is_approved for r in results)  # All should pass (score 7.5)

    @pytest.mark.asyncio
    async def test_batch_validation_empty_list(self, mock_orchestrator):
        """Test batch validation with empty list returns empty."""
        from stockai.autopilot.validator import AIValidator

        validator = AIValidator(orchestrator=mock_orchestrator)
        results = await validator.validate_signals_batch([])

        assert results == []


class TestCreateValidator:
    """Test create_validator factory function."""

    def test_create_validator_with_defaults(self):
        """Test create_validator with default parameters."""
        from stockai.autopilot.validator import create_validator

        validator = create_validator()

        assert validator.config.buy_threshold == 6.0
        assert validator.config.sell_threshold == 5.0
        assert validator.config.max_concurrent == 3

    def test_create_validator_with_custom_params(self):
        """Test create_validator with custom parameters."""
        from stockai.autopilot.validator import create_validator

        validator = create_validator(
            buy_threshold=7.5,
            sell_threshold=4.0,
            max_concurrent=5,
        )

        assert validator.config.buy_threshold == 7.5
        assert validator.config.sell_threshold == 4.0
        assert validator.config.max_concurrent == 5


class TestAutopilotConfigAI:
    """Test AI-related AutopilotConfig fields."""

    def test_default_ai_config_values(self):
        """Test default AI configuration in AutopilotConfig."""
        from stockai.autopilot.engine import AutopilotConfig

        config = AutopilotConfig()

        assert config.ai_enabled is True
        assert config.ai_buy_threshold == 6.0
        assert config.ai_sell_threshold == 5.0
        assert config.ai_concurrency == 3
        assert config.ai_verbose is False

    def test_custom_ai_config_values(self):
        """Test custom AI configuration in AutopilotConfig."""
        from stockai.autopilot.engine import AutopilotConfig

        config = AutopilotConfig(
            ai_enabled=False,
            ai_buy_threshold=7.0,
            ai_concurrency=5,
        )

        assert config.ai_enabled is False
        assert config.ai_buy_threshold == 7.0
        assert config.ai_concurrency == 5


class TestTradeSignalAIFields:
    """Test AI-related TradeSignal fields."""

    def test_trade_signal_ai_fields_defaults(self):
        """Test TradeSignal AI fields have correct defaults."""
        from stockai.autopilot.engine import TradeSignal

        signal = TradeSignal(
            symbol="TEST",
            action="BUY",
            score=70.0,
            current_price=100.0,
            lots=10,
            shares=1000,
            position_value=100_000.0,
            stop_loss=90.0,
            target=120.0,
            reason="Score 70 > 70",
        )

        assert signal.ai_validated is False
        assert signal.ai_score is None
        assert signal.ai_approved is None
        assert signal.ai_recommendation is None
        assert signal.ai_rejection_reason is None
        assert signal.ai_key_reasons == []
        assert signal.ai_risk_factors == []

    def test_trade_signal_with_ai_validation(self):
        """Test TradeSignal with AI validation fields populated."""
        from stockai.autopilot.engine import TradeSignal

        signal = TradeSignal(
            symbol="PWON",
            action="BUY",
            score=78.0,
            current_price=340.0,
            lots=58,
            shares=5800,
            position_value=1_972_000.0,
            stop_loss=310.0,
            target=385.0,
            reason="Score 78 > 70",
            ai_validated=True,
            ai_score=7.2,
            ai_approved=True,
            ai_recommendation="STRONG_BUY",
            ai_key_reasons=["Strong momentum", "Solid fundamentals"],
        )

        assert signal.ai_validated is True
        assert signal.ai_score == 7.2
        assert signal.ai_approved is True
        assert len(signal.ai_key_reasons) == 2


class TestAutopilotValidationModel:
    """Test AutopilotValidation database model."""

    def test_validation_model_creation(self):
        """Test AutopilotValidation model can be instantiated."""
        from stockai.data.models import AutopilotValidation

        validation = AutopilotValidation(
            run_id=1,
            symbol="PWON",
            signal_type="BUY",
            autopilot_score=78.0,
            ai_composite_score=7.2,
            ai_fundamental_score=8.0,
            ai_technical_score=7.5,
            ai_sentiment_score=6.5,
            ai_risk_score=6.0,
            ai_recommendation="STRONG_BUY",
            is_approved=True,
        )

        assert validation.symbol == "PWON"
        assert validation.signal_type == "BUY"
        assert validation.ai_composite_score == 7.2
        assert validation.is_approved is True

    def test_validation_model_with_rejection(self):
        """Test AutopilotValidation model with rejection."""
        from stockai.data.models import AutopilotValidation

        validation = AutopilotValidation(
            run_id=1,
            symbol="TLKM",
            signal_type="BUY",
            autopilot_score=72.0,
            ai_composite_score=5.4,
            ai_recommendation="HOLD",
            is_approved=False,
            rejection_reason="AI score 5.4 below BUY threshold 6.0",
        )

        assert validation.is_approved is False
        assert validation.rejection_reason is not None


class TestAutopilotTradeAIFields:
    """Test AI fields added to AutopilotTrade model."""

    def test_autopilot_trade_with_ai_fields(self):
        """Test AutopilotTrade model with AI validation fields."""
        from stockai.data.models import AutopilotTrade

        trade = AutopilotTrade(
            run_id=1,
            symbol="PWON",
            action="BUY",
            lots=58,
            shares=5800,
            price=Decimal("340"),
            total_value=Decimal("1972000"),
            score=78.0,
            ai_validated=True,
            ai_composite_score=7.2,
            ai_fundamental_score=8.0,
            ai_technical_score=7.5,
            ai_sentiment_score=6.5,
            ai_risk_score=6.0,
            ai_recommendation="STRONG_BUY",
            ai_approved=True,
        )

        assert trade.ai_validated is True
        assert trade.ai_composite_score == 7.2
        assert trade.ai_approved is True

    def test_autopilot_trade_ai_rejection(self):
        """Test AutopilotTrade model with AI rejection."""
        from stockai.data.models import AutopilotTrade

        trade = AutopilotTrade(
            run_id=1,
            symbol="TLKM",
            action="BUY",
            lots=0,
            shares=0,
            price=Decimal("0"),
            total_value=Decimal("0"),
            score=72.0,
            ai_validated=True,
            ai_composite_score=5.4,
            ai_recommendation="HOLD",
            ai_approved=False,
            ai_rejection_reason="AI score 5.4 below BUY threshold 6.0",
        )

        assert trade.ai_validated is True
        assert trade.ai_approved is False
        assert trade.ai_rejection_reason is not None
