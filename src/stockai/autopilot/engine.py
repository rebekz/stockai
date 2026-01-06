"""Autopilot Engine - Main Orchestration.

Coordinates the daily trading workflow:
1. SCAN: Load portfolio, fetch prices, calculate scores
2. SIGNAL: Generate BUY/SELL signals based on thresholds
3. AI GATE: Validate signals with 7-agent AI orchestrator (optional)
4. SIZING: Calculate safe position sizes (approved signals only)
5. EXECUTE: Paper trading execution
6. REPORT: Display and log results with AI insights
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import Any, TYPE_CHECKING
import logging
import pytz

if TYPE_CHECKING:
    from stockai.autopilot.validator import AIValidator, ValidationResult

from stockai.data.database import get_session
from stockai.data.models import AutopilotRun, AutopilotTrade, AutopilotValidation
from stockai.data.sources.idx import IDXIndexSource
from stockai.data.sources.yahoo import YahooFinanceSource
from stockai.scoring.factors import score_stock, FactorScores
from stockai.scoring.signals import SignalGenerator, Signal, SignalType
from stockai.risk.position_sizing import calculate_position_size, PositionSize

logger = logging.getLogger(__name__)

TIMEZONE = pytz.timezone("Asia/Jakarta")


class IndexType(Enum):
    """Supported index types."""
    JII70 = "JII70"
    IDX30 = "IDX30"
    LQ45 = "LQ45"
    ALL = "ALL"


@dataclass
class AutopilotConfig:
    """Configuration for autopilot trading."""

    # Index selection
    index: IndexType = IndexType.JII70

    # Capital management
    capital: float = 10_000_000  # Default 10M IDR

    # Risk parameters
    max_risk_percent: float = 2.0  # 2% risk per trade
    max_position_percent: float = 20.0  # Max 20% per stock
    max_sector_percent: float = 40.0  # Max 40% per sector
    max_positions: int = 10  # Max concurrent positions

    # Signal thresholds
    buy_threshold: float = 70.0  # Score > 70 = BUY
    sell_threshold: float = 50.0  # Score < 50 = SELL

    # ATR multiplier for stop-loss
    atr_multiplier: float = 2.0

    # Execution mode
    dry_run: bool = False

    # AI Validation (Phase 3: AI Gate)
    ai_enabled: bool = True  # Enable AI validation
    ai_buy_threshold: float = 6.0  # AI score >= 6.0 for BUY approval
    ai_sell_threshold: float = 5.0  # AI score <= 5.0 for SELL confirmation
    ai_concurrency: int = 3  # Parallel validation limit
    ai_verbose: bool = False  # Show detailed agent analysis


@dataclass
class TradeSignal:
    """A trade signal with position sizing and AI validation."""

    symbol: str
    action: str  # BUY or SELL
    score: float
    current_price: float
    lots: int
    shares: int
    position_value: float
    stop_loss: float | None
    target: float | None
    reason: str
    factor_scores: FactorScores | None = None

    # AI Validation fields
    ai_validated: bool = False  # Has been validated by AI
    ai_score: float | None = None  # AI composite score (0-10)
    ai_approved: bool | None = None  # AI approval status
    ai_recommendation: str | None = None  # AI recommendation string
    ai_rejection_reason: str | None = None  # Reason if rejected
    ai_key_reasons: list[str] = field(default_factory=list)  # AI insights
    ai_risk_factors: list[str] = field(default_factory=list)  # AI risks


@dataclass
class AutopilotResult:
    """Result of an autopilot run."""

    run_date: datetime
    index_scanned: str
    capital: float

    # Scan results
    stocks_scanned: int = 0
    scores: list[FactorScores] = field(default_factory=list)

    # Signals (before AI validation)
    buy_signals: list[TradeSignal] = field(default_factory=list)
    sell_signals: list[TradeSignal] = field(default_factory=list)

    # AI Validation results
    ai_enabled: bool = False
    ai_approved_buys: list[TradeSignal] = field(default_factory=list)
    ai_rejected_buys: list[TradeSignal] = field(default_factory=list)
    ai_approved_sells: list[TradeSignal] = field(default_factory=list)
    ai_rejected_sells: list[TradeSignal] = field(default_factory=list)

    # Execution
    executed_buys: list[TradeSignal] = field(default_factory=list)
    executed_sells: list[TradeSignal] = field(default_factory=list)

    # Portfolio state
    portfolio_value: float = 0
    cash_remaining: float = 0
    positions_count: int = 0

    # Alerts
    alerts: list[str] = field(default_factory=list)

    # Errors
    errors: list[str] = field(default_factory=list)


class AutopilotEngine:
    """Main autopilot trading engine with AI validation."""

    def __init__(self, config: AutopilotConfig | None = None):
        """Initialize autopilot engine.

        Args:
            config: Autopilot configuration
        """
        self.config = config or AutopilotConfig()
        self.idx_source = IDXIndexSource()
        self.yahoo_source = YahooFinanceSource()
        self.signal_generator = SignalGenerator()

        # Paper portfolio state (would load from DB in production)
        self.positions: dict[str, dict] = {}
        self.cash = self.config.capital

        # AI Validator (lazy-loaded to avoid LLM init overhead)
        self._ai_validator: AIValidator | None = None

    @property
    def ai_validator(self) -> AIValidator:
        """Lazy-load AI validator to avoid LLM initialization on import."""
        if self._ai_validator is None:
            from stockai.autopilot.validator import AIValidator, AIValidatorConfig

            validator_config = AIValidatorConfig(
                buy_threshold=self.config.ai_buy_threshold,
                sell_threshold=self.config.ai_sell_threshold,
                max_concurrent=self.config.ai_concurrency,
            )
            self._ai_validator = AIValidator(config=validator_config)
        return self._ai_validator

    def run(
        self,
        portfolio: dict[str, Any] | None = None,
    ) -> AutopilotResult:
        """Execute the daily autopilot workflow.

        Args:
            portfolio: Current portfolio state (positions, cash)

        Returns:
            AutopilotResult with all run details
        """
        result = AutopilotResult(
            run_date=datetime.now(TIMEZONE),
            index_scanned=self.config.index.value,
            capital=self.config.capital,
            ai_enabled=self.config.ai_enabled,
        )

        # Load portfolio if provided
        if portfolio:
            self.positions = portfolio.get("positions", {})
            self.cash = portfolio.get("cash", self.config.capital)

        result.cash_remaining = self.cash

        # Phase 1: SCAN
        logger.info(f"Scanning {self.config.index.value} stocks...")
        symbols = self._get_index_symbols()
        result.stocks_scanned = len(symbols)

        scores = self._scan_stocks(symbols)
        result.scores = scores

        # Phase 2: SIGNAL GENERATION
        logger.info("Generating signals...")
        buy_signals, sell_signals = self._generate_signals(scores)
        result.buy_signals = buy_signals
        result.sell_signals = sell_signals

        # Phase 3: AI GATE (optional)
        if self.config.ai_enabled and (buy_signals or sell_signals):
            logger.info("Running AI validation...")
            approved_buys, rejected_buys = self._validate_signals_with_ai(buy_signals)
            approved_sells, rejected_sells = self._validate_signals_with_ai(sell_signals)

            result.ai_approved_buys = approved_buys
            result.ai_rejected_buys = rejected_buys
            result.ai_approved_sells = approved_sells
            result.ai_rejected_sells = rejected_sells

            # Warn if all signals were rejected by AI
            total_signals = len(buy_signals) + len(sell_signals)
            total_approved = len(approved_buys) + len(approved_sells)
            if total_signals > 0 and total_approved == 0:
                logger.warning(
                    f"All {total_signals} signals were rejected by AI validation. "
                    "Consider using --no-ai flag if AI validation is unavailable or too strict."
                )

            # Only size and execute approved signals
            signals_to_size = approved_buys
            signals_to_sell = approved_sells
        else:
            # AI disabled - all signals proceed
            signals_to_size = buy_signals
            signals_to_sell = sell_signals

        # Phase 4: POSITION SIZING (for approved buys only)
        logger.info("Calculating position sizes...")
        sized_buys = self._size_positions(signals_to_size)

        # Phase 5: EXECUTION
        if not self.config.dry_run:
            logger.info("Executing trades...")
            # Execute sells first (free up capital)
            result.executed_sells = self._execute_sells(signals_to_sell)
            # Then execute buys (by score descending)
            result.executed_buys = self._execute_buys(sized_buys)

        # Phase 6: REPORTING
        result.portfolio_value = self._calculate_portfolio_value()
        result.cash_remaining = self.cash
        result.positions_count = len(self.positions)

        # Generate alerts
        result.alerts = self._check_alerts()

        # Save to database
        self._save_run_to_db(result)

        return result

    def _validate_signals_with_ai(
        self, signals: list[TradeSignal]
    ) -> tuple[list[TradeSignal], list[TradeSignal]]:
        """Validate signals with AI orchestrator.

        Args:
            signals: List of trade signals to validate

        Returns:
            Tuple of (approved_signals, rejected_signals)
        """
        if not signals:
            return [], []

        # Prepare batch for validation
        validation_requests = [
            (s.symbol, s.action, s.score) for s in signals
        ]

        # Run async validation
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        validation_results = loop.run_until_complete(
            self.ai_validator.validate_signals_batch(validation_requests)
        )

        # Apply validation results to signals
        approved = []
        rejected = []

        for signal, result in zip(signals, validation_results):
            # Update signal with AI results
            signal.ai_validated = True
            signal.ai_score = result.ai_composite_score
            signal.ai_approved = result.is_approved
            signal.ai_recommendation = result.recommendation
            signal.ai_key_reasons = result.key_reasons
            signal.ai_risk_factors = result.risk_factors

            if result.is_approved:
                approved.append(signal)
                logger.info(
                    f"{signal.action} {signal.symbol}: AI Score {result.ai_composite_score:.1f} ✓ APPROVED"
                )
            else:
                signal.ai_rejection_reason = result.rejection_reason
                rejected.append(signal)
                logger.info(
                    f"{signal.action} {signal.symbol}: AI Score {result.ai_composite_score:.1f} ✗ REJECTED "
                    f"({result.rejection_reason})"
                )

        return approved, rejected

    def _save_run_to_db(self, result: AutopilotResult) -> int | None:
        """Save autopilot run, trades, and AI validations to database.

        Args:
            result: AutopilotResult to save

        Returns:
            Run ID if saved, None if error
        """
        try:
            session = get_session()

            # Create run record
            run = AutopilotRun(
                run_date=result.run_date,
                index_scanned=result.index_scanned,
                stocks_scanned=result.stocks_scanned,
                signals_generated=len(result.buy_signals) + len(result.sell_signals),
                trades_executed=len(result.executed_buys) + len(result.executed_sells),
                initial_capital=Decimal(str(result.capital)),
                final_value=Decimal(str(result.portfolio_value)),
                is_dry_run=self.config.dry_run,
            )
            session.add(run)
            session.flush()  # Get run ID

            # Save executed buy trades with AI validation data
            for trade in result.executed_buys:
                trade_record = AutopilotTrade(
                    run_id=run.id,
                    symbol=trade.symbol,
                    action="BUY",
                    lots=trade.lots,
                    shares=trade.shares,
                    price=Decimal(str(trade.current_price)),
                    total_value=Decimal(str(trade.position_value)),
                    score=trade.score,
                    reason=trade.reason,
                    stop_loss=Decimal(str(trade.stop_loss)) if trade.stop_loss else None,
                    target=Decimal(str(trade.target)) if trade.target else None,
                    # AI validation fields
                    ai_validated=trade.ai_validated,
                    ai_composite_score=trade.ai_score,
                    ai_recommendation=trade.ai_recommendation,
                    ai_approved=trade.ai_approved,
                    ai_rejection_reason=trade.ai_rejection_reason,
                )
                session.add(trade_record)

            # Save executed sell trades with AI validation data
            for trade in result.executed_sells:
                trade_record = AutopilotTrade(
                    run_id=run.id,
                    symbol=trade.symbol,
                    action="SELL",
                    lots=trade.lots,
                    shares=trade.shares,
                    price=Decimal(str(trade.current_price)),
                    total_value=Decimal(str(trade.position_value)),
                    score=trade.score if trade.score else None,
                    reason=trade.reason,
                    # AI validation fields
                    ai_validated=trade.ai_validated,
                    ai_composite_score=trade.ai_score,
                    ai_recommendation=trade.ai_recommendation,
                    ai_approved=trade.ai_approved,
                    ai_rejection_reason=trade.ai_rejection_reason,
                )
                session.add(trade_record)

            # Save all AI validations (including rejections) for tracking
            if result.ai_enabled:
                all_validated_signals = (
                    result.ai_approved_buys +
                    result.ai_rejected_buys +
                    result.ai_approved_sells +
                    result.ai_rejected_sells
                )
                for signal in all_validated_signals:
                    validation_record = AutopilotValidation(
                        run_id=run.id,
                        symbol=signal.symbol,
                        signal_type=signal.action,
                        autopilot_score=signal.score,
                        ai_composite_score=signal.ai_score,
                        ai_recommendation=signal.ai_recommendation,
                        is_approved=signal.ai_approved if signal.ai_approved is not None else False,
                        rejection_reason=signal.ai_rejection_reason,
                    )
                    session.add(validation_record)

            session.commit()
            logger.info(f"Saved autopilot run {run.id} to database")
            return run.id

        except Exception as e:
            logger.error(f"Error saving autopilot run to database: {e}")
            result.errors.append(f"Database save error: {e}")
            return None

    def _get_index_symbols(self) -> list[str]:
        """Get stock symbols for the configured index."""
        if self.config.index == IndexType.JII70:
            return self.idx_source.get_jii70_symbols()
        elif self.config.index == IndexType.IDX30:
            return self.idx_source.get_idx30_symbols()
        elif self.config.index == IndexType.LQ45:
            return self.idx_source.get_lq45_symbols()
        elif self.config.index == IndexType.ALL:
            # Combine all indices, remove duplicates
            all_symbols = set()
            all_symbols.update(self.idx_source.get_jii70_symbols())
            all_symbols.update(self.idx_source.get_idx30_symbols())
            all_symbols.update(self.idx_source.get_lq45_symbols())
            return list(all_symbols)
        return []

    def _scan_stocks(self, symbols: list[str]) -> list[FactorScores]:
        """Scan stocks and calculate factor scores.

        Args:
            symbols: List of stock symbols to scan

        Returns:
            List of FactorScores for each stock
        """
        scores = []

        for symbol in symbols:
            try:
                # Get stock info (fundamentals)
                info = self.yahoo_source.get_stock_info(symbol)
                if not info:
                    logger.warning(f"No data for {symbol}, skipping")
                    continue

                # Get price history for momentum
                history = self.yahoo_source.get_price_history(symbol, period="6mo")

                # Calculate returns
                price_data = self._calculate_price_metrics(history)

                # Prepare fundamentals
                fundamentals = {
                    "pe_ratio": info.get("pe_ratio"),
                    "pb_ratio": info.get("pb_ratio"),
                    "roe": None,  # Not available directly from Yahoo
                    "debt_to_equity": None,
                    "profit_margin": None,
                }

                # Score the stock
                factor_scores = score_stock(
                    symbol=symbol,
                    fundamentals=fundamentals,
                    price_data=price_data,
                )

                scores.append(factor_scores)

            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")

        # Sort by composite score descending
        scores.sort(key=lambda x: x.composite_score, reverse=True)

        return scores

    def _calculate_price_metrics(self, history) -> dict[str, Any]:
        """Calculate price metrics from history.

        Args:
            history: DataFrame with price history

        Returns:
            Dict with returns, beta, std_dev
        """
        if history.empty:
            return {}

        try:
            closes = history["close"].values

            # Calculate returns
            if len(closes) >= 2:
                returns_6m = ((closes[-1] / closes[0]) - 1) * 100 if len(closes) >= 120 else None
                returns_3m = ((closes[-1] / closes[-60]) - 1) * 100 if len(closes) >= 60 else None
                returns_1m = ((closes[-1] / closes[-20]) - 1) * 100 if len(closes) >= 20 else None
            else:
                returns_6m = returns_3m = returns_1m = None

            # Calculate volatility
            import numpy as np
            daily_returns = np.diff(closes) / closes[:-1] * 100
            std_dev = np.std(daily_returns) * np.sqrt(252) if len(daily_returns) > 0 else None

            return {
                "returns_6m": returns_6m,
                "returns_3m": returns_3m,
                "returns_1m": returns_1m,
                "std_dev": std_dev,
                "beta": None,  # Would need market data to calculate
            }
        except Exception as e:
            logger.warning(f"Error calculating price metrics: {e}")
            return {}

    def _generate_signals(
        self, scores: list[FactorScores]
    ) -> tuple[list[TradeSignal], list[TradeSignal]]:
        """Generate BUY/SELL signals from scores.

        Args:
            scores: List of factor scores

        Returns:
            Tuple of (buy_signals, sell_signals)
        """
        buy_signals = []
        sell_signals = []

        for fs in scores:
            # Get current price
            price_info = self.yahoo_source.get_current_price(fs.symbol)
            if not price_info:
                continue

            current_price = price_info.get("price", 0)
            if current_price <= 0:
                continue

            # Check if already held
            is_held = fs.symbol in self.positions

            # BUY signal: Score > 70, not already held
            if fs.composite_score > self.config.buy_threshold and not is_held:
                signal = TradeSignal(
                    symbol=fs.symbol,
                    action="BUY",
                    score=fs.composite_score,
                    current_price=current_price,
                    lots=0,  # Will be sized later
                    shares=0,
                    position_value=0,
                    stop_loss=None,
                    target=None,
                    reason=f"Score {fs.composite_score:.0f} > {self.config.buy_threshold}",
                    factor_scores=fs,
                )
                buy_signals.append(signal)

            # SELL signal: Score < 50 for existing holdings
            elif fs.composite_score < self.config.sell_threshold and is_held:
                position = self.positions[fs.symbol]
                signal = TradeSignal(
                    symbol=fs.symbol,
                    action="SELL",
                    score=fs.composite_score,
                    current_price=current_price,
                    lots=position.get("lots", 0),
                    shares=position.get("shares", 0),
                    position_value=position.get("shares", 0) * current_price,
                    stop_loss=None,
                    target=None,
                    reason=f"Score {fs.composite_score:.0f} < {self.config.sell_threshold}",
                    factor_scores=fs,
                )
                sell_signals.append(signal)

        # Also check stop-loss and targets for existing positions
        for symbol, pos in self.positions.items():
            price_info = self.yahoo_source.get_current_price(symbol)
            if not price_info:
                continue

            current_price = price_info.get("price", 0)
            stop_loss = pos.get("stop_loss")
            target = pos.get("target")

            # Stop-loss hit
            if stop_loss and current_price <= stop_loss:
                signal = TradeSignal(
                    symbol=symbol,
                    action="SELL",
                    score=0,
                    current_price=current_price,
                    lots=pos.get("lots", 0),
                    shares=pos.get("shares", 0),
                    position_value=pos.get("shares", 0) * current_price,
                    stop_loss=stop_loss,
                    target=target,
                    reason="STOP-LOSS HIT",
                )
                sell_signals.append(signal)

            # Target reached
            elif target and current_price >= target:
                signal = TradeSignal(
                    symbol=symbol,
                    action="SELL",
                    score=0,
                    current_price=current_price,
                    lots=pos.get("lots", 0),
                    shares=pos.get("shares", 0),
                    position_value=pos.get("shares", 0) * current_price,
                    stop_loss=stop_loss,
                    target=target,
                    reason="TARGET REACHED",
                )
                sell_signals.append(signal)

        return buy_signals, sell_signals

    def _size_positions(self, buy_signals: list[TradeSignal]) -> list[TradeSignal]:
        """Calculate position sizes for buy signals.

        Args:
            buy_signals: List of buy signals to size

        Returns:
            List of signals with position sizes
        """
        sized_signals = []
        available_capital = self.cash
        current_positions = len(self.positions)

        for signal in buy_signals:
            # Check position limit
            if current_positions + len(sized_signals) >= self.config.max_positions:
                break

            # Calculate ATR-based stop-loss
            history = self.yahoo_source.get_price_history(signal.symbol, period="1mo")
            atr = self._calculate_atr(history)

            if atr and atr > 0:
                stop_loss = signal.current_price - (self.config.atr_multiplier * atr)
            else:
                # Fallback to 8% stop-loss
                stop_loss = signal.current_price * 0.92

            # Calculate target (1.5x the stop distance)
            stop_distance = signal.current_price - stop_loss
            target = signal.current_price + (stop_distance * 1.5)

            try:
                # Calculate position size
                pos_size = calculate_position_size(
                    capital=available_capital,
                    entry_price=signal.current_price,
                    stop_loss_price=stop_loss,
                    target_price=target,
                    symbol=signal.symbol,
                    max_risk_percent=self.config.max_risk_percent,
                    max_position_percent=self.config.max_position_percent,
                )

                # Check if we have enough capital
                if pos_size.total_cost > available_capital:
                    continue

                # Update signal with sizing
                signal.lots = pos_size.lots
                signal.shares = pos_size.shares
                signal.position_value = pos_size.position_value
                signal.stop_loss = stop_loss
                signal.target = target

                sized_signals.append(signal)
                available_capital -= pos_size.total_cost

            except ValueError as e:
                logger.warning(f"Position sizing error for {signal.symbol}: {e}")

        return sized_signals

    def _calculate_atr(self, history, period: int = 14) -> float | None:
        """Calculate Average True Range.

        Args:
            history: Price history DataFrame
            period: ATR period

        Returns:
            ATR value or None
        """
        if history.empty or len(history) < period + 1:
            return None

        try:
            import numpy as np

            high = history["high"].values
            low = history["low"].values
            close = history["close"].values

            tr1 = high[1:] - low[1:]
            tr2 = np.abs(high[1:] - close[:-1])
            tr3 = np.abs(low[1:] - close[:-1])

            tr = np.maximum(np.maximum(tr1, tr2), tr3)

            if len(tr) >= period:
                return np.mean(tr[-period:])
            return np.mean(tr)
        except Exception as e:
            logger.warning(f"ATR calculation error: {e}")
            return None

    def _execute_sells(self, sell_signals: list[TradeSignal]) -> list[TradeSignal]:
        """Execute sell orders (paper trading).

        Args:
            sell_signals: List of sell signals

        Returns:
            List of executed sells
        """
        executed = []

        for signal in sell_signals:
            if signal.symbol in self.positions:
                position = self.positions[signal.symbol]
                proceeds = signal.shares * signal.current_price

                # Remove position and add to cash
                del self.positions[signal.symbol]
                self.cash += proceeds

                executed.append(signal)
                logger.info(f"SOLD {signal.lots} lots of {signal.symbol} @ Rp {signal.current_price:,.0f}")

        return executed

    def _execute_buys(self, buy_signals: list[TradeSignal]) -> list[TradeSignal]:
        """Execute buy orders (paper trading).

        Args:
            buy_signals: List of buy signals with sizing

        Returns:
            List of executed buys
        """
        executed = []

        for signal in buy_signals:
            if signal.lots <= 0:
                continue

            cost = signal.shares * signal.current_price

            if cost > self.cash:
                continue

            # Create position
            self.positions[signal.symbol] = {
                "lots": signal.lots,
                "shares": signal.shares,
                "avg_price": signal.current_price,
                "stop_loss": signal.stop_loss,
                "target": signal.target,
                "entry_date": datetime.now(TIMEZONE),
            }

            self.cash -= cost
            executed.append(signal)
            logger.info(f"BOUGHT {signal.lots} lots of {signal.symbol} @ Rp {signal.current_price:,.0f}")

        return executed

    def _calculate_portfolio_value(self) -> float:
        """Calculate total portfolio value."""
        value = self.cash

        for symbol, pos in self.positions.items():
            price_info = self.yahoo_source.get_current_price(symbol)
            if price_info:
                value += pos["shares"] * price_info.get("price", pos["avg_price"])
            else:
                value += pos["shares"] * pos["avg_price"]

        return value

    def _check_alerts(self) -> list[str]:
        """Generate alerts for positions requiring attention."""
        alerts = []

        for symbol, pos in self.positions.items():
            price_info = self.yahoo_source.get_current_price(symbol)
            if not price_info:
                continue

            current_price = price_info.get("price", 0)
            stop_loss = pos.get("stop_loss")

            if stop_loss and current_price > 0:
                distance_pct = ((current_price - stop_loss) / current_price) * 100

                if distance_pct < 3:
                    alerts.append(f"{symbol}: {distance_pct:.1f}% above stop-loss (monitor closely)")

        return alerts


def format_autopilot_result(result: AutopilotResult, verbose: bool = False) -> str:
    """Format autopilot result for CLI display.

    Args:
        result: AutopilotResult to format
        verbose: Show detailed AI analysis

    Returns:
        Formatted string
    """
    ai_status = "AI: ON" if result.ai_enabled else "AI: OFF"
    lines = [
        "=" * 60,
        f"AUTOPILOT RUN - {result.run_date.strftime('%A, %d %B %Y')}",
        f"   Index: {result.index_scanned} | Capital: Rp {result.capital:,.0f} | {ai_status}",
        "=" * 60,
        "",
        f"SCANNED: {result.stocks_scanned} stocks | SIGNALS: {len(result.buy_signals)} BUY, {len(result.sell_signals)} SELL",
    ]

    # AI Validation summary (if enabled)
    if result.ai_enabled:
        approved_buys = len(result.ai_approved_buys)
        rejected_buys = len(result.ai_rejected_buys)
        approved_sells = len(result.ai_approved_sells)
        rejected_sells = len(result.ai_rejected_sells)

        lines.extend([
            "",
            "AI VALIDATION:",
            f"   BUY: {approved_buys} approved, {rejected_buys} rejected",
            f"   SELL: {approved_sells} confirmed, {rejected_sells} rejected",
        ])

        # Show BUY signal details
        if result.ai_approved_buys:
            lines.extend(["", "BUY SIGNALS (AI APPROVED):"])
            for trade in result.ai_approved_buys:
                ai_info = f"AI: {trade.ai_score:.1f}" if trade.ai_score else ""
                reasons = ", ".join(trade.ai_key_reasons[:2]) if trade.ai_key_reasons else ""
                lines.append(f"   {trade.symbol}: Score {trade.score:.0f} → {ai_info} ✓ ({reasons})")

        if result.ai_rejected_buys:
            lines.extend(["", "BUY SIGNALS (AI REJECTED):"])
            for trade in result.ai_rejected_buys:
                ai_info = f"AI: {trade.ai_score:.1f}" if trade.ai_score else ""
                reason = trade.ai_rejection_reason or "Unknown"
                lines.append(f"   {trade.symbol}: Score {trade.score:.0f} → {ai_info} ✗ ({reason})")

        # Show SELL signal details
        if result.ai_approved_sells:
            lines.extend(["", "SELL SIGNALS (AI CONFIRMED):"])
            for trade in result.ai_approved_sells:
                ai_info = f"AI: {trade.ai_score:.1f}" if trade.ai_score else ""
                lines.append(f"   {trade.symbol}: Score {trade.score:.0f} → {ai_info} ✓")

        if result.ai_rejected_sells:
            lines.extend(["", "SELL SIGNALS (AI REJECTED):"])
            for trade in result.ai_rejected_sells:
                ai_info = f"AI: {trade.ai_score:.1f}" if trade.ai_score else ""
                reason = trade.ai_rejection_reason or "Unknown"
                lines.append(f"   {trade.symbol}: Score {trade.score:.0f} → {ai_info} ✗ ({reason})")
    else:
        # AI disabled - show signals directly
        if result.buy_signals:
            lines.extend(["", "BUY SIGNALS:"])
            for trade in result.buy_signals:
                lines.append(
                    f"   {trade.symbol}: Score {trade.score:.0f} @ Rp {trade.current_price:,.0f} "
                    f"| SL: Rp {trade.stop_loss:,.0f} | Target: Rp {trade.target:,.0f}"
                )

        if result.sell_signals:
            lines.extend(["", "SELL SIGNALS:"])
            for trade in result.sell_signals:
                lines.append(f"   {trade.symbol}: Score {trade.score:.0f} @ Rp {trade.current_price:,.0f} ({trade.reason})")

    # Sell executions
    if result.executed_sells:
        lines.extend(["", "SELL EXECUTED:"])
        for trade in result.executed_sells:
            ai_info = f" [AI: {trade.ai_score:.1f}]" if trade.ai_validated and trade.ai_score else ""
            lines.append(f"   {trade.symbol}: {trade.lots} lots @ Rp {trade.current_price:,.0f} ({trade.reason}){ai_info}")

    # Buy executions
    if result.executed_buys:
        lines.extend(["", "BUY EXECUTED:"])
        for trade in result.executed_buys:
            ai_info = f" [AI: {trade.ai_score:.1f}]" if trade.ai_validated and trade.ai_score else ""
            lines.append(f"   {trade.symbol}: {trade.lots} lots @ Rp {trade.current_price:,.0f} (Score: {trade.score:.0f}){ai_info}")

            # Verbose mode: show AI insights
            if verbose and trade.ai_key_reasons:
                for reason in trade.ai_key_reasons[:3]:
                    lines.append(f"      • {reason}")

    # If no trades
    if not result.executed_sells and not result.executed_buys:
        lines.extend(["", "No trades executed"])

    # Portfolio summary
    lines.extend([
        "",
        "PORTFOLIO SUMMARY:",
        f"   Positions: {result.positions_count} | Value: Rp {result.portfolio_value:,.0f} | Cash: Rp {result.cash_remaining:,.0f}",
    ])

    # Alerts
    if result.alerts:
        lines.extend(["", "ALERTS:"])
        for alert in result.alerts:
            lines.append(f"   {alert}")

    # Errors
    if result.errors:
        lines.extend(["", "ERRORS:"])
        for error in result.errors:
            lines.append(f"   {error}")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


def get_autopilot_history(days: int = 30) -> list[dict]:
    """Get autopilot run history from database.

    Args:
        days: Number of days of history to retrieve

    Returns:
        List of run records with trades
    """
    from datetime import timedelta

    try:
        session = get_session()
        cutoff_date = datetime.now(TIMEZONE) - timedelta(days=days)

        runs = (
            session.query(AutopilotRun)
            .filter(AutopilotRun.run_date >= cutoff_date)
            .order_by(AutopilotRun.run_date.desc())
            .all()
        )

        history = []
        for run in runs:
            trades = [
                {
                    "symbol": t.symbol,
                    "action": t.action,
                    "lots": t.lots,
                    "price": float(t.price),
                    "total_value": float(t.total_value),
                    "score": t.score,
                    "reason": t.reason,
                }
                for t in run.trades
            ]

            history.append({
                "id": run.id,
                "run_date": run.run_date,
                "index_scanned": run.index_scanned,
                "stocks_scanned": run.stocks_scanned,
                "signals_generated": run.signals_generated,
                "trades_executed": run.trades_executed,
                "initial_capital": float(run.initial_capital) if run.initial_capital else 0,
                "final_value": float(run.final_value) if run.final_value else 0,
                "is_dry_run": run.is_dry_run,
                "trades": trades,
            })

        return history

    except Exception as e:
        logger.error(f"Error fetching autopilot history: {e}")
        return []


def format_autopilot_history(history: list[dict]) -> str:
    """Format autopilot history for CLI display.

    Args:
        history: List of run records

    Returns:
        Formatted string
    """
    if not history:
        return "No autopilot history found."

    lines = [
        "=" * 60,
        "AUTOPILOT HISTORY",
        "=" * 60,
    ]

    for run in history:
        run_date = run["run_date"]
        if hasattr(run_date, "strftime"):
            date_str = run_date.strftime("%d %b %Y %H:%M")
        else:
            date_str = str(run_date)

        mode = "[DRY RUN]" if run["is_dry_run"] else ""
        lines.extend([
            "",
            f"Run #{run['id']} - {date_str} {mode}",
            f"   Index: {run['index_scanned']} | Scanned: {run['stocks_scanned']} | Trades: {run['trades_executed']}",
        ])

        if run["trades"]:
            for trade in run["trades"]:
                lines.append(
                    f"   {trade['action']:4} {trade['symbol']:<8} {trade['lots']:>4} lots @ Rp {trade['price']:>10,.0f}"
                )

    lines.extend(["", "=" * 60])

    return "\n".join(lines)
