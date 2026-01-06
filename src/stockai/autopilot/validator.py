"""AI Validator for Autopilot Trading System.

Validates autopilot BUY/SELL signals using the 7-agent AI orchestrator.
AI agents have veto power over quantitative signals.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from stockai.agents.orchestrator import TradingOrchestrator

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of AI validation for a trade signal."""

    symbol: str
    signal_type: str  # "BUY" or "SELL"
    autopilot_score: float  # Original multi-factor score
    ai_composite_score: float  # 7-agent weighted score
    is_approved: bool  # Passed threshold?
    recommendation: str  # STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL

    # Agent breakdown scores (0-10 scale)
    fundamental_score: float | None = None
    technical_score: float | None = None
    sentiment_score: float | None = None
    portfolio_fit_score: float | None = None
    risk_score: float | None = None

    # Insights for display
    key_reasons: list[str] = field(default_factory=list)
    risk_factors: list[str] = field(default_factory=list)

    # AI-suggested levels
    entry_price: float | None = None
    stop_loss: float | None = None
    target_price: float | None = None

    # Metadata
    rejection_reason: str | None = None
    validated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    validation_time_ms: float = 0.0


@dataclass
class AIValidatorConfig:
    """Configuration for AI validator."""

    # Approval thresholds
    buy_threshold: float = 6.0  # AI score >= 6.0 for BUY approval
    sell_threshold: float = 5.0  # AI score <= 5.0 for SELL confirmation

    # Concurrency
    max_concurrent: int = 3  # Parallel validation limit

    # Timeouts
    per_stock_timeout: float = 120.0  # Seconds per validation
    max_retries: int = 1
    retry_delay: float = 2.0


class AIValidator:
    """Validates autopilot signals using the 7-agent AI orchestrator.

    Provides veto power over quantitative signals by requiring AI approval.

    Attributes:
        config: Validation configuration
        orchestrator: Trading orchestrator for analysis
    """

    def __init__(
        self,
        config: AIValidatorConfig | None = None,
        orchestrator: TradingOrchestrator | None = None,
    ):
        """Initialize the AI validator.

        Args:
            config: Validation configuration (optional)
            orchestrator: Pre-initialized orchestrator (optional)
        """
        self.config = config or AIValidatorConfig()
        self._orchestrator = orchestrator
        self._semaphore: asyncio.Semaphore | None = None

    @property
    def orchestrator(self) -> TradingOrchestrator:
        """Lazy-load orchestrator to avoid LLM initialization on import."""
        if self._orchestrator is None:
            self._orchestrator = TradingOrchestrator()
        return self._orchestrator

    async def validate_signal(
        self,
        symbol: str,
        signal_type: str,
        autopilot_score: float,
    ) -> ValidationResult:
        """Validate a single trade signal with AI analysis.

        Args:
            symbol: Stock symbol (e.g., "TLKM")
            signal_type: "BUY" or "SELL"
            autopilot_score: Original multi-factor score (0-100)

        Returns:
            ValidationResult with approval decision
        """
        start_time = datetime.utcnow()

        try:
            # Build query for AI analysis
            query = f"Analyze {symbol} for {signal_type} signal validation"

            # Run AI analysis with timeout
            result = await asyncio.wait_for(
                self._run_analysis(symbol, query),
                timeout=self.config.per_stock_timeout,
            )

            validation_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            # Extract scores from result
            ai_score = result.get("composite_score") or 5.0
            recommendation = result.get("recommendation") or "HOLD"

            # Extract individual agent scores
            fundamental = self._extract_agent_score(result, "fundamental_analysis")
            technical = self._extract_agent_score(result, "technical_analysis")
            sentiment = self._extract_agent_score(result, "sentiment_analysis")
            portfolio = self._extract_agent_score(result, "portfolio_recommendation")
            risk = self._extract_agent_score(result, "risk_assessment")

            # Determine approval based on signal type and threshold
            is_approved, rejection_reason = self._check_approval(
                signal_type, ai_score, recommendation
            )

            # Extract insights from analysis
            key_reasons = self._extract_key_reasons(result)
            risk_factors = self._extract_risk_factors(result)

            return ValidationResult(
                symbol=symbol,
                signal_type=signal_type,
                autopilot_score=autopilot_score,
                ai_composite_score=ai_score,
                is_approved=is_approved,
                recommendation=recommendation,
                fundamental_score=fundamental,
                technical_score=technical,
                sentiment_score=sentiment,
                portfolio_fit_score=portfolio,
                risk_score=risk,
                key_reasons=key_reasons,
                risk_factors=risk_factors,
                rejection_reason=rejection_reason,
                validation_time_ms=validation_time,
            )

        except asyncio.TimeoutError:
            logger.warning(f"AI validation timed out for {symbol}")
            return self._create_unavailable_result(
                symbol, signal_type, autopilot_score, "Validation timeout"
            )
        except Exception as e:
            logger.error(f"AI validation failed for {symbol}: {e}")
            return self._create_unavailable_result(
                symbol, signal_type, autopilot_score, str(e)
            )

    async def validate_signals_batch(
        self,
        signals: list[tuple[str, str, float]],
    ) -> list[ValidationResult]:
        """Validate multiple signals with concurrency limit.

        Args:
            signals: List of (symbol, signal_type, autopilot_score) tuples

        Returns:
            List of ValidationResult for each signal
        """
        if not signals:
            return []

        # Create semaphore for concurrency control
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent)

        async def validate_with_semaphore(
            symbol: str, signal_type: str, score: float
        ) -> ValidationResult:
            async with self._semaphore:
                return await self.validate_signal(symbol, signal_type, score)

        # Run validations in parallel with limit
        tasks = [
            validate_with_semaphore(symbol, signal_type, score)
            for symbol, signal_type, score in signals
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to unavailable results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                symbol, signal_type, score = signals[i]
                final_results.append(
                    self._create_unavailable_result(
                        symbol, signal_type, score, str(result)
                    )
                )
            else:
                final_results.append(result)

        return final_results

    async def _run_analysis(self, symbol: str, query: str) -> dict[str, Any]:
        """Run AI analysis through orchestrator.

        Args:
            symbol: Stock symbol
            query: Analysis query

        Returns:
            Analysis result dict
        """
        # Use sync run() wrapped in executor for now
        # TODO: Use async arun() when fully implemented
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: self.orchestrator.run(query, symbol)
        )
        return result

    def _check_approval(
        self,
        signal_type: str,
        ai_score: float,
        recommendation: str,
    ) -> tuple[bool, str | None]:
        """Check if signal meets approval threshold.

        Args:
            signal_type: "BUY" or "SELL"
            ai_score: AI composite score (0-10)
            recommendation: AI recommendation string

        Returns:
            Tuple of (is_approved, rejection_reason)
        """
        if signal_type == "BUY":
            # BUY requires AI score >= threshold
            if ai_score >= self.config.buy_threshold:
                return True, None
            else:
                return False, f"AI score {ai_score:.1f} below BUY threshold {self.config.buy_threshold}"

        elif signal_type == "SELL":
            # SELL requires AI score <= threshold (confirms weakness)
            if ai_score <= self.config.sell_threshold:
                return True, None
            else:
                return False, f"AI score {ai_score:.1f} above SELL threshold {self.config.sell_threshold}"

        return False, f"Unknown signal type: {signal_type}"

    def _extract_agent_score(
        self, result: dict[str, Any], agent_key: str
    ) -> float | None:
        """Extract score from a specific agent result."""
        agent_result = result.get(agent_key)
        if agent_result and isinstance(agent_result, dict):
            return agent_result.get("score")
        return None

    def _extract_key_reasons(self, result: dict[str, Any]) -> list[str]:
        """Extract key reasons from AI analysis."""
        reasons = []

        # Get recommendation context
        answer = result.get("answer", "")
        recommendation = result.get("recommendation", "")

        # Simple extraction - look for key indicators
        if recommendation in ("STRONG BUY", "BUY"):
            if "momentum" in answer.lower():
                reasons.append("Strong momentum indicators")
            if "fundamental" in answer.lower():
                reasons.append("Solid fundamentals")
            if "sentiment" in answer.lower() and "positive" in answer.lower():
                reasons.append("Positive market sentiment")
        elif recommendation in ("SELL", "STRONG SELL"):
            if "resistance" in answer.lower():
                reasons.append("Near resistance level")
            if "overvalued" in answer.lower():
                reasons.append("Potentially overvalued")
            if "weakness" in answer.lower():
                reasons.append("Technical weakness detected")

        return reasons[:3]  # Top 3 reasons

    def _extract_risk_factors(self, result: dict[str, Any]) -> list[str]:
        """Extract risk factors from AI analysis."""
        risks = []

        answer = result.get("answer", "")

        # Common risk patterns
        if "volatility" in answer.lower():
            risks.append("High volatility")
        if "resistance" in answer.lower():
            risks.append("Near resistance level")
        if "support" in answer.lower():
            risks.append("Approaching support level")
        if "volume" in answer.lower() and "low" in answer.lower():
            risks.append("Low trading volume")

        return risks[:3]  # Top 3 risks

    def _create_unavailable_result(
        self,
        symbol: str,
        signal_type: str,
        autopilot_score: float,
        error_message: str,
    ) -> ValidationResult:
        """Create result for unavailable/failed validation."""
        return ValidationResult(
            symbol=symbol,
            signal_type=signal_type,
            autopilot_score=autopilot_score,
            ai_composite_score=5.0,  # Neutral
            is_approved=False,
            recommendation="AI_UNAVAILABLE",
            rejection_reason=f"AI unavailable: {error_message}",
        )


def create_validator(
    buy_threshold: float = 6.0,
    sell_threshold: float = 5.0,
    max_concurrent: int = 3,
) -> AIValidator:
    """Create an AI validator with custom configuration.

    Args:
        buy_threshold: Minimum AI score for BUY approval
        sell_threshold: Maximum AI score for SELL confirmation
        max_concurrent: Parallel validation limit

    Returns:
        Configured AIValidator
    """
    config = AIValidatorConfig(
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        max_concurrent=max_concurrent,
    )
    return AIValidator(config=config)
