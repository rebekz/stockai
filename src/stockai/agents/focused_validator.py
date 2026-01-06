"""Focused 3-Agent Validator for Trade Signals.

Implements a streamlined validation pipeline with 3 specialized agents:
1. Technical Analyst - Validates entry point quality
2. Fundamental Analyst - Validates financial health
3. Risk Manager - Validates risk/reward profile

Each agent receives pre-computed data and returns a simple APPROVE/REJECT decision.
The pipeline short-circuits on first rejection for efficiency.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from stockai.agents.focused_prompts import (
    format_fundamental_prompt,
    format_risk_manager_prompt,
    format_technical_prompt,
    parse_agent_response,
)
from stockai.config import get_settings
from stockai.scoring.analyzer import AnalysisResult
from stockai.scoring.trade_plan import calculate_position_with_plan

logger = logging.getLogger(__name__)


# =============================================================================
# Result Dataclasses
# =============================================================================


@dataclass
class AgentDecision:
    """Decision from a single focused agent."""

    agent_name: str
    decision: str  # APPROVE or REJECT
    reason: str
    executed_at: str = ""


@dataclass
class FocusedValidationResult:
    """Result from the 3-agent focused validation pipeline."""

    approved: bool
    rejected_by: str | None = None  # technical, fundamental, risk
    rejection_reason: str | None = None

    # Individual agent results (if not short-circuited)
    technical_decision: AgentDecision | None = None
    fundamental_decision: AgentDecision | None = None
    risk_decision: AgentDecision | None = None

    # Composite reasons (if approved)
    approval_reasons: list[str] = field(default_factory=list)


# =============================================================================
# Focused Validator Class
# =============================================================================


class FocusedValidator:
    """3-agent validation pipeline for trade signals.

    Uses pre-computed analysis data to quickly validate signals with:
    1. Technical Analyst - Entry point validation
    2. Fundamental Analyst - Financial health check
    3. Risk Manager - Risk/reward assessment

    Short-circuits on first rejection for efficiency.
    """

    def __init__(
        self,
        model_name: str = "gemini-1.5-flash",
        timeout: float = 30.0,
    ):
        """Initialize the focused validator.

        Args:
            model_name: Google AI model to use (default: gemini-1.5-flash for speed)
            timeout: Timeout for each agent in seconds (default: 30s)
        """
        settings = get_settings()
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=settings.google_api_key,
            temperature=0.1,  # Low temperature for consistent decisions
            convert_system_message_to_human=True,
        )
        self.timeout = timeout

    async def validate(
        self,
        analysis: AnalysisResult,
        fundamentals: dict[str, Any] | None = None,
        capital: float = 10_000_000,
    ) -> FocusedValidationResult:
        """Validate a trade signal using 3 focused agents.

        Args:
            analysis: Complete analysis result from analyze_stock()
            fundamentals: Optional fundamentals dict (pe_ratio, roe, etc.)
            capital: Capital for position sizing (default: 10M IDR)

        Returns:
            FocusedValidationResult with approval status and reasons
        """
        fundamentals = fundamentals or {}

        # Step 1: Technical Analyst
        try:
            tech_decision = await self._run_technical_analyst(analysis)
        except Exception as e:
            logger.warning(f"Technical analyst error: {e}")
            tech_decision = AgentDecision(
                agent_name="technical",
                decision="APPROVE",
                reason="Agent error - defaulting to approve",
            )

        if tech_decision.decision == "REJECT":
            return FocusedValidationResult(
                approved=False,
                rejected_by="technical",
                rejection_reason=tech_decision.reason,
                technical_decision=tech_decision,
            )

        # Step 2: Fundamental Analyst
        try:
            fund_decision = await self._run_fundamental_analyst(
                analysis, fundamentals
            )
        except Exception as e:
            logger.warning(f"Fundamental analyst error: {e}")
            fund_decision = AgentDecision(
                agent_name="fundamental",
                decision="APPROVE",
                reason="Agent error - defaulting to approve",
            )

        if fund_decision.decision == "REJECT":
            return FocusedValidationResult(
                approved=False,
                rejected_by="fundamental",
                rejection_reason=fund_decision.reason,
                technical_decision=tech_decision,
                fundamental_decision=fund_decision,
            )

        # Step 3: Risk Manager
        try:
            risk_decision = await self._run_risk_manager(analysis, capital)
        except Exception as e:
            logger.warning(f"Risk manager error: {e}")
            risk_decision = AgentDecision(
                agent_name="risk",
                decision="APPROVE",
                reason="Agent error - defaulting to approve",
            )

        if risk_decision.decision == "REJECT":
            return FocusedValidationResult(
                approved=False,
                rejected_by="risk",
                rejection_reason=risk_decision.reason,
                technical_decision=tech_decision,
                fundamental_decision=fund_decision,
                risk_decision=risk_decision,
            )

        # All agents approved
        return FocusedValidationResult(
            approved=True,
            technical_decision=tech_decision,
            fundamental_decision=fund_decision,
            risk_decision=risk_decision,
            approval_reasons=[
                tech_decision.reason,
                fund_decision.reason,
                risk_decision.reason,
            ],
        )

    async def _run_technical_analyst(
        self, analysis: AnalysisResult
    ) -> AgentDecision:
        """Run the technical analyst agent."""
        # Extract technical data
        tech_score = (analysis.momentum_score + (100 - analysis.volatility_score)) / 2
        adx = analysis.adx.get("adx", 0)
        trend_strength = analysis.adx.get("trend_strength", "UNKNOWN")

        # Calculate distances
        support_dist = analysis.support_resistance.distance_to_support_pct or 999
        if analysis.support_resistance.nearest_resistance:
            resistance_dist = (
                (analysis.support_resistance.nearest_resistance - analysis.current_price)
                / analysis.current_price
            ) * 100
        else:
            resistance_dist = 999

        # Format prompt
        prompt = format_technical_prompt(
            ticker=analysis.ticker,
            tech_score=tech_score,
            rsi=50.0,  # Default if not available
            macd_signal="NEUTRAL",  # Default if not available
            adx=adx,
            trend_strength=trend_strength,
            pct_above_sma20=0.0,  # Would need additional data
            support_distance=support_dist,
            resistance_distance=resistance_dist,
            current_price=analysis.current_price,
        )

        # Call LLM with timeout
        response = await asyncio.wait_for(
            self._invoke_llm(prompt),
            timeout=self.timeout,
        )

        decision, reason = parse_agent_response(response)
        return AgentDecision(
            agent_name="technical",
            decision=decision,
            reason=reason,
        )

    async def _run_fundamental_analyst(
        self,
        analysis: AnalysisResult,
        fundamentals: dict[str, Any],
    ) -> AgentDecision:
        """Run the fundamental analyst agent."""
        # Calculate fundamental score
        fund_score = (analysis.value_score + analysis.quality_score) / 2

        # Format prompt
        prompt = format_fundamental_prompt(
            ticker=analysis.ticker,
            fund_score=fund_score,
            pe_ratio=fundamentals.get("pe_ratio"),
            pb_ratio=fundamentals.get("pb_ratio"),
            roe=fundamentals.get("roe"),
            debt_to_equity=fundamentals.get("debt_to_equity"),
            profit_margin=fundamentals.get("profit_margin"),
            sector=fundamentals.get("sector", "Unknown"),
        )

        # Call LLM with timeout
        response = await asyncio.wait_for(
            self._invoke_llm(prompt),
            timeout=self.timeout,
        )

        decision, reason = parse_agent_response(response)
        return AgentDecision(
            agent_name="fundamental",
            decision=decision,
            reason=reason,
        )

    async def _run_risk_manager(
        self,
        analysis: AnalysisResult,
        capital: float,
    ) -> AgentDecision:
        """Run the risk manager agent."""
        if not analysis.trade_plan:
            return AgentDecision(
                agent_name="risk",
                decision="REJECT",
                reason="No trade plan generated - gates likely failed",
            )

        plan = analysis.trade_plan

        # Calculate position sizing
        position = calculate_position_with_plan(capital, plan)

        # Calculate percentages
        entry_mid = (plan.entry_low + plan.entry_high) / 2
        stop_loss_pct = ((entry_mid - plan.stop_loss) / entry_mid) * 100
        tp1_pct = ((plan.take_profit_1 - entry_mid) / entry_mid) * 100
        tp2_pct = ((plan.take_profit_2 - entry_mid) / entry_mid) * 100
        tp3_pct = ((plan.take_profit_3 - entry_mid) / entry_mid) * 100

        # Format prompt
        prompt = format_risk_manager_prompt(
            ticker=analysis.ticker,
            entry_low=plan.entry_low,
            entry_high=plan.entry_high,
            stop_loss=plan.stop_loss,
            stop_loss_pct=stop_loss_pct,
            tp1=plan.take_profit_1,
            tp1_pct=tp1_pct,
            tp2=plan.take_profit_2,
            tp2_pct=tp2_pct,
            tp3=plan.take_profit_3,
            tp3_pct=tp3_pct,
            rr_ratio=plan.risk_reward_ratio,
            lots=position.get("lots", 0),
            position_value=position.get("position_value", 0),
            max_loss=position.get("max_loss", 0),
            smart_money=analysis.smart_money.score,
            smart_money_interpretation=analysis.smart_money.interpretation,
            adx=analysis.adx.get("adx", 0),
            trend_strength=analysis.adx.get("trend_strength", "UNKNOWN"),
            gates_passed=analysis.gates.gates_passed,
        )

        # Call LLM with timeout
        response = await asyncio.wait_for(
            self._invoke_llm(prompt),
            timeout=self.timeout,
        )

        decision, reason = parse_agent_response(response)
        return AgentDecision(
            agent_name="risk",
            decision=decision,
            reason=reason,
        )

    async def _invoke_llm(self, prompt: str) -> str:
        """Invoke the LLM with a prompt."""
        messages = [HumanMessage(content=prompt)]
        response = await self.llm.ainvoke(messages)
        return response.content


# =============================================================================
# Convenience Function
# =============================================================================


async def validate_with_focused_agents(
    analysis: AnalysisResult,
    fundamentals: dict[str, Any] | None = None,
    capital: float = 10_000_000,
    model_name: str = "gemini-1.5-flash",
) -> FocusedValidationResult:
    """Convenience function to validate a signal with focused agents.

    Args:
        analysis: Analysis result from analyze_stock()
        fundamentals: Optional fundamentals dict
        capital: Capital for position sizing
        model_name: LLM model to use

    Returns:
        FocusedValidationResult
    """
    validator = FocusedValidator(model_name=model_name)
    return await validator.validate(analysis, fundamentals, capital)
