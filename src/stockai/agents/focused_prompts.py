"""Focused Agent Prompts for Efficient Validation.

Simplified prompts for 3-agent validation pipeline.
Each agent has a specific, narrow focus and returns a simple APPROVE/REJECT decision.
"""

import re

# =============================================================================
# Technical Analyst - Validates Entry Point Quality
# =============================================================================

TECHNICAL_ANALYST_PROMPT = """You are validating a potential BUY signal for {ticker}.

PRE-COMPUTED DATA:
- Technical Score: {tech_score:.1f}/100
- RSI (14-day): {rsi:.1f}
- MACD Signal: {macd_signal}
- ADX: {adx:.1f} ({trend_strength})
- Price vs SMA20: {pct_above_sma20:+.1f}%
- Distance to Support: {support_distance:.1f}%
- Distance to Resistance: {resistance_distance:.1f}%
- Current Price: Rp {current_price:,.0f}

YOUR TASK: Validate this is a good ENTRY POINT.

CHECK THESE CRITERIA:
1. Is price near support (within 5%)? - Currently {support_distance:.1f}%
2. Is trend favorable (price above or near SMA20)?
3. Is momentum positive (MACD bullish or neutral)?
4. Is there room to resistance (at least 5% upside)?
5. Is RSI not overbought (< 70)?

IMPORTANT: Be decisive. Moderate conditions are acceptable if other factors are strong.

OUTPUT (exactly this format):
DECISION: APPROVE or REJECT
REASON: One sentence explaining the key factor"""


# =============================================================================
# Fundamental Analyst - Validates Financial Health
# =============================================================================

FUNDAMENTAL_ANALYST_PROMPT = """You are validating a potential BUY signal for {ticker}.

PRE-COMPUTED DATA:
- Fundamental Score: {fund_score:.1f}/100
- P/E Ratio: {pe_ratio}
- P/B Ratio: {pb_ratio}
- ROE: {roe}%
- Debt-to-Equity: {debt_to_equity}
- Profit Margin: {profit_margin}%
- Sector: {sector}

YOUR TASK: Validate the company is financially healthy for investment.

CHECK THESE CRITERIA:
1. Is P/E reasonable (< 25 for growth, < 15 for value)?
2. Is P/B not extremely high (< 5)?
3. Is ROE positive and reasonable (> 10% preferred)?
4. Is debt manageable (D/E < 1.5 for most sectors)?
5. Is the company profitable (positive margin)?

IMPORTANT: Missing data is acceptable if available data looks strong.
Indonesian banking stocks may have higher D/E ratios - this is normal.

OUTPUT (exactly this format):
DECISION: APPROVE or REJECT
REASON: One sentence explaining the key factor"""


# =============================================================================
# Risk Manager - Validates Risk/Reward Profile
# =============================================================================

RISK_MANAGER_PROMPT = """You are validating a potential BUY signal for {ticker}.

TRADE PLAN:
- Entry Range: Rp {entry_low:,.0f} - Rp {entry_high:,.0f}
- Stop Loss: Rp {stop_loss:,.0f} ({stop_loss_pct:.1f}% risk)
- Take Profit 1: Rp {tp1:,.0f} ({tp1_pct:.1f}% gain)
- Take Profit 2: Rp {tp2:,.0f} ({tp2_pct:.1f}% gain)
- Take Profit 3: Rp {tp3:,.0f} ({tp3_pct:.1f}% gain)
- Risk/Reward Ratio: {rr_ratio:.2f}

POSITION SIZING (2% risk rule):
- Recommended Lots: {lots}
- Position Value: Rp {position_value:,.0f}
- Max Loss: Rp {max_loss:,.0f}

MARKET CONTEXT:
- Smart Money Score: {smart_money:.1f} ({smart_money_interpretation})
- ADX Trend Strength: {adx:.1f} ({trend_strength})
- Gates Passed: {gates_passed}/6

YOUR TASK: Validate the trade has acceptable risk/reward.

CHECK THESE CRITERIA:
1. Is Risk/Reward ratio at least 1.5:1?
2. Is position size reasonable (not over-leveraged)?
3. Is Smart Money showing accumulation (score >= 2)?
4. Is stop loss logically placed (near support)?
5. Are take profit targets achievable?

IMPORTANT: A good risk/reward ratio can compensate for moderate scores elsewhere.

OUTPUT (exactly this format):
DECISION: APPROVE or REJECT
REASON: One sentence explaining the key factor"""


# =============================================================================
# Response Parsing Helper
# =============================================================================

def parse_agent_response(response: str) -> tuple[str, str]:
    """Parse agent response to extract decision and reason.

    Args:
        response: Raw response from the agent

    Returns:
        Tuple of (decision, reason) where decision is "APPROVE" or "REJECT"

    Raises:
        ValueError: If response cannot be parsed
    """
    # Normalize response
    response = response.strip()

    # Try to find DECISION line
    decision_match = re.search(
        r"DECISION:\s*(APPROVE|REJECT)",
        response,
        re.IGNORECASE
    )

    if not decision_match:
        # Fallback: look for the words anywhere
        if "APPROVE" in response.upper():
            decision = "APPROVE"
        elif "REJECT" in response.upper():
            decision = "REJECT"
        else:
            raise ValueError(f"Could not parse decision from response: {response[:100]}")
    else:
        decision = decision_match.group(1).upper()

    # Try to find REASON line
    reason_match = re.search(
        r"REASON:\s*(.+?)(?:\n|$)",
        response,
        re.IGNORECASE
    )

    if reason_match:
        reason = reason_match.group(1).strip()
    else:
        # Fallback: use the last non-empty line
        lines = [l.strip() for l in response.split("\n") if l.strip()]
        reason = lines[-1] if lines else "No reason provided"

    return decision, reason


# =============================================================================
# Prompt Formatters
# =============================================================================

def format_technical_prompt(
    ticker: str,
    tech_score: float,
    rsi: float,
    macd_signal: str,
    adx: float,
    trend_strength: str,
    pct_above_sma20: float,
    support_distance: float,
    resistance_distance: float,
    current_price: float,
) -> str:
    """Format the technical analyst prompt with data."""
    return TECHNICAL_ANALYST_PROMPT.format(
        ticker=ticker,
        tech_score=tech_score,
        rsi=rsi,
        macd_signal=macd_signal,
        adx=adx,
        trend_strength=trend_strength,
        pct_above_sma20=pct_above_sma20,
        support_distance=support_distance,
        resistance_distance=resistance_distance,
        current_price=current_price,
    )


def format_fundamental_prompt(
    ticker: str,
    fund_score: float,
    pe_ratio: float | None,
    pb_ratio: float | None,
    roe: float | None,
    debt_to_equity: float | None,
    profit_margin: float | None,
    sector: str,
) -> str:
    """Format the fundamental analyst prompt with data."""
    return FUNDAMENTAL_ANALYST_PROMPT.format(
        ticker=ticker,
        fund_score=fund_score,
        pe_ratio=f"{pe_ratio:.1f}" if pe_ratio else "N/A",
        pb_ratio=f"{pb_ratio:.1f}" if pb_ratio else "N/A",
        roe=f"{roe:.1f}" if roe else "N/A",
        debt_to_equity=f"{debt_to_equity:.2f}" if debt_to_equity else "N/A",
        profit_margin=f"{profit_margin:.1f}" if profit_margin else "N/A",
        sector=sector or "Unknown",
    )


def format_risk_manager_prompt(
    ticker: str,
    entry_low: float,
    entry_high: float,
    stop_loss: float,
    stop_loss_pct: float,
    tp1: float,
    tp1_pct: float,
    tp2: float,
    tp2_pct: float,
    tp3: float,
    tp3_pct: float,
    rr_ratio: float,
    lots: int,
    position_value: float,
    max_loss: float,
    smart_money: float,
    smart_money_interpretation: str,
    adx: float,
    trend_strength: str,
    gates_passed: int,
) -> str:
    """Format the risk manager prompt with data."""
    return RISK_MANAGER_PROMPT.format(
        ticker=ticker,
        entry_low=entry_low,
        entry_high=entry_high,
        stop_loss=stop_loss,
        stop_loss_pct=stop_loss_pct,
        tp1=tp1,
        tp1_pct=tp1_pct,
        tp2=tp2,
        tp2_pct=tp2_pct,
        tp3=tp3,
        tp3_pct=tp3_pct,
        rr_ratio=rr_ratio,
        lots=lots,
        position_value=position_value,
        max_loss=max_loss,
        smart_money=smart_money,
        smart_money_interpretation=smart_money_interpretation,
        adx=adx,
        trend_strength=trend_strength,
        gates_passed=gates_passed,
    )
