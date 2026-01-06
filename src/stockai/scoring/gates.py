"""Six-Gate Decision Filter Module.

Implements the gate validation system that all BUY signals must pass through.
Each gate represents a quality threshold that must be met for high-confidence signals.

Gates:
1. Overall Score >= 70
2. Technical Score >= 60
3. Smart Money Score >= 3.0
4. Distance to Support <= 5%
5. ADX Trend Strength >= 20
6. Fundamental Score >= 60
"""

from dataclasses import dataclass, field


@dataclass
class GateConfig:
    """Configuration for gate thresholds."""

    overall_min: float = 70.0
    technical_min: float = 60.0
    smart_money_min: float = 3.0
    near_support_pct: float = 5.0
    adx_min: float = 20.0
    fundamental_min: float = 60.0


@dataclass
class GateResult:
    """Result of gate validation."""

    all_passed: bool
    gates_passed: int
    total_gates: int
    passed_gates: list[str] = field(default_factory=list)
    rejection_reasons: list[str] = field(default_factory=list)
    confidence: str = "REJECTED"  # HIGH, WATCH, REJECTED


def validate_gates(
    stock_data: dict,
    config: GateConfig | None = None,
) -> GateResult:
    """Validate a stock against all 6 gates.

    Args:
        stock_data: Dictionary containing:
            - overall_score: float (0-100)
            - technical_score: float (0-100)
            - smart_money_score: float (-2 to 5)
            - distance_to_support_pct: float | None
            - adx: float (0-100)
            - fundamental_score: float (0-100)
        config: Optional GateConfig with custom thresholds

    Returns:
        GateResult with pass/fail status and details
    """
    if config is None:
        config = GateConfig()

    passed_gates: list[str] = []
    rejection_reasons: list[str] = []
    total_gates = 6

    # Gate 1: Overall Score
    overall = stock_data.get("overall_score", 0)
    if overall >= config.overall_min:
        passed_gates.append(f"Overall Score: {overall:.1f} >= {config.overall_min}")
    else:
        rejection_reasons.append(
            f"Overall Score: {overall:.1f} < {config.overall_min}"
        )

    # Gate 2: Technical Score
    technical = stock_data.get("technical_score", 0)
    if technical >= config.technical_min:
        passed_gates.append(
            f"Technical Score: {technical:.1f} >= {config.technical_min}"
        )
    else:
        rejection_reasons.append(
            f"Technical Score: {technical:.1f} < {config.technical_min}"
        )

    # Gate 3: Smart Money Score
    smart_money = stock_data.get("smart_money_score", 0)
    if smart_money >= config.smart_money_min:
        passed_gates.append(
            f"Smart Money Score: {smart_money:.1f} >= {config.smart_money_min}"
        )
    else:
        rejection_reasons.append(
            f"Smart Money Score: {smart_money:.1f} < {config.smart_money_min}"
        )

    # Gate 4: Near Support
    distance_to_support = stock_data.get("distance_to_support_pct")
    if distance_to_support is not None:
        if distance_to_support <= config.near_support_pct:
            passed_gates.append(
                f"Distance to Support: {distance_to_support:.1f}% <= {config.near_support_pct}%"
            )
        else:
            rejection_reasons.append(
                f"Distance to Support: {distance_to_support:.1f}% > {config.near_support_pct}%"
            )
    else:
        rejection_reasons.append("Distance to Support: No support level found")

    # Gate 5: ADX Trend Strength
    adx = stock_data.get("adx", 0)
    if adx >= config.adx_min:
        passed_gates.append(f"ADX Trend Strength: {adx:.1f} >= {config.adx_min}")
    else:
        rejection_reasons.append(f"ADX Trend Strength: {adx:.1f} < {config.adx_min}")

    # Gate 6: Fundamental Score
    fundamental = stock_data.get("fundamental_score", 0)
    if fundamental >= config.fundamental_min:
        passed_gates.append(
            f"Fundamental Score: {fundamental:.1f} >= {config.fundamental_min}"
        )
    else:
        rejection_reasons.append(
            f"Fundamental Score: {fundamental:.1f} < {config.fundamental_min}"
        )

    # Determine results
    gates_passed = len(passed_gates)
    all_passed = gates_passed == total_gates

    # Determine confidence level
    if all_passed:
        confidence = "HIGH"
    elif len(rejection_reasons) <= 2 and overall >= 60:
        confidence = "WATCH"
    else:
        confidence = "REJECTED"

    return GateResult(
        all_passed=all_passed,
        gates_passed=gates_passed,
        total_gates=total_gates,
        passed_gates=passed_gates,
        rejection_reasons=rejection_reasons,
        confidence=confidence,
    )
