# StockAI Enhancement Design: Quality Over Quantity

**Date:** 2026-01-06
**Status:** Proposed
**Goal:** Improve AI agent accuracy and reduce false positive signals

---

## Executive Summary

This design enhances StockAI with a "Quality Over Quantity" approach inspired by professional trading systems. The key changes are:

1. Add Smart Money Score (institutional flow tracking)
2. Add Support/Resistance detection (entry timing)
3. Add 6-Gate Filter System (strict qualification before AI)
4. Add Trade Plan Generation (actionable SL/TP/position sizing)
5. Reduce AI agents from 7 to 3 focused specialists
6. Add Rejection Reasons to all outputs

**Expected Outcomes:**
- Signal accuracy: +30-40% (fewer false positives)
- AI cost: -80-90% (only run on qualified stocks)
- User clarity: +100% (know exactly why rejected)

---

## 1. Problem Statement

### Current Issues

The 7-agent orchestrator has accuracy problems due to:

1. **Inconsistent Reasoning** - Each agent uses different mental models per run
2. **No Hard Gates** - Agents may approve despite weak technicals if fundamentals look good
3. **Vague Thresholds** - The 6.0/5.0 approval scores are arbitrary, not calibrated
4. **Missing Context** - Agents don't know if price is near support or in no-man's land
5. **High Cost** - All 7 agents run on every stock, even obvious rejects

### Current vs. Proposed Approach

| Aspect | Current | Proposed |
|--------|---------|----------|
| Signal Rate | ~30% of stocks get BUY | 5-10% (quality filter) |
| Decision Logic | Score threshold only | 6-gate filter + AI validation |
| AI Usage | All stocks, all agents | Pre-qualified stocks only, 3 agents |
| Output | Score + signal type | Score + rejection reasons + trade plan |

---

## 2. Scoring System Enhancement

### 2.1 Current Scoring Components

| Component | Weight | What It Measures |
|-----------|--------|------------------|
| Value | 25% | P/E, P/B ratios vs sector |
| Quality | 30% | ROE, debt-to-equity, profit margins |
| Momentum | 25% | 6-month returns with trend confirmation |
| Volatility | 20% | Beta, std dev, max drawdown |

**Gap Identified:** No Smart Money (institutional flow) tracking.

### 2.2 New Component: Smart Money Score

**Purpose:** Track institutional accumulation/distribution patterns.

**Range:** -2.0 to 5.0

**Indicators:**

| Indicator | Weight | Description |
|-----------|--------|-------------|
| Volume-Price Analysis | 40% | Accumulation vs Distribution days |
| On-Balance Volume (OBV) | 25% | OBV trend direction |
| Money Flow Index (MFI) | 20% | Volume-weighted RSI |
| Unusual Volume | 15% | Spike detection on up days |

**Calculation Logic:**

```python
def calculate_smart_money_score(df: pd.DataFrame) -> dict:
    """
    Smart Money Score: -2.0 to 5.0
    Tracks institutional accumulation/distribution
    """
    score = 0.0
    details = {}

    # 1. ACCUMULATION/DISTRIBUTION DAYS (last 20 days)
    recent = df.tail(20)
    acc_days = ((recent['return'] > 0) & (recent['vol_ratio'] > 1.2)).sum()
    dist_days = ((recent['return'] < 0) & (recent['vol_ratio'] > 1.2)).sum()
    net_acc = acc_days - dist_days

    if net_acc >= 5:
        score += 2.5
    elif net_acc >= 3:
        score += 1.5
    elif net_acc >= 1:
        score += 0.5
    elif net_acc >= -3:
        score -= 1.0
    else:
        score -= 2.0

    # 2. ON-BALANCE VOLUME TREND
    obv_sma_5 = df['obv'].tail(5).mean()
    obv_sma_20 = df['obv'].tail(20).mean()

    if obv_sma_5 > obv_sma_20 * 1.05:
        score += 1.0
    elif obv_sma_5 > obv_sma_20:
        score += 0.5
    elif obv_sma_5 < obv_sma_20 * 0.95:
        score -= 0.5

    # 3. MONEY FLOW INDEX
    mfi = ta.mfi(df['high'], df['low'], df['close'], df['volume']).iloc[-1]

    if mfi > 70:
        score += 0.5  # Strong inflow but overbought
    elif mfi > 50:
        score += 1.0
    elif mfi > 30:
        score += 0
    else:
        score -= 0.5

    # 4. UNUSUAL VOLUME SPIKE
    if latest_vol_ratio > 2.0 and latest_return > 0.02:
        score += 0.5
    elif latest_vol_ratio > 2.0 and latest_return < -0.02:
        score -= 0.5

    return {
        'score': max(-2.0, min(5.0, score)),
        'details': details,
        'interpretation': 'ACCUMULATION' if score >= 3.0 else 'NEUTRAL' if score >= 0 else 'DISTRIBUTION'
    }
```

---

## 3. Support/Resistance Detection

### 3.1 Purpose

Filter stocks that are NOT near a support level. Buying far from support = poor risk/reward.

### 3.2 Detection Methods

| Method | Priority | Description |
|--------|----------|-------------|
| Pivot Points | High | Local highs/lows over 5-day window |
| Fibonacci Retracement | Medium | 38.2%, 50%, 61.8% levels |
| Volume Profile | Medium | Price levels with high volume |
| Round Numbers | Low | Psychological levels (1000, 5000, etc.) |

### 3.3 Implementation

```python
def find_support_resistance(df: pd.DataFrame, lookback: int = 60) -> dict:
    """Find support and resistance levels using pivot points."""
    from scipy.signal import argrelextrema

    recent = df.tail(lookback)
    current_price = df['close'].iloc[-1]

    # Find pivot points
    pivot_high_idx = argrelextrema(recent['high'].values, np.greater, order=5)[0]
    pivot_low_idx = argrelextrema(recent['low'].values, np.less, order=5)[0]

    # Extract levels within 20% of current price
    resistances = [h for h in pivot_highs if current_price < h < current_price * 1.2][:3]
    supports = [s for s in pivot_lows if current_price * 0.8 < s < current_price][-3:]

    nearest_support = supports[-1] if supports else None

    # Calculate distance to support
    if nearest_support:
        distance_pct = (current_price - nearest_support) / current_price * 100
        is_near_support = distance_pct <= 5.0
    else:
        is_near_support = False

    return {
        'current_price': current_price,
        'nearest_support': nearest_support,
        'nearest_resistance': resistances[0] if resistances else None,
        'distance_to_support_pct': distance_pct,
        'is_near_support': is_near_support,
        'suggested_stop_loss': nearest_support * 0.97 if nearest_support else current_price * 0.92
    }
```

---

## 4. Gate Filter System

### 4.1 Gate Definitions

All gates must pass for a BUY signal. Any failure = NO TRADE with specific reason.

| Gate | Condition | Threshold | Rejection Reason |
|------|-----------|-----------|------------------|
| 1 | Overall Score | ≥ 70 (on 0-100 scale) | "Low overall score (X < 70)" |
| 2 | Technical Score | ≥ 60 | "Weak technical score (X < 60)" |
| 3 | Smart Money Score | ≥ 3.0 | "No smart money support" |
| 4 | Near Support | ≤ 5% distance | "Not near support level (X% away)" |
| 5 | ADX Trend | ≥ 20 | "Weak trend strength (ADX X < 20)" |
| 6 | Fundamental Score | ≥ 60 | "Weak fundamentals (X < 60)" |

### 4.2 Confidence Levels

| Confidence | Criteria | Action |
|------------|----------|--------|
| **HIGH** | All 6 gates pass | BUY signal with trade plan |
| **WATCH** | 1-2 gates fail, Overall ≥ 60 | Monitor for entry opportunity |
| **REJECTED** | 3+ gates fail OR Overall < 60 | No trade |

### 4.3 Implementation

```python
def validate_gates(stock_data: dict, config: GateConfig) -> tuple[bool, list[str]]:
    """
    Validate all gates for a stock.
    Returns (all_passed, rejection_reasons)
    """
    rejection_reasons = []

    # Gate 1: Overall Score
    if stock_data['overall_score'] < config.overall_min:
        rejection_reasons.append(
            f"Low overall score ({stock_data['overall_score']} < {config.overall_min})"
        )

    # Gate 2: Technical Score
    if stock_data['technical_score'] < config.technical_min:
        rejection_reasons.append(
            f"Weak technical score ({stock_data['technical_score']} < {config.technical_min})"
        )

    # Gate 3: Smart Money Score
    if stock_data['smart_money_score'] < config.smart_money_min:
        rejection_reasons.append("No smart money support")

    # Gate 4: Near Support
    if not stock_data['is_near_support']:
        dist = stock_data['distance_to_support_pct']
        rejection_reasons.append(f"Not near support level ({dist:.1f}% away)")

    # Gate 5: ADX Trend
    if stock_data['adx'] < config.adx_min:
        rejection_reasons.append(
            f"Weak trend strength (ADX {stock_data['adx']} < {config.adx_min})"
        )

    # Gate 6: Fundamental Score
    if stock_data['fundamental_score'] < config.fundamental_min:
        rejection_reasons.append(
            f"Weak fundamentals ({stock_data['fundamental_score']} < {config.fundamental_min})"
        )

    all_passed = len(rejection_reasons) == 0
    return all_passed, rejection_reasons
```

---

## 5. Trade Plan Generation

### 5.1 Components

When a BUY signal is generated, create a complete trade plan:

| Component | Description | Calculation |
|-----------|-------------|-------------|
| Entry Low | Lower bound of entry range | Support level × 1.01 |
| Entry High | Upper bound of entry range | Current price |
| Stop Loss | Exit if price falls here | Support level × 0.97 (or ATR-based) |
| TP1 | First take profit (25% of position) | Nearest resistance or +5% |
| TP2 | Second take profit (50% of position) | Second resistance or +10% |
| TP3 | Final take profit (25% remaining) | Third resistance or +15% |
| Risk/Reward | Ratio of potential gain vs loss | (TP1 - Entry) / (Entry - SL) |

### 5.2 Position Sizing Integration

Use existing risk management (2% rule) with trade plan:

```python
def calculate_position_with_plan(
    capital: float,
    trade_plan: dict,
    risk_per_trade: float = 0.02
) -> dict:
    """Calculate position size based on trade plan stop loss."""
    entry_price = (trade_plan['entry_low'] + trade_plan['entry_high']) / 2
    stop_loss = trade_plan['stop_loss']

    max_loss = capital * risk_per_trade
    risk_per_share = entry_price - stop_loss

    shares = int(max_loss / risk_per_share)
    lots = shares // 100  # IDX lot size
    shares = lots * 100

    return {
        'shares': shares,
        'lots': lots,
        'position_value': shares * entry_price,
        'max_loss': shares * risk_per_share,
        'risk_pct': (shares * risk_per_share) / capital * 100
    }
```

---

## 6. AI Agent Restructuring

### 6.1 Current State (7 Agents)

| Agent | Current Role | Issue |
|-------|--------------|-------|
| Market Scanner | Find opportunities | Replaced by gate filter |
| Researcher | Gather data | Data already computed |
| Technical Analyst | Chart analysis | Keep - focused validation |
| Sentiment Analyst | News sentiment | Already in scoring |
| Portfolio Manager | Allocation | Handled by risk module |
| Risk Manager | Risk assessment | Keep - validate plan |
| Trading Executor | Execute trades | Separate concern |

### 6.2 Proposed State (3 Focused Agents)

| New Agent | Role | Key Question |
|-----------|------|--------------|
| **Technical Analyst** | Validate chart setup | "Is this a valid entry point?" |
| **Fundamental Analyst** | Verify financials | "Are the financials truly strong?" |
| **Risk Manager** | Validate trade plan | "Is the risk/reward acceptable?" |

### 6.3 New Flow

```
Stock Passes All Gates
       ↓
┌──────────────────────────────────────────────────────┐
│  Technical Analyst                                    │
│  Input: Price data, indicators, support/resistance   │
│  Check: Chart pattern valid? Entry timing good?      │
│  Output: APPROVE/REJECT with reason                  │
└──────────────────────────────────────────────────────┘
       ↓ (if approved)
┌──────────────────────────────────────────────────────┐
│  Fundamental Analyst                                  │
│  Input: Financial metrics, sector comparison         │
│  Check: Financials truly healthy? Any red flags?     │
│  Output: APPROVE/REJECT with reason                  │
└──────────────────────────────────────────────────────┘
       ↓ (if approved)
┌──────────────────────────────────────────────────────┐
│  Risk Manager                                         │
│  Input: Trade plan, portfolio state, correlations    │
│  Check: R/R acceptable? Position size safe?          │
│  Output: APPROVE/REJECT with reason                  │
└──────────────────────────────────────────────────────┘
       ↓ (if all approve)
    FINAL BUY SIGNAL
```

### 6.4 Agent Prompt Structure

Each agent gets a structured prompt with:
1. **Quantitative data** - Pre-computed scores and metrics
2. **Specific criteria** - What to check (not open-ended)
3. **Binary output** - APPROVE or REJECT with one-sentence reason
4. **Veto power** - Any rejection stops the process

Example prompt for Technical Analyst:

```
You are validating a potential BUY signal for {ticker}.

PRE-COMPUTED DATA:
- Technical Score: {tech_score}/100
- RSI: {rsi} (14-day)
- MACD: {macd_signal}
- ADX: {adx} ({trend_strength})
- Price vs SMA20: {pct_above_sma20}%
- Price vs SMA50: {pct_above_sma50}%
- Distance to Support: {support_distance}%
- Distance to Resistance: {resistance_distance}%

CHART SETUP:
- Nearest Support: {support_level}
- Nearest Resistance: {resistance_level}
- Current Price: {current_price}

YOUR TASK:
Validate that this is a good ENTRY POINT for a swing trade (1-4 weeks hold).

CHECK THESE SPECIFIC CRITERIA:
1. Is price near support (within 5%)?
2. Is the trend direction favorable (price above SMA20)?
3. Is momentum positive (MACD bullish or histogram rising)?
4. Is there room to resistance (at least 5% upside)?

OUTPUT FORMAT:
DECISION: APPROVE or REJECT
REASON: One sentence explaining why
```

---

## 7. Integration Strategy

### 7.1 Hybrid Approach

Layer new gates on top of existing scoring, don't replace:

```python
# Existing flow (unchanged)
current_score = calculate_composite_score(stock)  # 0-100

# NEW: Calculate additional components
smart_money = calculate_smart_money_score(df)
support_resistance = find_support_resistance(df)
adx = calculate_adx(df)

# NEW: Gate validation
gates_passed, rejection_reasons = validate_gates({
    'overall_score': current_score,
    'technical_score': current_score * 0.25,  # Extract component
    'smart_money_score': smart_money['score'],
    'is_near_support': support_resistance['is_near_support'],
    'distance_to_support_pct': support_resistance['distance_to_support_pct'],
    'adx': adx['adx'],
    'fundamental_score': current_score * 0.30,  # Extract component
}, config)

if not gates_passed:
    return Signal(
        type=SignalType.NO_TRADE,
        confidence=Confidence.REJECTED,
        rejection_reasons=rejection_reasons
    )

# NEW: Generate trade plan for qualified stocks
trade_plan = generate_trade_plan(
    current_price=support_resistance['current_price'],
    support=support_resistance['nearest_support'],
    resistances=support_resistance['resistances']
)

# Run AI validation only on qualified stocks
if config.enable_ai_validation:
    ai_result = run_focused_ai_validation(stock, trade_plan)
    if not ai_result['approved']:
        return Signal(
            type=SignalType.NO_TRADE,
            confidence=Confidence.REJECTED,
            rejection_reasons=[ai_result['rejection_reason']]
        )

return Signal(
    type=SignalType.BUY,
    confidence=Confidence.HIGH,
    trade_plan=trade_plan
)
```

### 7.2 New Modules to Create

| Module | Purpose | Location |
|--------|---------|----------|
| `smart_money.py` | Accumulation/distribution scoring | `src/stockai/scoring/` |
| `support_resistance.py` | Pivot point detection | `src/stockai/scoring/` |
| `gates.py` | Gate validation logic | `src/stockai/scoring/` |
| `trade_plan.py` | Entry/SL/TP generation | `src/stockai/scoring/` |

### 7.3 Modules to Modify

| Module | Changes |
|--------|---------|
| `autopilot/engine.py` | Add gate check before AI phase |
| `agents/orchestrator.py` | Reduce to 3 focused agents |
| `agents/prompts/` | New structured prompts |
| `cli/main.py` | Add `analyze` command with trade plan output |
| `data/models.py` | Add gate validation fields to AutopilotValidation |

---

## 8. CLI Output Enhancement

### 8.1 New `analyze` Command

```
$ stockai analyze BBCA

┌─────────────────────────────────────────────────────────────────────────┐
│  BBCA - Bank Central Asia Tbk                           BUY (HIGH)     │
├─────────────────────────────────────────────────────────────────────────┤
│  Current Price: Rp 9,500                                               │
│  Overall Score: 82/100                                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  SCORES                                                                │
│  ├── Technical:    78/100  ████████████████░░░░                        │
│  ├── Smart Money:  4.2/5   ████████████████░░░░  (ACCUMULATION)        │
│  ├── Fundamental:  85/100  █████████████████░░░                        │
│  └── ADX:          28      STRONG TREND                                │
│                                                                         │
│  GATES PASSED: 6/6                                                     │
│  ✓ Overall Score (82 >= 70)                                            │
│  ✓ Technical Score (78 >= 60)                                          │
│  ✓ Smart Money (4.2 >= 3.0)                                            │
│  ✓ Near Support (3.2% <= 5%)                                           │
│  ✓ ADX Trend (28 >= 20)                                                │
│  ✓ Fundamental Score (85 >= 60)                                        │
│                                                                         │
│  TRADE PLAN                                                            │
│  ├── Entry Range:   Rp 9,350 - Rp 9,500                                │
│  ├── Stop Loss:     Rp 9,100 (-4.2%)                                   │
│  ├── Take Profit 1: Rp 9,800 (+3.2%) - Sell 25%                        │
│  ├── Take Profit 2: Rp 10,200 (+7.4%) - Sell 50%                       │
│  ├── Take Profit 3: Rp 10,800 (+13.7%) - Sell 25%                      │
│  └── Risk/Reward:   1:2.3                                              │
│                                                                         │
│  POSITION SIZE (Capital: Rp 10,000,000, Risk: 2%)                      │
│  └── Recommended: 500 shares (5 lots) = Rp 4,750,000                   │
│                                                                         │
│  AI VALIDATION: APPROVED                                               │
│  ├── Technical Analyst: Valid uptrend with pullback to support         │
│  ├── Fundamental Analyst: Strong ROE, low debt, growing earnings       │
│  └── Risk Manager: R/R acceptable, within sector limits                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 8.2 Rejection Example

```
$ stockai analyze GOTO

┌─────────────────────────────────────────────────────────────────────────┐
│  GOTO - GoTo Gojek Tokopedia                         NO TRADE (REJECT) │
├─────────────────────────────────────────────────────────────────────────┤
│  Current Price: Rp 268                                                 │
│  Overall Score: 42/100                                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  GATES PASSED: 1/6                                                     │
│  ✗ Overall Score (42 < 70)                                             │
│  ✗ Technical Score (35 < 60)                                           │
│  ✗ Smart Money (-0.5 < 3.0) - DISTRIBUTION                             │
│  ✗ Near Support (12.3% > 5%)                                           │
│  ✗ ADX Trend (15 < 20) - WEAK                                          │
│  ✓ Fundamental Score (62 >= 60)                                        │
│                                                                         │
│  REJECTION REASONS:                                                    │
│  1. Low overall score (42 < 70)                                        │
│  2. Weak technical score (35 < 60)                                     │
│  3. No smart money support - distribution pattern                      │
│  4. Not near support level (12.3% away)                                │
│  5. Weak trend strength (ADX 15 < 20)                                  │
│                                                                         │
│  AI VALIDATION: SKIPPED (failed gate filter)                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Database Schema Updates

### 9.1 New Fields for AutopilotValidation

```python
class AutopilotValidation(Base):
    # Existing fields...

    # NEW: Gate validation fields
    gates_passed = Column(Integer)  # Count of gates passed
    total_gates = Column(Integer, default=6)
    rejection_reasons = Column(JSON)  # List of rejection reason strings

    # NEW: Trade plan fields
    entry_low = Column(Float)
    entry_high = Column(Float)
    stop_loss = Column(Float)
    take_profit_1 = Column(Float)
    take_profit_2 = Column(Float)
    take_profit_3 = Column(Float)
    risk_reward_ratio = Column(Float)

    # NEW: Support/Resistance fields
    nearest_support = Column(Float)
    nearest_resistance = Column(Float)
    distance_to_support_pct = Column(Float)

    # NEW: Smart Money fields
    smart_money_score = Column(Float)
    smart_money_interpretation = Column(String)  # ACCUMULATION, NEUTRAL, DISTRIBUTION

    # NEW: ADX fields
    adx_value = Column(Float)
    adx_trend_strength = Column(String)  # STRONG, MODERATE, WEAK, ABSENT
```

---

## 10. Implementation Phases

### Phase 1: Core Scoring (3-4 days)

- [ ] Create `scoring/smart_money.py`
- [ ] Create `scoring/support_resistance.py`
- [ ] Create `scoring/gates.py`
- [ ] Create `scoring/trade_plan.py`
- [ ] Add unit tests for each module

### Phase 2: Integration (2-3 days)

- [ ] Modify `autopilot/engine.py` - add gate validation phase
- [ ] Update `data/models.py` - add new fields
- [ ] Create database migration

### Phase 3: AI Agent Restructure (3-4 days)

- [ ] Create focused agent prompts
- [ ] Reduce orchestrator to 3 agents
- [ ] Add veto logic (any rejection stops)
- [ ] Test agent accuracy

### Phase 4: CLI Enhancement (2 days)

- [ ] Add `analyze` command
- [ ] Add trade plan output formatting
- [ ] Add rejection reasons display

### Phase 5: Testing & Tuning (3-4 days)

- [ ] Backtest gate thresholds on historical data
- [ ] Tune Smart Money score weights
- [ ] Validate signal quality improvement
- [ ] Document configuration options

---

## 11. Configuration

### 11.1 Gate Thresholds (Configurable)

```python
class GateConfig:
    overall_min: float = 70.0      # Overall score minimum (0-100)
    technical_min: float = 60.0    # Technical score minimum
    smart_money_min: float = 3.0   # Smart money score minimum
    near_support_pct: float = 5.0  # Max distance to support (%)
    adx_min: float = 20.0          # ADX minimum for tradeable trend
    fundamental_min: float = 60.0  # Fundamental score minimum
```

### 11.2 Trade Plan Defaults

```python
class TradePlanConfig:
    stop_loss_pct_below_support: float = 0.03  # 3% below support
    tp1_pct: float = 0.05   # +5% or first resistance
    tp2_pct: float = 0.10   # +10% or second resistance
    tp3_pct: float = 0.15   # +15% or third resistance
    tp1_sell_pct: float = 0.25  # Sell 25% at TP1
    tp2_sell_pct: float = 0.50  # Sell 50% at TP2
    tp3_sell_pct: float = 0.25  # Sell remaining 25% at TP3
```

---

## 12. Success Metrics

| Metric | Current (Est.) | Target | Measurement |
|--------|----------------|--------|-------------|
| Signal Rate | ~30% BUY | 5-10% BUY | % of analyzed stocks |
| Win Rate | ~50% | >60% | % hitting TP1 |
| Avg Win/Loss | ~1.2:1 | >2:1 | Average gain vs loss |
| AI Cost | 7 agents/stock | 3 agents/10% stocks | API calls |
| False Positives | ~40% | <20% | % hitting SL |

---

## 13. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Over-filtering (too few signals) | Miss opportunities | Make thresholds configurable |
| Smart Money false signals | Bad entries | Combine with other indicators |
| Support detection inaccurate | Wrong SL levels | Use multiple detection methods |
| IDX volatility breaks rules | Signals fail on volatile days | Add ATR-based adjustments |

---

## 14. Next Steps

1. **Review this design** - Get feedback on approach
2. **Create implementation plan** - Detailed task breakdown
3. **Set up git worktree** - Isolated development branch
4. **Implement Phase 1** - Core scoring modules
5. **Iterate** - Test, tune, refine

---

**Document Status:** Ready for review
**Author:** Claude (AI Assistant)
**Reviewed by:** Pending
