# StockAI Quality Over Quantity - Implementation Plan

**Date:** 2026-01-06
**Design Doc:** `docs/plans/2026-01-06-stockai-quality-over-quantity-design.md`
**Estimated Total:** 15-20 implementation tasks

---

## Phase 1: Core Scoring Modules (4 tasks)

### Task 1.1: Create Smart Money Score Module

**File:** `src/stockai/scoring/smart_money.py`

**Steps:**
1. Create file with module docstring
2. Add imports: pandas, pandas_ta (existing in project)
3. Create `SmartMoneyResult` dataclass:
   ```python
   @dataclass
   class SmartMoneyResult:
       score: float  # -2.0 to 5.0
       accumulation_days: int
       distribution_days: int
       net_accumulation: int
       obv_trend: str  # BULLISH, NEUTRAL, BEARISH
       mfi: float
       mfi_signal: str
       unusual_volume: str
       interpretation: str  # ACCUMULATION, NEUTRAL, DISTRIBUTION
   ```
4. Implement `calculate_smart_money_score(df: pd.DataFrame) -> SmartMoneyResult`:
   - Calculate daily returns and volume ratio
   - Count accumulation days (price up + volume > 1.2x avg)
   - Count distribution days (price down + volume > 1.2x avg)
   - Calculate OBV using pandas_ta
   - Calculate MFI using pandas_ta
   - Detect unusual volume spikes
   - Score and clamp to -2.0 to 5.0 range
5. Add to `scoring/__init__.py` exports

**Verification:**
```bash
uv run python -c "from stockai.scoring.smart_money import calculate_smart_money_score; print('OK')"
```

---

### Task 1.2: Create Support/Resistance Module

**File:** `src/stockai/scoring/support_resistance.py`

**Steps:**
1. Create file with module docstring
2. Add imports: pandas, numpy, scipy.signal.argrelextrema
3. Create `SupportResistanceResult` dataclass:
   ```python
   @dataclass
   class SupportResistanceResult:
       current_price: float
       supports: list[float]  # Up to 3 levels
       resistances: list[float]  # Up to 3 levels
       nearest_support: float | None
       nearest_resistance: float | None
       distance_to_support_pct: float | None
       is_near_support: bool
       suggested_stop_loss: float
   ```
4. Implement `find_support_resistance(df: pd.DataFrame, lookback: int = 60) -> SupportResistanceResult`:
   - Use argrelextrema to find pivot highs (order=5)
   - Use argrelextrema to find pivot lows (order=5)
   - Filter levels within 20% of current price
   - Calculate distance to nearest support
   - Set is_near_support = True if distance <= 5%
   - Calculate suggested stop loss (3% below support or 8% below price)
5. Add to `scoring/__init__.py` exports

**Verification:**
```bash
uv run python -c "from stockai.scoring.support_resistance import find_support_resistance; print('OK')"
```

---

### Task 1.3: Create Gate Validation Module

**File:** `src/stockai/scoring/gates.py`

**Steps:**
1. Create file with module docstring
2. Add imports: dataclasses
3. Create `GateConfig` dataclass:
   ```python
   @dataclass
   class GateConfig:
       overall_min: float = 70.0
       technical_min: float = 60.0
       smart_money_min: float = 3.0
       near_support_pct: float = 5.0
       adx_min: float = 20.0
       fundamental_min: float = 60.0
   ```
4. Create `GateResult` dataclass:
   ```python
   @dataclass
   class GateResult:
       all_passed: bool
       gates_passed: int
       total_gates: int
       passed_gates: list[str]
       rejection_reasons: list[str]
       confidence: str  # HIGH, WATCH, REJECTED
   ```
5. Implement `validate_gates(stock_data: dict, config: GateConfig = None) -> GateResult`:
   - Check each of 6 gates
   - Build passed_gates and rejection_reasons lists
   - Determine confidence level:
     - HIGH: 0 failures
     - WATCH: 1-2 failures AND overall >= 60
     - REJECTED: 3+ failures OR overall < 60
6. Add to `scoring/__init__.py` exports

**Verification:**
```bash
uv run python -c "from stockai.scoring.gates import validate_gates, GateConfig; print('OK')"
```

---

### Task 1.4: Create Trade Plan Module

**File:** `src/stockai/scoring/trade_plan.py`

**Steps:**
1. Create file with module docstring
2. Add imports: dataclasses
3. Create `TradePlanConfig` dataclass:
   ```python
   @dataclass
   class TradePlanConfig:
       stop_loss_pct_below_support: float = 0.03
       tp1_pct: float = 0.05
       tp2_pct: float = 0.10
       tp3_pct: float = 0.15
       tp1_sell_pct: float = 0.25
       tp2_sell_pct: float = 0.50
       tp3_sell_pct: float = 0.25
   ```
4. Create `TradePlan` dataclass:
   ```python
   @dataclass
   class TradePlan:
       entry_low: float
       entry_high: float
       stop_loss: float
       take_profit_1: float
       take_profit_2: float
       take_profit_3: float
       risk_reward_ratio: float
       risk_pct: float
       summary: str
   ```
5. Implement `generate_trade_plan(current_price, support, resistances, config) -> TradePlan`:
   - Calculate entry range (support*1.01 to current_price)
   - Calculate stop loss (support * 0.97 or current * 0.92 if no support)
   - Calculate TPs from resistances or default percentages
   - Ensure TPs are above entry and ascending
   - Calculate risk/reward ratio
6. Implement `calculate_position_with_plan(capital, trade_plan, risk_pct=0.02) -> dict`:
   - Use existing 2% risk rule logic
   - Return shares, lots, position_value, max_loss
7. Add to `scoring/__init__.py` exports

**Verification:**
```bash
uv run python -c "from stockai.scoring.trade_plan import generate_trade_plan, TradePlan; print('OK')"
```

---

## Phase 2: Database & Integration (3 tasks)

### Task 2.1: Update Database Models

**File:** `src/stockai/data/models.py`

**Steps:**
1. Read current AutopilotValidation model
2. Add new fields (after existing fields):
   ```python
   # Gate validation
   gates_passed = Column(Integer)
   total_gates = Column(Integer, default=6)
   rejection_reasons_json = Column(JSON)  # List of strings

   # Trade plan
   entry_low = Column(Float)
   entry_high = Column(Float)
   stop_loss = Column(Float)
   take_profit_1 = Column(Float)
   take_profit_2 = Column(Float)
   take_profit_3 = Column(Float)
   risk_reward_ratio = Column(Float)

   # Support/Resistance
   nearest_support = Column(Float)
   nearest_resistance = Column(Float)
   distance_to_support_pct = Column(Float)

   # Smart Money
   smart_money_score = Column(Float)
   smart_money_interpretation = Column(String(20))

   # ADX
   adx_value = Column(Float)
   adx_trend_strength = Column(String(20))
   ```
3. Run migration test (init db in temp location)

**Verification:**
```bash
uv run python -c "from stockai.data.models import AutopilotValidation; print([c.name for c in AutopilotValidation.__table__.columns if 'gate' in c.name or 'smart' in c.name])"
```

---

### Task 2.2: Add ADX Calculation Helper

**File:** `src/stockai/tools/stock_tools.py`

**Steps:**
1. Read existing get_technical_indicators function
2. Add new function `calculate_adx(df, period=14) -> dict`:
   ```python
   def calculate_adx(df: pd.DataFrame, period: int = 14) -> dict:
       """Calculate ADX and directional indicators."""
       adx_data = ta.adx(df['high'], df['low'], df['close'], length=period)
       current_adx = adx_data[f'ADX_{period}'].iloc[-1]
       plus_di = adx_data[f'DMP_{period}'].iloc[-1]
       minus_di = adx_data[f'DMN_{period}'].iloc[-1]

       trend_direction = "BULLISH" if plus_di > minus_di else "BEARISH"

       if current_adx >= 50:
           trend_strength = "VERY_STRONG"
       elif current_adx >= 25:
           trend_strength = "STRONG"
       elif current_adx >= 20:
           trend_strength = "MODERATE"
       elif current_adx >= 15:
           trend_strength = "WEAK"
       else:
           trend_strength = "ABSENT"

       return {
           'adx': round(current_adx, 1),
           'plus_di': round(plus_di, 1),
           'minus_di': round(minus_di, 1),
           'trend_direction': trend_direction,
           'trend_strength': trend_strength,
           'is_tradeable': current_adx >= 20
       }
   ```
3. Add to module exports

**Verification:**
```bash
uv run python -c "from stockai.tools.stock_tools import calculate_adx; print('OK')"
```

---

### Task 2.3: Create Integrated Analyzer Service

**File:** `src/stockai/scoring/analyzer.py`

**Steps:**
1. Create file - this is the main integration point
2. Create `AnalysisResult` dataclass combining all components:
   ```python
   @dataclass
   class AnalysisResult:
       ticker: str
       current_price: float

       # Existing scores (from factors.py)
       composite_score: float
       value_score: float
       quality_score: float
       momentum_score: float
       volatility_score: float

       # New scores
       smart_money: SmartMoneyResult
       support_resistance: SupportResistanceResult
       adx: dict

       # Gate validation
       gates: GateResult

       # Trade plan (if qualified)
       trade_plan: TradePlan | None

       # Final decision
       decision: str  # BUY, NO_TRADE
       confidence: str  # HIGH, WATCH, REJECTED
   ```
3. Implement `analyze_stock(ticker: str, df: pd.DataFrame, fundamentals: dict, config: GateConfig = None) -> AnalysisResult`:
   - Call existing score_stock() for composite scores
   - Call calculate_smart_money_score()
   - Call find_support_resistance()
   - Call calculate_adx()
   - Call validate_gates() with all data
   - If gates pass, call generate_trade_plan()
   - Return complete AnalysisResult
4. Add to `scoring/__init__.py` exports

**Verification:**
```bash
uv run python -c "from stockai.scoring.analyzer import analyze_stock, AnalysisResult; print('OK')"
```

---

## Phase 3: Autopilot Integration (2 tasks)

### Task 3.1: Add Gate Phase to Autopilot Engine

**File:** `src/stockai/autopilot/engine.py`

**Steps:**
1. Read current _generate_signals method
2. Add new method `_apply_gate_filter(self, buy_signals: list[TradeSignal]) -> tuple[list, list]`:
   - For each buy signal:
     - Fetch price history for the stock
     - Call analyze_stock() to get AnalysisResult
     - If gates.all_passed: add to qualified list with trade_plan
     - Else: add to rejected list with rejection_reasons
   - Return (qualified_signals, rejected_signals)
3. Modify run() workflow:
   - After SIGNAL phase, add GATE phase
   - Call _apply_gate_filter() on buy_signals
   - Only pass qualified_signals to AI GATE phase
   - Log rejected signals with reasons
4. Update AutopilotResult to include gate statistics

**Verification:**
```bash
uv run stockai autopilot --dry-run --verbose 2>&1 | head -50
```

---

### Task 3.2: Update Autopilot Validation Storage

**File:** `src/stockai/autopilot/engine.py`

**Steps:**
1. Find where AutopilotValidation records are created
2. Add new fields when saving:
   ```python
   validation = AutopilotValidation(
       # ... existing fields ...
       gates_passed=result.gates.gates_passed,
       total_gates=result.gates.total_gates,
       rejection_reasons_json=result.gates.rejection_reasons,
       entry_low=result.trade_plan.entry_low if result.trade_plan else None,
       entry_high=result.trade_plan.entry_high if result.trade_plan else None,
       stop_loss=result.trade_plan.stop_loss if result.trade_plan else None,
       take_profit_1=result.trade_plan.take_profit_1 if result.trade_plan else None,
       take_profit_2=result.trade_plan.take_profit_2 if result.trade_plan else None,
       take_profit_3=result.trade_plan.take_profit_3 if result.trade_plan else None,
       risk_reward_ratio=result.trade_plan.risk_reward_ratio if result.trade_plan else None,
       nearest_support=result.support_resistance.nearest_support,
       nearest_resistance=result.support_resistance.nearest_resistance,
       distance_to_support_pct=result.support_resistance.distance_to_support_pct,
       smart_money_score=result.smart_money.score,
       smart_money_interpretation=result.smart_money.interpretation,
       adx_value=result.adx['adx'],
       adx_trend_strength=result.adx['trend_strength'],
   )
   ```
3. Test with dry-run

**Verification:**
```bash
uv run stockai autopilot --dry-run --limit 5
```

---

## Phase 4: AI Agent Restructure (3 tasks)

### Task 4.1: Create Focused Agent Prompts

**File:** `src/stockai/agents/prompts/focused_agents.py`

**Steps:**
1. Create file with 3 prompt templates
2. `TECHNICAL_ANALYST_PROMPT`:
   ```python
   TECHNICAL_ANALYST_PROMPT = """
   You are validating a potential BUY signal for {ticker}.

   PRE-COMPUTED DATA:
   - Technical Score: {tech_score}/100
   - RSI: {rsi} (14-day)
   - MACD: {macd_signal}
   - ADX: {adx} ({trend_strength})
   - Price vs SMA20: {pct_above_sma20}%
   - Distance to Support: {support_distance}%
   - Distance to Resistance: {resistance_distance}%

   YOUR TASK: Validate this is a good ENTRY POINT.

   CHECK THESE CRITERIA:
   1. Is price near support (within 5%)?
   2. Is trend favorable (price above SMA20)?
   3. Is momentum positive (MACD bullish)?
   4. Is there room to resistance (at least 5% upside)?

   OUTPUT (exactly this format):
   DECISION: APPROVE or REJECT
   REASON: One sentence
   """
   ```
3. `FUNDAMENTAL_ANALYST_PROMPT` - focus on ROE, debt, earnings quality
4. `RISK_MANAGER_PROMPT` - focus on R/R ratio, position size, sector exposure
5. Add parsing helper `parse_agent_response(response: str) -> tuple[str, str]`

**Verification:**
```bash
uv run python -c "from stockai.agents.prompts.focused_agents import TECHNICAL_ANALYST_PROMPT; print(len(TECHNICAL_ANALYST_PROMPT))"
```

---

### Task 4.2: Create Focused Validator

**File:** `src/stockai/agents/focused_validator.py`

**Steps:**
1. Create new validator class (don't modify existing orchestrator yet)
2. `FocusedValidator` class:
   ```python
   class FocusedValidator:
       def __init__(self, model_name: str = "gemini-1.5-flash"):
           self.llm = ChatGoogleGenerativeAI(model=model_name)

       async def validate(self, analysis: AnalysisResult) -> FocusedValidationResult:
           # Step 1: Technical Analyst
           tech_result = await self._run_technical_analyst(analysis)
           if tech_result.decision == "REJECT":
               return FocusedValidationResult(approved=False, rejected_by="technical", reason=tech_result.reason)

           # Step 2: Fundamental Analyst
           fund_result = await self._run_fundamental_analyst(analysis)
           if fund_result.decision == "REJECT":
               return FocusedValidationResult(approved=False, rejected_by="fundamental", reason=fund_result.reason)

           # Step 3: Risk Manager
           risk_result = await self._run_risk_manager(analysis)
           if risk_result.decision == "REJECT":
               return FocusedValidationResult(approved=False, rejected_by="risk", reason=risk_result.reason)

           return FocusedValidationResult(
               approved=True,
               tech_reason=tech_result.reason,
               fund_reason=fund_result.reason,
               risk_reason=risk_result.reason
           )
   ```
3. Implement `_run_technical_analyst`, `_run_fundamental_analyst`, `_run_risk_manager`
4. Add timeout handling (30s per agent)

**Verification:**
```bash
uv run python -c "from stockai.agents.focused_validator import FocusedValidator; print('OK')"
```

---

### Task 4.3: Integrate Focused Validator into Autopilot

**File:** `src/stockai/autopilot/engine.py`

**Steps:**
1. Add config option: `use_focused_validation: bool = True`
2. Modify `_validate_signals_with_ai`:
   ```python
   if self.config.use_focused_validation:
       validator = FocusedValidator()
       for signal in qualified_signals:
           result = await validator.validate(signal.analysis)
           if result.approved:
               approved.append(signal)
           else:
               rejected.append((signal, result.reason))
   else:
       # Use existing 7-agent orchestrator
       ...
   ```
3. Update logging to show which agent rejected
4. Test with both validation modes

**Verification:**
```bash
uv run stockai autopilot --dry-run --ai --limit 3
```

---

## Phase 5: CLI Enhancement (2 tasks)

### Task 5.1: Add `analyze` Command

**File:** `src/stockai/cli/main.py`

**Steps:**
1. Find existing stock analysis commands
2. Add new `analyze` command:
   ```python
   @app.command()
   def analyze(
       symbol: str = typer.Argument(..., help="Stock symbol to analyze"),
       capital: float = typer.Option(10_000_000, help="Capital for position sizing"),
       ai: bool = typer.Option(False, "--ai", help="Run AI validation"),
   ):
       """Full stock analysis with gate filter and trade plan."""
   ```
3. Implement:
   - Fetch price history
   - Fetch fundamentals
   - Call analyze_stock()
   - If --ai and gates pass, run focused_validator
   - Display results using Rich
4. Create helper `format_analysis_result(result: AnalysisResult) -> Panel`

**Verification:**
```bash
uv run stockai analyze BBCA
```

---

### Task 5.2: Add Rich Output Formatting

**File:** `src/stockai/cli/formatters.py` (new file)

**Steps:**
1. Create file with Rich formatting helpers
2. `format_scores_table(result: AnalysisResult) -> Table`:
   - Technical, Smart Money, Fundamental, ADX with progress bars
3. `format_gates_panel(gates: GateResult) -> Panel`:
   - Show each gate with checkmark or X
   - List rejection reasons
4. `format_trade_plan_panel(plan: TradePlan, capital: float) -> Panel`:
   - Entry range, SL, TPs with percentages
   - Position sizing recommendation
5. `format_ai_validation_panel(validation: FocusedValidationResult) -> Panel`:
   - Each agent's decision and reason
6. `format_full_analysis(result: AnalysisResult) -> Group`:
   - Combine all panels into cohesive output

**Verification:**
```bash
uv run stockai analyze BBCA --ai
```

---

## Phase 6: Testing & Verification (2 tasks)

### Task 6.1: Add Unit Tests for New Modules

**File:** `tests/scoring/test_quality_gates.py`

**Steps:**
1. Create test file
2. Test `calculate_smart_money_score`:
   - Test accumulation pattern (score > 3)
   - Test distribution pattern (score < 0)
   - Test neutral pattern
3. Test `find_support_resistance`:
   - Test with clear pivot points
   - Test near support detection
   - Test no support found
4. Test `validate_gates`:
   - Test all gates pass
   - Test 1-2 gates fail (WATCH)
   - Test 3+ gates fail (REJECTED)
5. Test `generate_trade_plan`:
   - Test with resistance levels
   - Test without resistance (default percentages)
   - Test R/R calculation

**Verification:**
```bash
uv run pytest tests/scoring/test_quality_gates.py -v
```

---

### Task 6.2: Integration Test with Real Data

**File:** `tests/integration/test_analyzer_flow.py`

**Steps:**
1. Create integration test
2. Test full flow for known stocks:
   - BBCA (likely high score)
   - GOTO (likely low score)
3. Verify:
   - All components return valid data
   - Gate validation matches expected results
   - Trade plan generated for qualified stocks
   - Rejection reasons clear for failed stocks
4. Performance test:
   - Analyze 10 stocks, verify < 30 seconds

**Verification:**
```bash
uv run pytest tests/integration/test_analyzer_flow.py -v
```

---

## Execution Order Summary

| Batch | Tasks | Verification |
|-------|-------|--------------|
| 1 | 1.1, 1.2, 1.3 | Import tests pass |
| 2 | 1.4, 2.1, 2.2 | Import tests pass |
| 3 | 2.3, 3.1 | Dry-run autopilot works |
| 4 | 3.2, 4.1, 4.2 | Focused validator works |
| 5 | 4.3, 5.1, 5.2 | `analyze` command works |
| 6 | 6.1, 6.2 | All tests pass |

---

**Document Status:** Ready for execution
**Author:** Claude (AI Assistant)
