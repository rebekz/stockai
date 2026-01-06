# AI-Enhanced Autopilot Trading System Design

**Date:** 2026-01-06
**Status:** Approved
**Author:** Claude + User

## Overview

Enhance the autopilot trading system with AI agent validation. All BUY and SELL signals pass through the existing 7-agent orchestrator before execution, giving AI agents veto power over trades.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Integration Approach | Validate + Enhance | AI confirms autopilot signals and provides deeper analysis |
| Trigger | Every run | All signals require AI validation before execution |
| Agents Involved | Full orchestra (7) | Maximum analysis depth for high-conviction trades |
| AI Authority | Veto power | AI can reject signals that don't meet threshold |
| BUY Threshold | AI score ≥ 6.0 | Matches existing "Buy" recommendation level |
| SELL Threshold | AI score ≤ 5.0 | Confirms stock weakness before exit |
| Execution | Parallel with limit | Default 3 concurrent, configurable via `--ai-concurrency` |

## Architecture

### Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                    AUTOPILOT WORKFLOW                        │
├─────────────────────────────────────────────────────────────┤
│  1. SCAN      → Fetch prices, calculate multi-factor scores │
│  2. SIGNAL    → Generate BUY/SELL candidates (score-based)  │
│  3. AI GATE   → [NEW] 7-agent validation with veto power    │
│  4. SIZING    → Calculate position sizes (validated only)   │
│  5. EXECUTE   → Paper trading execution                     │
│  6. REPORT    → Display results + AI insights               │
└─────────────────────────────────────────────────────────────┘
```

### AI Gate Rules

- All BUY signals require AI composite score ≥ 6.0 to proceed
- All SELL signals require AI composite score ≤ 5.0 to proceed
- Signals failing validation are logged but not executed
- Validation runs in parallel (default 3 concurrent, configurable)

## Components

### ValidationResult Dataclass

```python
@dataclass
class ValidationResult:
    symbol: str
    signal_type: str              # "BUY" or "SELL"
    autopilot_score: float        # Original multi-factor score
    ai_composite_score: float     # 7-agent weighted score
    is_approved: bool             # Passed threshold?
    recommendation: str           # STRONG_BUY, BUY, HOLD, SELL, etc.

    # Agent breakdown
    fundamental_score: float
    technical_score: float
    sentiment_score: float
    portfolio_fit_score: float
    risk_score: float

    # Insights for display
    key_reasons: list[str]        # Top 3 reasons for decision
    risk_factors: list[str]       # Key risks identified
    entry_price: float | None     # AI-suggested entry
    stop_loss: float | None       # AI-suggested stop
    target_price: float | None    # AI-suggested target
```

### AIValidator Class

Located in `src/stockai/autopilot/validator.py`:

1. Receive `TradeSignal` from autopilot
2. Call `TradingOrchestrator.analyze(symbol)` - runs all 7 agents
3. Extract composite score and agent breakdown
4. Compare against threshold (≥6.0 for BUY, ≤5.0 for SELL)
5. Return `ValidationResult` with approval status + insights

Uses `asyncio.Semaphore(n)` to limit parallel validations.

## CLI Integration

### Enhanced `autopilot run` Command

```
stockai autopilot run [OPTIONS]

Options:
  -i, --index          JII70|IDX30|LQ45|ALL  [default: JII70]
  -c, --capital        Available capital in Rupiah
  -n, --dry-run        Show signals without executing
  -f, --force          Execute even if already run today

  # AI Options:
  --ai / --no-ai       Enable/disable AI validation  [default: --ai]
  --ai-concurrency N   Parallel validation limit     [default: 3]
  --ai-threshold FLOAT BUY approval threshold        [default: 6.0]
  --ai-verbose         Show detailed agent analysis
```

### New Subcommand

```
stockai autopilot validate SYMBOL [OPTIONS]

  Run AI validation on a specific stock without full autopilot workflow.

Options:
  --verbose    Show full 7-agent breakdown
```

### Output Format

```
BUY SIGNALS:
  PWON: Score 78 → AI: 7.2 ✓ APPROVED (Strong fundamentals, bullish MACD)
  TLKM: Score 72 → AI: 5.4 ✗ REJECTED (Sentiment negative, resistance ahead)

SELL SIGNALS:
  BRPT: Score 45 → AI: 4.1 ✓ CONFIRMED (Weak technicals, sector rotation)
```

## Database Schema

### Enhanced AutopilotTrade

```python
class AutopilotTrade(Base):
    # Existing fields...

    # NEW: AI Validation fields
    ai_validated = Column(Boolean, default=False)
    ai_composite_score = Column(Float)
    ai_fundamental_score = Column(Float)
    ai_technical_score = Column(Float)
    ai_sentiment_score = Column(Float)
    ai_risk_score = Column(Float)
    ai_recommendation = Column(String(20))
    ai_approved = Column(Boolean)
    ai_rejection_reason = Column(String(200))
```

### New AutopilotValidation Table

```python
class AutopilotValidation(Base):
    """Track all AI validations, including rejections."""

    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey("autopilot_runs.id"))
    symbol = Column(String(10), nullable=False)
    signal_type = Column(String(4))  # BUY/SELL
    autopilot_score = Column(Float)
    ai_composite_score = Column(Float)
    is_approved = Column(Boolean)
    rejection_reason = Column(String(200))
    created_at = Column(DateTime)
```

## Error Handling

### API Failures

| Scenario | Behavior |
|----------|----------|
| Single stock fails | Retry once, then mark as "AI_UNAVAILABLE" - signal skipped |
| All validations fail | Abort run, show error, suggest `--no-ai` fallback |
| Partial failures | Execute approved signals, log failures for review |

### Edge Cases

| Case | Handling |
|------|----------|
| No BUY signals generated | Skip AI validation, report "no opportunities" |
| AI approves but insufficient cash | Normal sizing logic applies, may skip |
| Same stock has BUY and SELL signal | SELL takes priority |
| AI score exactly at threshold (6.0) | Approved (≥ is inclusive) |

### Timeouts

- Per-stock validation timeout: 60 seconds
- Full run timeout: 10 minutes max
- Retry delay: 2 seconds between retries

## File Structure

### New Files

```
src/stockai/autopilot/
├── validator.py          # AIValidator, ValidationResult
tests/unit/
├── test_autopilot_ai.py  # Unit tests for AI validation
```

### Modified Files

```
src/stockai/autopilot/engine.py    # Add AI gate step
src/stockai/autopilot/__init__.py  # Export new classes
src/stockai/data/models.py         # Add AI fields, new table
src/stockai/cli/main.py            # Add --ai options, validate command
```

## Testing Strategy

### Unit Tests

```python
class TestAIValidator:
    def test_buy_approved_when_ai_score_above_threshold()
    def test_buy_rejected_when_ai_score_below_threshold()
    def test_sell_confirmed_when_ai_score_below_threshold()
    def test_sell_rejected_when_ai_score_above_threshold()
    def test_score_exactly_at_threshold_is_approved()
    def test_api_failure_marks_signal_unavailable()
    def test_timeout_triggers_retry()

class TestAutopilotWithAI:
    def test_ai_enabled_runs_validation()
    def test_no_ai_flag_skips_validation()
    def test_rejected_signals_not_executed()
    def test_concurrency_limit_respected()
```

### Mocking Strategy

- Mock `TradingOrchestrator` to avoid real LLM calls
- Return predetermined scores for test scenarios
- Test both approval and rejection paths

## Implementation Priority

1. **Phase 1**: `ValidationResult` dataclass + `AIValidator` class
2. **Phase 2**: Integrate into `AutopilotEngine.run()` workflow
3. **Phase 3**: Database model updates
4. **Phase 4**: CLI options and output formatting
5. **Phase 5**: Unit tests
6. **Phase 6**: Error handling refinement

## Scope Estimate

- ~400 lines new code (validator + tests)
- ~150 lines modifications (engine, CLI, models)
- Reuses existing `TradingOrchestrator` unchanged
