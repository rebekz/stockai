# Autopilot Trading System Design

**Date:** 2025-01-06
**Status:** Approved
**Author:** Claude + User

## Overview

Automated daily trading system for StockAI that scans Indonesian stock indices, generates buy/sell signals using multi-factor scoring, and executes trades via paper trading.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Execution Mode | Paper trading only | Validate algorithm before live trading |
| Index Support | JII70 (default), IDX30, LQ45, ALL | Flexibility with sharia-compliant default |
| Scoring System | New multi-factor (Value/Quality/Momentum/Volatility) | Comprehensive fundamental + technical analysis |
| Position Sizing | ATR-based + 20% hard cap | Respects volatility while preventing over-concentration |
| Command Structure | Single `autopilot` with subcommands | Clean namespace, easy to extend |

## Command Structure

```
stockai autopilot run [OPTIONS]    # Execute daily workflow
stockai autopilot status           # Current state
stockai autopilot alerts           # Triggered alerts
stockai autopilot history          # Trade history
stockai autopilot rebalance        # Monthly rebalancing
```

### `autopilot run` Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--index` | `-i` | JII70 | Index to scan (JII70, IDX30, LQ45, ALL) |
| `--capital` | `-c` | cached | Available capital in Rupiah |
| `--dry-run` | `-n` | false | Show signals without executing |
| `--force` | `-f` | false | Execute even if already run today |

## Multi-Factor Scoring System

### Weights

| Factor | Weight | Components |
|--------|--------|------------|
| Value | 25% | P/E ratio, P/B ratio, Dividend Yield |
| Quality | 30% | ROE, Debt/Equity, Profit Margin |
| Momentum | 25% | RSI, MACD, Price vs SMA |
| Volatility | 20% | Beta, ATR (lower = better) |

### Signal Thresholds

| Score Range | Signal | Action |
|-------------|--------|--------|
| > 70 | BUY | Add to portfolio if not held |
| 50-70 | HOLD | No action |
| < 50 | SELL | Exit existing positions |

### Scoring Details

**Value Score (25%)**
- P/E ratio vs sector median (lower = better, max 25 points)
- P/B ratio vs sector median (lower = better, max 25 points)
- Dividend yield > 3% = good (max 50 points)

**Quality Score (30%)**
- ROE > 15% = 40 points, > 10% = 20 points
- Debt/Equity < 0.5 = 30 points, < 1.0 = 15 points
- Profit margin vs sector (max 30 points)

**Momentum Score (25%)**
- RSI 30-50 = 40 points (oversold recovery)
- RSI 50-70 = 30 points (healthy trend)
- MACD bullish crossover = 30 points
- Price > 20-day SMA = 30 points

**Volatility Score (20%)**
- Beta < 0.8 = 50 points (defensive)
- Beta 0.8-1.2 = 30 points (market-aligned)
- Low ATR% = higher score (stability rewarded)

## Position Sizing & Risk Management

### Constants

```python
MAX_POSITION_PCT = 0.20      # Max 20% of portfolio per stock
MAX_SECTOR_PCT = 0.40        # Max 40% per sector
RISK_PER_TRADE = 0.02        # Risk 2% per trade
ATR_MULTIPLIER = 2.0         # Stop-loss = Entry - 2×ATR
MAX_POSITIONS = 10           # Maximum concurrent positions
```

### Position Size Calculation

```
1. stop_loss = entry_price - (2 × ATR)
2. risk_per_share = entry_price - stop_loss
3. shares_by_risk = (portfolio × 2%) ÷ risk_per_share
4. shares_by_cap = (portfolio × 20%) ÷ entry_price
5. final_shares = min(shares_by_risk, shares_by_cap)
6. lots = floor(final_shares ÷ 100)
```

### Example Calculation

```
Portfolio: Rp 10,000,000
Stock PWON: Price Rp 340, ATR Rp 15

Stop-loss: 340 - (2 × 15) = Rp 310
Risk per share: Rp 30

By 2% rule: (10M × 2%) ÷ 30 = 6,666 shares
By 20% cap: (10M × 20%) ÷ 340 = 5,882 shares

Final: 5,800 shares = 58 lots
Total investment: Rp 1,972,000 (19.7% of portfolio)
```

### Diversification Rules

- Max 20% allocation per stock
- Max 40% allocation per sector
- Max 10 concurrent positions
- Alert if any position exceeds 25%

## Daily Workflow

### Execution Flow

```
1. SCAN PHASE
   ├── Load paper portfolio (capital, positions)
   ├── Fetch prices for index stocks
   └── Calculate multi-factor scores

2. SIGNAL GENERATION
   ├── BUY: Score > 70, not already held
   ├── SELL: Score < 50 for holdings
   ├── STOP-LOSS: Price < stop level
   └── TARGET: Price > target level

3. POSITION SIZING
   ├── Calculate safe lot size for each BUY
   ├── Check diversification limits
   └── Skip if would exceed limits

4. EXECUTION (Paper)
   ├── Execute SELL signals first
   ├── Execute BUY signals (by score desc)
   └── Update paper portfolio

5. REPORTING
   ├── Display executed trades
   ├── Show updated portfolio
   ├── Show pending alerts
   └── Save to database
```

### Daily Run Guard

- Store last run timestamp in database
- Prevent multiple runs per day
- Use `--force` to override

## File Structure

### New Files

```
src/stockai/
├── core/
│   ├── scoring.py          # MultiFactorScorer
│   └── position_sizing.py  # PositionSizer
├── autopilot/
│   ├── __init__.py
│   ├── engine.py           # AutopilotEngine
│   ├── signals.py          # Signal generation
│   └── executor.py         # Paper execution
```

### Modified Files

```
src/stockai/cli/main.py     # Add autopilot commands
src/stockai/data/models.py  # Add database models
```

### Database Schema

```sql
-- Track autopilot runs
CREATE TABLE autopilot_runs (
    id INTEGER PRIMARY KEY,
    run_date DATE NOT NULL,
    index_scanned VARCHAR(10),
    stocks_scanned INTEGER,
    signals_generated INTEGER,
    trades_executed INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Track autopilot trades
CREATE TABLE autopilot_trades (
    id INTEGER PRIMARY KEY,
    run_id INTEGER REFERENCES autopilot_runs(id),
    symbol VARCHAR(10) NOT NULL,
    action VARCHAR(4) NOT NULL,  -- BUY/SELL
    lots INTEGER NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    score DECIMAL(5,2),
    reason VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Output Format

### autopilot run

```
═══════════════════════════════════════════════════════
🤖 AUTOPILOT RUN - Monday, 06 January 2025
   Index: JII70 | Capital: Rp 10,000,000
═══════════════════════════════════════════════════════

📊 SCANNED: 70 stocks | SIGNALS: 3 BUY, 1 SELL

🔴 SELL EXECUTED:
   BRPT: 10 lots @ Rp 1,250 (Score: 45, dropped below 50)

🟢 BUY EXECUTED:
   PWON: 58 lots @ Rp 340 (Score: 78, Value+Quality)
   TLKM: 25 lots @ Rp 3,800 (Score: 72, Momentum)

💼 PORTFOLIO SUMMARY:
   Positions: 3 | Value: Rp 9,850,000 | Cash: Rp 150,000

⚠️ ALERTS:
   PANI: 2.1% above stop-loss (monitor closely)
```

### autopilot status

```
💼 AUTOPILOT STATUS - 06 January 2025

📊 PORTFOLIO:
┌────────┬──────┬──────────┬──────────┬────────┬────────┐
│ Symbol │ Lots │ Avg Cost │ Current  │ P&L    │ Score  │
├────────┼──────┼──────────┼──────────┼────────┼────────┤
│ PWON   │ 58   │ Rp 340   │ Rp 345   │ +1.5%  │ 78     │
│ TLKM   │ 25   │ Rp 3,800 │ Rp 3,820 │ +0.5%  │ 72     │
│ BBRI   │ 20   │ Rp 4,500 │ Rp 4,480 │ -0.4%  │ 65     │
└────────┴──────┴──────────┴──────────┴────────┴────────┘

💰 SUMMARY:
   Total Value: Rp 9,850,000
   Total P&L: +Rp 45,000 (+0.46%)
   Cash: Rp 150,000

📈 PENDING SIGNALS:
   None (last run: today 08:30)
```

## Dependencies

### Existing Components Used

- `YahooFinanceSource` - Price and fundamental data
- `IDXIndexSource` - Index stock listings (IDX30, LQ45, JII70)
- `PaperPortfolio` - Paper trading execution
- Database models and initialization

### Not Included (Future Enhancements)

- Live broker integration
- Scheduled cron execution
- Push notifications/alerts
- Mobile app integration
- Backtesting framework

## Implementation Priority

1. **Phase 1: Core Scoring** - `MultiFactorScorer` class
2. **Phase 2: Position Sizing** - `PositionSizer` class
3. **Phase 3: Engine** - `AutopilotEngine` orchestration
4. **Phase 4: CLI** - `autopilot` command group
5. **Phase 5: Database** - Run/trade logging
6. **Phase 6: Testing** - Unit and integration tests
