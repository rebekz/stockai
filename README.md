# StockAI

AI-Powered Indonesian Stock Analysis CLI - Your personal hedge fund toolkit for IDX investing.

Designed for **passive investors** with:
- 15 minutes/day
- Small capital (< Rp 10 juta)
- Systematic, data-driven approach

## Features

### Quality Over Quantity System (NEW)
- **6-Gate Decision Filter** - Only trade when ALL quality gates pass
- **Smart Money Score** - Track institutional accumulation via OBV, MFI, volume
- **Support/Resistance Detection** - Automated pivot point analysis
- **Trade Plan Generation** - Entry range, stop-loss, take-profit levels
- **3-Agent AI Validation** - Technical, Fundamental, Risk Manager review

### Core Analysis
- **Multi-Agent Trading System** - 7 specialized AI agents (Analyst, Researcher, Risk Manager, etc.)
- **Technical Analysis** - RSI, MACD, Bollinger Bands, EMA crossovers, ADX
- **Sentiment Analysis** - Gemini-powered Indonesian news analysis

### Quantitative Tools (Hedge Fund Style)
- **Multi-Factor Scoring** - Value (25%), Quality (30%), Momentum (25%), Volatility (20%)
- **Position Sizing** - Professional 2% risk rule
- **Diversification Checks** - Max 20%/stock, 40%/sector limits
- **Portfolio Risk Metrics** - VaR, Sharpe ratio, beta, max drawdown

### Daily Workflow
- **Morning Briefing** - Pre-market alerts, stop-loss warnings
- **Evening Briefing** - Daily P&L, score changes, tomorrow's focus
- **Weekly Review** - Performance vs IHSG, win rate, lessons learned

### Practice & Learning
- **Paper Trading** - Risk-free practice with virtual capital
- **Interactive Tutorials** - 8 lessons on Indonesian stock market basics
- **Quizzes** - Test your knowledge before risking real money

## Installation

```bash
# Using uv (recommended)
git clone https://github.com/rebekz/stockai.git
cd stockai
uv sync

# Or using pip
pip install -e .
```

## Quick Start

```bash
# Verify installation
uv run stockai --version

# Show all commands
uv run stockai --help
```

---

## Passive Trader Workflow (15 Minutes/Day)

Perfect for busy professionals with small capital (Rp 5-10 juta) who want systematic, stress-free investing.

### Daily Routine (5-10 min)

```bash
# Morning Check (before 9:00 WIB) - 3 min
uv run stockai morning

# Evening Review (after 16:00 WIB) - 5 min
uv run stockai evening
```

### Weekend Analysis (30 min once/week)

```bash
# 1. Weekly performance review
uv run stockai weekly

# 2. Find quality opportunities with 6-gate filter
uv run stockai quality BBCA
uv run stockai quality BBRI
uv run stockai quality TLKM

# 3. Run autopilot scan (finds candidates automatically)
uv run stockai autopilot --dry-run --limit 20
```

### Before Buying (5 min per stock)

```bash
# Full quality analysis with trade plan
uv run stockai quality BBCA --capital 5000000

# Example output:
# Gate Status: 5/6 PASSED (WATCH)
# Smart Money: 3.2 (ACCUMULATION)
# Entry Range: Rp 9,850 - Rp 10,050
# Stop Loss: Rp 9,550 (-4.5%)
# Take Profit 1: Rp 10,500 (+5%)
# Risk/Reward: 1:1.5
# Position Size: 1 lot (Rp 1,005,000)
```

### Monthly Rebalancing (15 min)

```bash
# Check portfolio diversification
uv run stockai risk diversification

# Review portfolio risk metrics
uv run stockai risk portfolio

# Score your current holdings
uv run stockai score stock BBCA
```

### Sample Weekly Schedule

| Day | Time | Task | Command |
|-----|------|------|---------|
| Mon-Fri | 08:45 | Morning check | `uv run stockai morning` |
| Mon-Fri | 16:15 | Evening review | `uv run stockai evening` |
| Saturday | 10:00 | Weekly analysis | `uv run stockai weekly` |
| Saturday | 10:15 | Find opportunities | `uv run stockai autopilot --dry-run` |
| Saturday | 10:30 | Quality check top picks | `uv run stockai quality SYMBOL` |

---

## Quality Analysis Command

The `quality` command performs comprehensive analysis using the Quality Over Quantity system:

```bash
# Basic analysis
uv run stockai quality BBCA

# With custom capital for position sizing
uv run stockai quality BBCA --capital 10000000

# With AI validation (requires API key)
uv run stockai quality BBCA --ai

# Verbose output
uv run stockai quality BBCA --verbose
```

### 6-Gate Decision Filter

Only trade when ALL gates pass:

| Gate | Threshold | Purpose |
|------|-----------|---------|
| Overall Score | >= 70 | Composite quality check |
| Technical Score | >= 60 | Entry timing validation |
| Smart Money Score | >= 3.0 | Institutional accumulation |
| Distance to Support | <= 5% | Risk management |
| ADX Trend Strength | >= 20 | Trend confirmation |
| Fundamental Score | >= 60 | Financial health |

### Confidence Levels

- **HIGH** (6/6 gates pass) - Execute trade
- **WATCH** (4-5 gates pass, score >= 60) - Monitor for improvement
- **REJECTED** (< 4 gates or score < 60) - Skip this opportunity

---

## All Commands

| Category | Command | Description |
|----------|---------|-------------|
| **Quality** | `quality SYMBOL` | Full 6-gate analysis with trade plan |
| **Briefings** | `morning` | Morning briefing (pre-market) |
| | `evening` | Evening briefing (post-market) |
| | `weekly` | Weekly performance review |
| **Autopilot** | `autopilot` | Automated daily trading system |
| | `autopilot --dry-run` | Preview without trading |
| **Scoring** | `score stock SYMBOL` | Multi-factor score analysis |
| | `score rank` | Rank stocks by composite score |
| **Risk** | `risk position` | Position size calculator (2% rule) |
| | `risk diversification` | Check portfolio limits |
| | `risk portfolio` | Portfolio risk metrics (VaR, Sharpe) |
| **AI Agents** | `agents scan` | Scan market for opportunities |
| | `agents recommend` | Portfolio recommendations |
| | `agents daily` | Daily trading recommendations |
| **Analysis** | `analyze SYMBOL` | AI-powered stock analysis |
| | `sentiment SYMBOL` | News sentiment analysis |
| | `info SYMBOL` | Stock information |
| | `history SYMBOL` | Price history |
| **Portfolio** | `portfolio list` | View holdings |
| | `portfolio add` | Add position |
| | `portfolio sell` | Sell position |
| **Paper Trading** | `paper buy` | Virtual buy |
| | `paper sell` | Virtual sell |
| | `paper view` | View paper portfolio |
| | `paper reset` | Reset paper account |
| **Learning** | `learn start` | Begin tutorials |
| | `learn quiz` | Take quiz |
| **Other** | `list` | List stocks in index |
| | `suggest` | Technical buy signals |
| | `web` | Start web dashboard |

---

## Capital Allocation Guide

### Small Capital (Rp 5-10 juta)

Focus on 3-5 high-quality stocks:

```
Rp 10,000,000 total
├── Stock 1: Rp 2,000,000 (20%) - Blue chip
├── Stock 2: Rp 2,000,000 (20%) - Blue chip
├── Stock 3: Rp 2,000,000 (20%) - Growth
├── Stock 4: Rp 2,000,000 (20%) - Dividend
└── Cash Reserve: Rp 2,000,000 (20%) - For opportunities
```

### Risk Management Rules

- Never risk more than 2% of capital per trade
- Max 20% in any single stock
- Max 40% in any sector
- Always use stop-losses (typically 5-8% below entry)
- Only buy when 6-gate filter passes

---

## Multi-Factor Scoring System

The scoring system evaluates stocks using hedge fund methodology:

| Factor | Weight | Metrics |
|--------|--------|---------|
| **Value** | 25% | P/E ratio, P/B ratio vs sector |
| **Quality** | 30% | ROE, debt-to-equity, profit margins |
| **Momentum** | 25% | 6-month returns, trend strength |
| **Volatility** | 20% | Beta, standard deviation (lower = safer) |

**Score Interpretation:**
- 80-100: Excellent (Strong Buy)
- 70-79: Good (Buy)
- 60-69: Fair (Hold/Watch)
- 50-59: Poor (Sell)
- Below 50: Very Poor (Strong Sell)

---

## Configuration

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

**Required:**
- `GOOGLE_API_KEY` - For Gemini AI models

**Optional:**
- `FIRECRAWL_API_KEY` - For deep web research
- `TELEGRAM_BOT_TOKEN` - For trading alerts
- `TELEGRAM_CHAT_ID` - Your Telegram chat ID

---

## Architecture

```
stockai/
├── agents/          # Multi-agent trading system
│   ├── orchestrator.py
│   ├── focused_validator.py  # 3-agent validation
│   ├── focused_prompts.py
│   └── tools.py
├── scoring/         # Multi-factor scoring & gates
│   ├── analyzer.py      # Integrated analysis
│   ├── factors.py       # Value/Quality/Momentum/Volatility
│   ├── gates.py         # 6-gate decision filter
│   ├── smart_money.py   # OBV/MFI/Volume analysis
│   ├── support_resistance.py
│   └── trade_plan.py    # Entry/SL/TP generation
├── risk/            # Risk management
│   ├── position_sizing.py
│   ├── diversification.py
│   └── portfolio_risk.py
├── briefing/        # Daily/weekly briefings
│   ├── daily.py
│   └── weekly.py
├── autopilot/       # Automated trading system
│   └── engine.py
├── tutorial/        # Learning system
│   ├── lessons.py
│   ├── quiz.py
│   └── paper_trading.py
└── cli/             # Command-line interface
    └── main.py
```

---

## Disclaimer

This tool is for educational and research purposes only. Stock investments involve risk. Past performance does not guarantee future results. Always do your own research before making investment decisions.

## License

MIT
