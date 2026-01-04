# StockAI

AI-Powered Indonesian Stock Analysis CLI - Your personal hedge fund toolkit for IDX investing.

Designed for **passive investors** with:
- 15 minutes/day
- Small capital (< Rp 5 juta)
- Systematic, data-driven approach

## Features

### Core Analysis
- **Multi-Agent Trading System** - 7 specialized AI agents (Analyst, Researcher, Risk Manager, etc.)
- **Technical Analysis** - RSI, MACD, Bollinger Bands, EMA crossovers
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
pip install stockai
```

Or from source:
```bash
git clone https://github.com/rebekz/stockai.git
cd stockai
pip install -e .
```

## Quick Start

### Daily Routine (15 min)

```bash
# Morning (before 9:00 WIB) - Check alerts
python -m stockai morning

# Evening (after 16:00 WIB) - Review day
python -m stockai evening
```

### Weekly Analysis

```bash
# Performance review
python -m stockai weekly

# Find top opportunities
python -m stockai score rank --top 5

# Scan market with AI agents
python -m stockai agents scan --top 10
```

### Before Buying

```bash
# Calculate safe position size (2% risk rule)
python -m stockai risk position --capital 5000000 --price 4500 --stop-pct 7

# Analyze stock score
python -m stockai score stock BBCA

# Check portfolio diversification
python -m stockai risk diversification
```

### Paper Trading (Practice First!)

```bash
# Start with virtual capital
python -m stockai paper reset --capital 5000000

# Buy 1 lot
python -m stockai paper buy BBCA 1

# View portfolio
python -m stockai paper view

# Sell
python -m stockai paper sell BBCA 1
```

### Learning

```bash
# Start tutorial
python -m stockai learn start

# Take quiz
python -m stockai learn quiz
```

## All Commands

| Category | Command | Description |
|----------|---------|-------------|
| **Briefings** | `morning` | Morning briefing (pre-market) |
| | `evening` | Evening briefing (post-market) |
| | `weekly` | Weekly performance review |
| **Scoring** | `score stock SYMBOL` | Multi-factor score analysis |
| | `score rank` | Rank stocks by composite score |
| **Risk** | `risk position` | Position size calculator (2% rule) |
| | `risk diversification` | Check portfolio limits |
| | `risk portfolio` | Portfolio risk metrics (VaR, Sharpe) |
| **AI Agents** | `agents scan` | Scan market for opportunities |
| | `agents recommend` | Portfolio recommendations |
| | `agents daily` | Daily trading recommendations |
| | `agents signal` | Quick trading signals |
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
| | `auto schedule` | Automated scanning |

## Capital Allocation Guide

For small capital (< Rp 5 juta), focus on 3-5 stocks:

```
Rp 5,000,000 total
├── Stock 1: Rp 1,000,000 (20%)
├── Stock 2: Rp 1,000,000 (20%)
├── Stock 3: Rp 1,000,000 (20%)
├── Stock 4: Rp 1,000,000 (20%)
└── Cash Reserve: Rp 1,000,000 (20%)
```

**Risk Management Rules:**
- Never risk more than 2% of capital per trade
- Max 20% in any single stock
- Max 40% in any sector
- Always use stop-losses (typically 5-8% below entry)

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
- 60-69: Fair (Hold)
- 50-59: Poor (Sell)
- Below 50: Very Poor (Strong Sell)

## Architecture

```
stockai/
├── agents/          # Multi-agent trading system
│   ├── orchestrator.py
│   ├── subagents.py
│   └── tools.py
├── scoring/         # Multi-factor scoring
│   ├── factors.py
│   ├── screener.py
│   └── signals.py
├── risk/            # Risk management
│   ├── position_sizing.py
│   ├── diversification.py
│   └── portfolio_risk.py
├── briefing/        # Daily/weekly briefings
│   ├── daily.py
│   └── weekly.py
├── tutorial/        # Learning system
│   ├── lessons.py
│   ├── quiz.py
│   └── paper_trading.py
├── automation/      # Scheduled tasks
│   ├── scheduler.py
│   ├── runner.py
│   └── notifier.py
└── cli/             # Command-line interface
    └── main.py
```

## Disclaimer

This tool is for educational and research purposes only. Stock investments involve risk. Past performance does not guarantee future results. Always do your own research before making investment decisions.

## License

MIT
