# StockAI - UX Design Document

**Version:** 1.0.0
**Date:** 2026-01-02
**Author:** Sally (UX Designer Agent) with BMAD Team
**Status:** APPROVED (YOLO Mode)

---

## Design Philosophy

### Core Principles

1. **Conversational Intelligence** - The CLI should feel like talking to a brilliant analyst friend
2. **Progressive Disclosure** - Simple by default, detailed when needed
3. **Transparent AI** - Always show the agent's reasoning, never be a black box
4. **Empowering Data** - Present data that enables decisions, not overwhelms
5. **Indonesian Context** - Design for local market nuances and preferences

### Design Inspiration

```
Claude Code meets Bloomberg Terminal
───────────────────────────────────
Professional power, conversational ease
```

---

## CLI Experience Design

### 1. First-Run Experience (Onboarding)

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│   🤖 Welcome to StockAI!                                        │
│   Your AI-powered Indonesian stock analysis companion            │
│                                                                  │
│   ─────────────────────────────────────────────────────────     │
│                                                                  │
│   Let's set you up in 3 quick steps:                            │
│                                                                  │
│   [1/3] What's your primary focus?                              │
│         ○ IDX30 (Blue-chip, safest)                             │
│         ● LQ45 (Liquid, balanced)                               │
│         ○ All stocks (Advanced)                                 │
│                                                                  │
│   [2/3] Your investment style?                                  │
│         ○ Day trading (Technical focus)                         │
│         ● Swing trading (1-2 weeks)                             │
│         ○ Long-term (Fundamental focus)                         │
│                                                                  │
│   [3/3] Set up your AI model (requires API key):                │
│         Enter OPENAI_API_KEY: ●●●●●●●●●●●●                      │
│                                                                  │
│   ✅ Setup complete! Type 'stock help' to get started.          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2. Help & Discovery

```
$ stock help

┌─────────────────────────────────────────────────────────────────┐
│ 🤖 StockAI - AI-Powered Indonesian Stock Analysis               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│ QUICK START                                                      │
│   stock analyze BBCA      Full AI analysis with prediction       │
│   stock morning           Daily briefing for your watchlist      │
│   stock predict TLKM      Direction prediction (UP/DOWN)         │
│                                                                  │
│ RESEARCH                                                         │
│   stock info <ticker>     Company information                    │
│   stock price <ticker>    Price history and trends               │
│   stock technical <ticker> Technical indicators & signals        │
│   stock sentiment <ticker> Indonesian news sentiment             │
│                                                                  │
│ DISCOVERY                                                        │
│   stock screen            Find stocks matching criteria          │
│   stock index <code>      Show index members (IDX30, LQ45)       │
│   stock compare A B       Side-by-side comparison                │
│                                                                  │
│ PORTFOLIO                                                        │
│   stock portfolio         View your portfolio                    │
│   stock add <ticker> <qty> Add position                          │
│   stock pnl               Show profit/loss                       │
│                                                                  │
│ FLAGS                                                            │
│   -d, --detailed          Show detailed output                   │
│   -j, --json              Output as JSON                         │
│   --days <n>              Days of history (default: 30)          │
│   --model <name>          AI model (gpt-4o, claude-sonnet)       │
│                                                                  │
│ Examples:                                                        │
│   $ stock analyze BBCA                                          │
│   $ stock predict ASII --horizon 7                              │
│   $ stock screen --rsi-below 30 --sector banking                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3. Stock Information Display

```
$ stock info BBCA

┌─────────────────────────────────────────────────────────────────┐
│ 📊 BBCA - Bank Central Asia Tbk                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│ Basic Information                                                │
│ ───────────────────────────────────────────────                  │
│ Sector          Banking                                          │
│ Subsector       Private Bank                                     │
│ Market Cap      Rp 1,234.5 T                                    │
│ Listed          2000-05-31                                       │
│ Index Member    IDX30, LQ45, IDX-BUMN20                         │
│                                                                  │
│ Last Updated: 2026-01-02 09:15 WIB                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4. Price History Display

```
$ stock price BBCA -d 30

┌─────────────────────────────────────────────────────────────────┐
│ 💰 BBCA - Current Price                                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Price: Rp 9,850                                               │
│   Change: ▲ +75 (+0.77%)                                        │
│                                                                  │
│   High: Rp 9,875   │   Volume: 45.2M                            │
│   Low:  Rp 9,750   │   Value:  Rp 443.8B                        │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│ 📈 30-Day Price Chart                                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│ 10000 ┤                          ╭───╮                          │
│  9900 ┤        ╭──╮    ╭───╮    │   │    ╭──                    │
│  9800 ┤   ╭───╯  ╰────╯   ╰────╯   ╰────╯                       │
│  9700 ┤──╯                                                       │
│  9600 ┤                                                          │
│       └───────────────────────────────────────────────          │
│        Dec 03            Dec 17            Jan 02                │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│ 📋 Recent History                                               │
├─────────────────────────────────────────────────────────────────┤
│ Date       │   Open │   High │    Low │  Close │     Volume    │
│ ───────────┼────────┼────────┼────────┼────────┼────────────── │
│ 2026-01-02 │  9,775 │  9,875 │  9,750 │  9,850 │   45,234,500 │
│ 2026-01-01 │  9,800 │  9,825 │  9,725 │  9,775 │   38,123,400 │
│ 2025-12-31 │  9,750 │  9,850 │  9,700 │  9,800 │   52,456,700 │
│ 2025-12-30 │  9,825 │  9,900 │  9,750 │  9,750 │   41,234,100 │
│ 2025-12-27 │  9,700 │  9,850 │  9,675 │  9,825 │   48,567,800 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5. Technical Analysis Display

```
$ stock technical BBCA

┌─────────────────────────────────────────────────────────────────┐
│ 📊 BBCA - Technical Analysis                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │  Overall Signal: 🟢 BULLISH                                 │ │
│ │  Confidence: 67%                                            │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│ 🎯 Signal Breakdown                                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│ Indicator       │ Value    │ Signal                             │
│ ────────────────┼──────────┼─────────────────────────────────── │
│ RSI (14)        │ 42.5     │ 🟡 NEUTRAL                         │
│ MACD            │ +15.3    │ 🟢 BULLISH CROSS                   │
│ MA Cross        │ SMA50>200│ 🟢 BULLISH TREND                   │
│ Bollinger       │ Mid-band │ 🟡 WITHIN BANDS                    │
│ Stochastic      │ 35/40    │ 🟢 OVERSOLD RECOVERY               │
│ Volume          │ 1.8x avg │ 🟢 HIGH VOLUME                     │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│ 📏 Key Levels                                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│ Resistance 2:  Rp 10,100  (52-week high)                        │
│ Resistance 1:  Rp 9,950   (BB Upper)                            │
│ ─ Current ──:  Rp 9,850   ◀                                     │
│ Support 1:     Rp 9,650   (SMA20)                               │
│ Support 2:     Rp 9,400   (BB Lower)                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6. AI Analysis Display (Star Feature)

```
$ stock analyze BBCA

┌─────────────────────────────────────────────────────────────────┐
│ 🤖 StockAI Analysis: BBCA                                       │
│ Generated: 2026-01-02 14:32 WIB                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│ 💭 Agent Thinking...                                            │
│                                                                  │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Planning: Decomposing your request into research tasks...   │ │
│ │                                                             │ │
│ │ ✓ Task 1: Fetch BBCA price data (30 days)                  │ │
│ │ ✓ Task 2: Calculate technical indicators                   │ │
│ │ ✓ Task 3: Analyze Indonesian news sentiment                │ │
│ │ ✓ Task 4: Generate ML prediction                           │ │
│ │ ✓ Task 5: Synthesize comprehensive analysis                │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│ 📊 Analysis Summary                                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │                                                             │ │
│ │   🟢 PREDICTION: UP                                        │ │
│ │   Confidence: 67%  │  Horizon: 3 days                      │ │
│ │                                                             │ │
│ │   Price Target: Rp 10,050 - 10,150                         │ │
│ │   Stop Loss: Rp 9,550 (-3.0%)                              │ │
│ │                                                             │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│ KEY FINDINGS                                                     │
│ ────────────────────────────────────────────                    │
│                                                                  │
│ ✅ Technical: MACD bullish crossover confirmed, RSI recovering  │
│    from neutral zone. Volume 80% above 20-day average.          │
│                                                                  │
│ ✅ Sentiment: Indonesian news sentiment POSITIVE (0.72/1.0)     │
│    - "BCA laba bersih naik 12% YoY" (Kontan)                   │
│    - "Kredit korporasi BCA tumbuh solid" (Bisnis)              │
│                                                                  │
│ ✅ Pattern: Price bounced off SMA20 support with high volume    │
│    suggesting institutional accumulation.                        │
│                                                                  │
│ ⚠️  Risk: General market volatility ahead of BI rate decision   │
│    scheduled for January 15th.                                   │
│                                                                  │
│ MODEL CONTRIBUTIONS                                              │
│ ────────────────────────────────────────────                    │
│ XGBoost:   65% confidence → UP                                  │
│ LSTM:      70% confidence → UP                                  │
│ Sentiment: +5% modifier (positive news)                         │
│ Ensemble:  67% confidence → UP                                  │
│                                                                  │
│ TOP FEATURES (by importance)                                    │
│ ────────────────────────────────────────────                    │
│ 1. MACD Histogram Positive Crossover                            │
│ 2. Volume Ratio > 1.5x                                          │
│ 3. Price Above SMA20                                            │
│ 4. RSI Recovering from <50                                      │
│ 5. Positive News Sentiment Score                                │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│ 📝 Disclaimer: Not financial advice. AI predictions are         │
│ probabilistic. Always do your own research.                     │
└─────────────────────────────────────────────────────────────────┘
```

### 7. Morning Briefing Display

```
$ stock morning

┌─────────────────────────────────────────────────────────────────┐
│ ☀️  Good Morning, Fitra!                                        │
│ Thursday, January 2, 2026  •  08:45 WIB                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│ 📈 MARKET OVERVIEW                                              │
│ ────────────────────────────────────────────                    │
│ IHSG:    7,234.56  ▲ +0.45%  (Pre-market estimate)             │
│ IDX30:   523.12    ▲ +0.52%                                     │
│ USD/IDR: 15,890    ▼ -0.12%                                     │
│                                                                  │
│ 📌 YOUR WATCHLIST (3 stocks)                                    │
│ ────────────────────────────────────────────                    │
│                                                                  │
│ Ticker │ Last Price │ Change  │ Signal  │ Action               │
│ ───────┼────────────┼─────────┼─────────┼───────────────────── │
│ BBCA   │ Rp 9,850   │ ▲ +0.8% │ 🟢 BUY  │ Near support, +vol   │
│ TLKM   │ Rp 3,420   │ ▼ -0.3% │ 🟡 HOLD │ Consolidating        │
│ ASII   │ Rp 5,125   │ ▲ +1.2% │ 🟢 BUY  │ Breakout attempt     │
│                                                                  │
│ 🔔 ALERTS                                                       │
│ ────────────────────────────────────────────                    │
│ ⚠️  BBRI: RSI crossed below 30 (oversold)                       │
│ ✅ ASII: Price broke above resistance Rp 5,100                  │
│                                                                  │
│ 📰 TOP NEWS (Sentiment Impact)                                  │
│ ────────────────────────────────────────────                    │
│ 🟢 "Bank sentral pertahankan suku bunga" - Neutral to Positive │
│ 🟢 "Ekonomi RI tumbuh 5.1% di Q4" - Positive for market        │
│ 🔴 "Rupiah melemah tipis terhadap USD" - Watch export stocks   │
│                                                                  │
│ 💡 TODAY'S AI INSIGHT                                           │
│ ────────────────────────────────────────────                    │
│ "Banking sector showing strength with MACD bullish crosses      │
│ across BBCA, BMRI, BBRI. Consider accumulation if market       │
│ confirms above IHSG 7,200 support level."                       │
│                                                                  │
│ Run 'stock analyze <ticker>' for deep analysis                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 8. Prediction Display

```
$ stock predict BBCA --horizon 7

┌─────────────────────────────────────────────────────────────────┐
│ 🎯 BBCA - 7-Day Direction Prediction                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │                                                             │ │
│ │         ███████████████████████████                        │ │
│ │         ███████████████████████████                        │ │
│ │         ███████████████████████████  67%                   │ │
│ │         ███████████████████████████                        │ │
│ │         ███████████████████████████                        │ │
│ │                                                             │ │
│ │            🟢 UP (67% confidence)                          │ │
│ │                                                             │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│ Prediction Details                                               │
│ ────────────────────────────────────────────                    │
│                                                                  │
│ Current Price:     Rp 9,850                                     │
│ Expected Range:    Rp 9,950 - Rp 10,200                        │
│ Target (Mid):      Rp 10,075 (+2.3%)                           │
│ Risk/Reward:       1:2.3 (favorable)                            │
│                                                                  │
│ Suggested Stop:    Rp 9,550 (-3.0%)                            │
│ Suggested Target:  Rp 10,150 (+3.0%)                           │
│                                                                  │
│ Model Breakdown                                                  │
│ ────────────────────────────────────────────                    │
│                                                                  │
│ XGBoost         [██████████████████░░░░░░░░] 70% UP            │
│ LSTM            [████████████████░░░░░░░░░░] 65% UP            │
│ Sentiment       [██████████████████████░░░░] 85% Positive      │
│                                                                  │
│ Historical Accuracy (this model + horizon)                       │
│ ────────────────────────────────────────────                    │
│ Last 30 predictions: 62% correct                                │
│ Last 10 predictions: 70% correct                                │
│ Confidence calibration: Good (predictions at 67% hit 65%)       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 9. Agent Reasoning Display (Transparent AI)

```
$ stock analyze TLKM --verbose

┌─────────────────────────────────────────────────────────────────┐
│ 🤖 Agent Execution Log                                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│ [14:32:01] 🧠 PLANNING AGENT                                    │
│ ├─ Query: "Analyze TLKM for investment decision"                │
│ ├─ Decomposed into 5 research tasks:                            │
│ │   1. Fetch historical prices (30d)                            │
│ │   2. Calculate technical indicators                           │
│ │   3. Retrieve fundamental metrics                             │
│ │   4. Analyze Indonesian news sentiment                        │
│ │   5. Generate ensemble prediction                             │
│ └─ Estimated time: 25 seconds                                   │
│                                                                  │
│ [14:32:03] 🔧 ACTION AGENT - Task 1                             │
│ ├─ Tool: get_stock_price("TLKM", days=30)                       │
│ ├─ Status: ✅ Success                                           │
│ └─ Result: 30 price records fetched                             │
│                                                                  │
│ [14:32:05] 🔧 ACTION AGENT - Task 2                             │
│ ├─ Tool: calculate_technical_indicators("TLKM")                 │
│ ├─ Status: ✅ Success                                           │
│ └─ Result: 12 indicators calculated                             │
│                                                                  │
│ [14:32:08] 🔧 ACTION AGENT - Task 3                             │
│ ├─ Tool: get_fundamentals("TLKM")                               │
│ ├─ Status: ✅ Success                                           │
│ └─ Result: PER=15.2, PBV=2.8, ROE=18.5%                        │
│                                                                  │
│ [14:32:12] 🔧 ACTION AGENT - Task 4                             │
│ ├─ Tool: analyze_sentiment("TLKM")                              │
│ ├─ Status: ✅ Success                                           │
│ └─ Result: Sentiment=0.62 (Slightly Positive)                   │
│                                                                  │
│ [14:32:18] 🔧 ACTION AGENT - Task 5                             │
│ ├─ Tool: predict_direction("TLKM", horizon=3)                   │
│ ├─ Status: ✅ Success                                           │
│ └─ Result: UP (58% confidence)                                  │
│                                                                  │
│ [14:32:20] ✓ VALIDATION AGENT                                   │
│ ├─ Checking task completion...                                  │
│ ├─ All 5 tasks completed successfully                           │
│ ├─ Data quality: GOOD                                           │
│ └─ Proceeding to synthesis                                      │
│                                                                  │
│ [14:32:22] 📝 ANSWER AGENT                                      │
│ ├─ Synthesizing findings...                                     │
│ └─ Analysis complete                                            │
│                                                                  │
│ Total execution time: 21 seconds                                │
│ Tools invoked: 5                                                │
│ LLM calls: 7                                                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Color System

### Signal Colors

| Signal | Color | Hex | Usage |
|--------|-------|-----|-------|
| Bullish/Buy | Green | `#00FF00` | Positive signals, up movement |
| Bearish/Sell | Red | `#FF0000` | Negative signals, down movement |
| Neutral/Hold | Yellow | `#FFFF00` | Neutral signals, sideways |
| Info | Cyan | `#00FFFF` | Informational elements |
| Warning | Orange | `#FFA500` | Alerts, caution |
| Muted | Gray | `#808080` | Secondary information |

### Rich Console Theme

```python
STOCKAI_THEME = {
    "info": "cyan",
    "success": "green",
    "warning": "yellow",
    "error": "red",
    "price_up": "green",
    "price_down": "red",
    "neutral": "yellow",
    "muted": "dim",
    "header": "bold cyan",
    "value": "white",
}
```

---

## Interaction Patterns

### 1. Loading States

```
┌─────────────────────────────────────────────────────────────────┐
│ 🔄 Analyzing BBCA...                                            │
│                                                                  │
│ [████████████░░░░░░░░░░░░░░░░░░] 40%                            │
│                                                                  │
│ ✓ Fetching price data                                           │
│ ✓ Calculating indicators                                        │
│ ◐ Analyzing sentiment...                                        │
│ ○ Generating prediction                                         │
│ ○ Synthesizing analysis                                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2. Error States

```
┌─────────────────────────────────────────────────────────────────┐
│ ❌ Error: Unable to fetch data for XXXX                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│ The ticker 'XXXX' was not found on IDX.                         │
│                                                                  │
│ Did you mean:                                                    │
│   • XXYY - PT Example Corp Tbk                                  │
│   • XXYZ - PT Another Corp Tbk                                  │
│                                                                  │
│ Run 'stock index LQ45' to see available tickers.                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3. Confirmation Prompts

```
┌─────────────────────────────────────────────────────────────────┐
│ ⚠️  Add GOTO to portfolio?                                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│ Ticker: GOTO                                                     │
│ Quantity: 10,000 shares                                          │
│ Price: Rp 85 (current market)                                   │
│ Total Value: Rp 850,000                                         │
│                                                                  │
│ ⚠️  Note: GOTO has HIGH volatility (ATR 8.5%)                   │
│                                                                  │
│ [Y] Confirm    [N] Cancel    [?] More info                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Web Report Design (Post-MVP)

### Report Layout

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│ ┌──────────────────────────────────────────────────────────────────────────────┐ │
│ │ 🤖 StockAI Analysis Report                                                   │ │
│ │ BBCA - Bank Central Asia Tbk                                                 │ │
│ │ Generated: January 2, 2026 at 14:32 WIB                                     │ │
│ └──────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
│ ┌────────────────────────┐ ┌────────────────────────────────────────────────────┐│
│ │ PREDICTION             │ │ PRICE CHART (Interactive)                          ││
│ │                        │ │                                                    ││
│ │    🟢 UP               │ │  [==============================]                 ││
│ │    67%                 │ │  Plotly candlestick chart                         ││
│ │                        │ │  with technical overlays                          ││
│ │ Target: Rp 10,050     │ │                                                    ││
│ │ Stop:   Rp 9,550      │ │                                                    ││
│ │                        │ │                                                    ││
│ └────────────────────────┘ └────────────────────────────────────────────────────┘│
│                                                                                  │
│ ┌─────────────────────────────────────┐ ┌──────────────────────────────────────┐ │
│ │ TECHNICAL SIGNALS                   │ │ SENTIMENT ANALYSIS                   │ │
│ │                                     │ │                                      │ │
│ │ RSI:      42.5  🟡 Neutral         │ │ Score: 0.72 / 1.0  🟢               │ │
│ │ MACD:     +15.3 🟢 Bullish Cross   │ │                                      │ │
│ │ MA Cross: 🟢 Bullish Trend         │ │ Recent Headlines:                    │ │
│ │ Volume:   1.8x  🟢 High            │ │ • "BCA laba naik 12%..."            │ │
│ │                                     │ │ • "Kredit korporasi tumbuh..."      │ │
│ └─────────────────────────────────────┘ └──────────────────────────────────────┘ │
│                                                                                  │
│ ┌──────────────────────────────────────────────────────────────────────────────┐ │
│ │ AI REASONING                                                                 │ │
│ │                                                                              │ │
│ │ The analysis indicates a bullish setup for BBCA based on:                   │ │
│ │ 1. Technical momentum with MACD bullish crossover...                        │ │
│ │ 2. Positive sentiment from recent earnings news...                          │ │
│ │ 3. Strong volume suggesting institutional interest...                       │ │
│ │                                                                              │ │
│ └──────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
│ ┌──────────────────────────────────────────────────────────────────────────────┐ │
│ │ [Download PDF]  [Share Report]  [Save to Portfolio]                         │ │
│ └──────────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Accessibility Considerations

1. **Screen Reader Support** - All ASCII art has text alternatives
2. **Color Blind Friendly** - Icons (▲▼●○) supplement color signals
3. **High Contrast** - Works in both light and dark terminals
4. **Keyboard Navigation** - All interactive elements accessible

---

## Mobile/Responsive (Web)

For web reports, design breakpoints:

| Breakpoint | Layout |
|------------|--------|
| Desktop (>1200px) | Full two-column layout |
| Tablet (768-1200px) | Stacked cards, smaller charts |
| Mobile (<768px) | Single column, collapsible sections |

---

**Document Status:** APPROVED
**Next Step:** Epics & Stories
