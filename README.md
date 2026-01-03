# StockAI

AI-Powered Indonesian Stock Analysis CLI - Think "Claude Code for IDX investing."

## Features

- 🤖 **AI Research Agent** - Autonomous analysis using LangChain DeepAgents
- 📊 **Technical Analysis** - RSI, MACD, Bollinger Bands, and more
- 🔮 **ML Predictions** - XGBoost + LSTM ensemble for UP/DOWN signals
- 💬 **Sentiment Analysis** - IndoBERT for Indonesian news/social media
- 💼 **Portfolio Tracking** - Manage and monitor your investments

## Installation

```bash
pip install stockai
```

## Quick Start

```bash
# Analyze a stock
stock analyze BBCA

# Get prediction
stock predict TLKM --days 7

# View price history
stock history BBRI --period 3mo

# Manage portfolio
stock portfolio add BBCA 100 9500
stock portfolio list
```

## Configuration

Copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
```

Required:
- `GOOGLE_API_KEY` - For Gemini AI models

Optional:
- `FIRECRAWL_API_KEY` - For deep web research
- `OPENAI_API_KEY` - Alternative LLM
- `ANTHROPIC_API_KEY` - Alternative LLM

## License

MIT
