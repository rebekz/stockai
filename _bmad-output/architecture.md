# StockAI - System Architecture Document

**Version:** 1.0.0
**Date:** 2026-01-02
**Author:** Winston (Architect Agent) with BMAD Team
**Status:** APPROVED (YOLO Mode)

---

## Architecture Overview

StockAI follows a **multi-agent architecture** inspired by Dexter and Kai-Code, implemented using **LangChain DeepAgents** framework. The system is designed as a CLI-first application with optional web report generation.

```
┌─────────────────────────────────────────────────────────────────┐
│                        StockAI CLI                               │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                   Typer + Rich Console                    │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              LangChain DeepAgent Orchestrator             │    │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐        │    │
│  │  │Planning │→│ Action  │→│Validate │→│ Answer  │        │    │
│  │  │ Agent   │ │ Agent   │ │ Agent   │ │ Agent   │        │    │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘        │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                     Tool Layer                            │   │
│  │ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐     │   │
│  │ │Data Tools│ │TA Tools  │ │ML Tools  │ │NLP Tools │     │   │
│  │ └──────────┘ └──────────┘ └──────────┘ └──────────┘     │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   Data Layer                              │   │
│  │ ┌────────────┐ ┌────────────┐ ┌────────────┐            │   │
│  │ │ SQLite DB  │ │ DuckDB     │ │ File Cache │            │   │
│  │ └────────────┘ └────────────┘ └────────────┘            │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## System Components

### 1. CLI Layer (`stockai/cli/`)

**Technology:** Typer + Rich

```
stockai/cli/
├── __init__.py
├── main.py              # Entry point, Typer app
├── commands/
│   ├── __init__.py
│   ├── stock.py         # stock info, price, technical
│   ├── analyze.py       # stock analyze, predict
│   ├── portfolio.py     # portfolio management
│   ├── briefing.py      # morning, evening briefings
│   └── web.py           # web serve command
└── ui/
    ├── __init__.py
    ├── tables.py        # Rich table formatters
    ├── panels.py        # Rich panel components
    └── charts.py        # ASCII chart rendering
```

**Key Design Decisions:**
- Single entry point via `stock` command
- Subcommands follow natural language patterns
- Rich console for beautiful terminal output
- Progressive disclosure (simple → detailed with flags)

### 2. Agent Layer (`stockai/agent/`)

**Technology:** LangChain DeepAgents (from kai-code-1 pattern)

```
stockai/agent/
├── __init__.py
├── orchestrator.py      # Main DeepAgent orchestrator
├── config.py            # Agent configuration
├── state.py             # Conversation state management
├── phases/
│   ├── __init__.py
│   ├── planning.py      # Task decomposition
│   ├── action.py        # Tool execution
│   ├── validation.py    # Result verification
│   └── answer.py        # Response synthesis
├── prompts/
│   ├── __init__.py
│   ├── system.py        # System prompts
│   ├── planning.py      # Planning prompts
│   └── analysis.py      # Domain-specific prompts
└── memory/
    ├── __init__.py
    ├── session.py       # Session persistence
    └── checkpointer.py  # LangGraph checkpointing
```

**Agent Architecture (Dexter-inspired):**

```python
from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model

class StockAIAgent:
    """Autonomous financial research agent."""

    def __init__(self, model: str = "openai:gpt-4o"):
        self.model = init_chat_model(model)
        self.tools = self._build_tools()
        self.graph = create_deep_agent(
            model=self.model,
            tools=self.tools,
            system_prompt=STOCKAI_SYSTEM_PROMPT,
            backend=StockAIBackend(),
            interrupt_on={"execute": False},  # YOLO mode
        )

    def analyze(self, query: str) -> AnalysisResult:
        """Run autonomous analysis on user query."""
        state = self.graph.invoke({"messages": [{"role": "user", "content": query}]})
        return self._parse_result(state)
```

**Phase Flow:**

```
User Query → Planning Agent → Task List
                   ↓
              Action Agent → Tool Execution
                   ↓
            Validation Agent → Quality Check
                   ↓ (loop if needed)
              Answer Agent → Final Response
```

### 3. Tool Layer (`stockai/tools/`)

**Technology:** LangChain Tools

```
stockai/tools/
├── __init__.py
├── registry.py          # Tool registration
├── data/
│   ├── __init__.py
│   ├── yahoo.py         # Yahoo Finance connector
│   ├── sectors.py       # Sectors.app API
│   ├── idx.py           # IDX index data
│   └── news.py          # News RSS feeds
├── technical/
│   ├── __init__.py
│   ├── indicators.py    # TA calculations
│   ├── patterns.py      # Chart patterns
│   └── signals.py       # Signal generation
├── fundamental/
│   ├── __init__.py
│   ├── ratios.py        # Financial ratios
│   └── valuation.py     # Valuation models
├── ml/
│   ├── __init__.py
│   ├── features.py      # Feature engineering
│   ├── predictor.py     # XGBoost/LSTM models
│   └── ensemble.py      # Ensemble predictions
└── nlp/
    ├── __init__.py
    ├── sentiment.py     # IndoBERT sentiment
    └── summarizer.py    # Text summarization
```

**Tool Definition Pattern:**

```python
from langchain_core.tools import tool

@tool("get_stock_price")
def get_stock_price(ticker: str, days: int = 30) -> str:
    """Fetch historical OHLCV data for an Indonesian stock.

    Args:
        ticker: Stock ticker (e.g., 'BBCA', 'TLKM')
        days: Number of days of history (default: 30)

    Returns:
        JSON string with price data
    """
    connector = YahooFinanceConnector()
    df = connector.fetch_prices(ticker, period=f"{days}d")
    return df.to_json(orient='records')

@tool("calculate_technical_indicators")
def calculate_technical_indicators(ticker: str) -> str:
    """Calculate technical indicators for a stock.

    Args:
        ticker: Stock ticker

    Returns:
        JSON with RSI, MACD, Bollinger, signals
    """
    analyzer = TechnicalAnalyzer(ticker)
    return json.dumps(analyzer.get_summary())

@tool("predict_direction")
def predict_direction(ticker: str, horizon: int = 3) -> str:
    """Predict price direction using ML ensemble.

    Args:
        ticker: Stock ticker
        horizon: Prediction horizon in days (1, 3, 7)

    Returns:
        JSON with direction (UP/DOWN), confidence, features
    """
    predictor = DirectionPredictor()
    return json.dumps(predictor.predict(ticker, horizon))

@tool("analyze_sentiment")
def analyze_sentiment(ticker: str) -> str:
    """Analyze Indonesian news sentiment for a stock.

    Args:
        ticker: Stock ticker

    Returns:
        JSON with sentiment score, headlines, sources
    """
    analyzer = IndoBERTSentiment()
    return json.dumps(analyzer.analyze(ticker))
```

### 4. Core Domain Layer (`stockai/core/`)

```
stockai/core/
├── __init__.py
├── models/
│   ├── __init__.py
│   ├── stock.py         # Stock data models
│   ├── price.py         # Price data models
│   ├── prediction.py    # Prediction models
│   └── portfolio.py     # Portfolio models
├── technical/
│   ├── __init__.py
│   ├── indicators.py    # TA-Lib wrappers
│   └── patterns.py      # Pattern recognition
├── fundamental/
│   ├── __init__.py
│   └── ratios.py        # Financial calculations
├── predictor/
│   ├── __init__.py
│   ├── features.py      # Feature engineering (~105 features)
│   ├── xgboost_model.py # XGBoost classifier
│   ├── lstm_model.py    # LSTM sequence model
│   └── ensemble.py      # Model ensemble
└── sentiment/
    ├── __init__.py
    ├── indobert.py      # IndoBERT wrapper
    └── news_fetcher.py  # News aggregation
```

### 5. Data Layer (`stockai/data/`)

```
stockai/data/
├── __init__.py
├── sources/
│   ├── __init__.py
│   ├── yahoo.py         # Yahoo Finance
│   ├── sectors.py       # Sectors.app
│   └── idx.py           # IDX constituents
├── storage/
│   ├── __init__.py
│   ├── database.py      # SQLAlchemy models
│   ├── cache.py         # File-based cache
│   └── migrations.py    # Schema migrations
└── processing/
    ├── __init__.py
    ├── cleaner.py       # Data cleaning
    └── transformer.py   # Data transformations
```

**Database Schema:**

```sql
-- Core tables
CREATE TABLE stocks (
    ticker VARCHAR(10) PRIMARY KEY,
    name VARCHAR(255),
    sector VARCHAR(100),
    market_cap FLOAT,
    updated_at TIMESTAMP
);

CREATE TABLE prices (
    id INTEGER PRIMARY KEY,
    ticker VARCHAR(10) REFERENCES stocks(ticker),
    date DATE,
    open FLOAT, high FLOAT, low FLOAT, close FLOAT,
    volume FLOAT
);

CREATE TABLE predictions (
    id INTEGER PRIMARY KEY,
    ticker VARCHAR(10) REFERENCES stocks(ticker),
    date DATE,
    horizon INTEGER,
    direction VARCHAR(10),  -- UP, DOWN
    confidence FLOAT,
    features_json TEXT,
    actual_direction VARCHAR(10),  -- For validation
    created_at TIMESTAMP
);

CREATE TABLE portfolio (
    id INTEGER PRIMARY KEY,
    ticker VARCHAR(10) REFERENCES stocks(ticker),
    quantity INTEGER,
    avg_cost FLOAT,
    added_date DATE
);

-- Index tables
CREATE TABLE index_members (
    id INTEGER PRIMARY KEY,
    index_code VARCHAR(20),  -- IDX30, LQ45
    ticker VARCHAR(10),
    effective_date DATE
);
```

### 6. Web Layer (`stockai/web/`) [Post-MVP]

```
stockai/web/
├── __init__.py
├── app.py               # FastAPI app
├── routers/
│   ├── __init__.py
│   ├── analysis.py      # Analysis endpoints
│   └── reports.py       # Report generation
├── templates/
│   ├── base.html
│   ├── report.html
│   └── dashboard.html
└── static/
    ├── css/
    └── js/
```

---

## Technology Stack

### Core Dependencies

```toml
# pyproject.toml
[project]
name = "stockai"
version = "0.1.0"
requires-python = ">=3.11"

dependencies = [
    # CLI
    "typer[all]>=0.9.0",
    "rich>=13.0.0",

    # Agent Framework
    "langchain>=0.3.0",
    "langchain-openai>=0.2.0",
    "langchain-anthropic>=0.2.0",
    "langgraph>=0.2.0",
    "deepagents>=0.1.0",

    # Data
    "yfinance>=0.2.36",
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "requests>=2.31.0",
    "beautifulsoup4>=4.12.0",

    # Database
    "sqlalchemy>=2.0.0",
    "duckdb>=0.9.0",

    # Technical Analysis
    "pandas-ta>=0.3.14b",

    # ML
    "scikit-learn>=1.3.0",
    "xgboost>=2.0.0",
    "torch>=2.1.0",
    "transformers>=4.36.0",

    # Config
    "pydantic>=2.5.0",
    "pydantic-settings>=2.0.0",
    "tomli>=2.0.0",
]

[project.optional-dependencies]
web = [
    "fastapi>=0.109.0",
    "uvicorn>=0.25.0",
    "jinja2>=3.1.0",
    "plotly>=5.18.0",
]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "ruff>=0.1.0",
    "mypy>=1.0.0",
]

[project.scripts]
stock = "stockai.cli.main:app"
```

### Model Selection

| Model | Use Case | Provider |
|-------|----------|----------|
| GPT-4o | Primary agent reasoning | OpenAI |
| Claude Sonnet 4 | Alternative reasoning | Anthropic |
| IndoBERT | Indonesian sentiment | HuggingFace |
| XGBoost | Quick predictions | Local |
| LSTM | Sequence modeling | Local (PyTorch) |

---

## Data Flow Architecture

### Analysis Flow

```
User: "Analyze BBCA for potential buy"
              │
              ▼
┌─────────────────────────────┐
│      Planning Agent         │
│  "I need to:                │
│   1. Fetch BBCA price data  │
│   2. Calculate technicals   │
│   3. Check sentiment        │
│   4. Generate prediction    │
│   5. Synthesize analysis"   │
└─────────────────────────────┘
              │
              ▼
┌─────────────────────────────┐
│       Action Agent          │
│  Execute tools:             │
│  → get_stock_price(BBCA)    │
│  → calculate_technical()    │
│  → analyze_sentiment()      │
│  → predict_direction()      │
└─────────────────────────────┘
              │
              ▼
┌─────────────────────────────┐
│     Validation Agent        │
│  Check:                     │
│  ✓ All data retrieved       │
│  ✓ Indicators calculated    │
│  ✓ Sentiment analyzed       │
│  ✓ Prediction generated     │
└─────────────────────────────┘
              │
              ▼
┌─────────────────────────────┐
│       Answer Agent          │
│  Synthesize final response: │
│  - Price summary            │
│  - Technical signals        │
│  - Sentiment score          │
│  - ML prediction            │
│  - Recommendation           │
└─────────────────────────────┘
              │
              ▼
User: Comprehensive analysis with UP/DOWN signal
```

### Prediction Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    Feature Engineering                       │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │Price Features│ │TA Features  │ │Volume Feat. │           │
│  │  - Returns   │ │ - RSI       │ │ - Vol Ratio │           │
│  │  - Volatility│ │ - MACD      │ │ - OBV       │           │
│  │  - Range     │ │ - BB        │ │ - VWAP      │           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
│         │               │               │                   │
│         └───────────────┴───────────────┘                   │
│                         │                                   │
│                    Feature Vector (~105 features)           │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    Model Ensemble                            │
│  ┌─────────────┐    ┌─────────────┐                        │
│  │  XGBoost    │    │    LSTM     │                        │
│  │  Classifier │    │  Sequence   │                        │
│  │  (0.4 weight)│    │  (0.4 weight)│                       │
│  └─────────────┘    └─────────────┘                        │
│         │                  │                                │
│         └────────┬─────────┘                                │
│                  │                                          │
│  ┌─────────────────────────────┐                           │
│  │    IndoBERT Sentiment       │                           │
│  │    (0.2 weight modifier)    │                           │
│  └─────────────────────────────┘                           │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    Final Prediction                          │
│  {                                                          │
│    "direction": "UP",                                       │
│    "confidence": 0.67,                                      │
│    "horizon_days": 3,                                       │
│    "model_contributions": {                                 │
│      "xgboost": 0.65,                                       │
│      "lstm": 0.70,                                          │
│      "sentiment_modifier": +0.05                            │
│    },                                                       │
│    "key_features": ["rsi_oversold", "macd_bullish_cross"]  │
│  }                                                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
stockai/
├── pyproject.toml
├── README.md
├── .env.example
├── .gitignore
│
├── src/
│   └── stockai/
│       ├── __init__.py
│       ├── __main__.py
│       │
│       ├── cli/                 # CLI Layer
│       │   ├── __init__.py
│       │   ├── main.py
│       │   ├── commands/
│       │   └── ui/
│       │
│       ├── agent/               # Agent Layer
│       │   ├── __init__.py
│       │   ├── orchestrator.py
│       │   ├── phases/
│       │   ├── prompts/
│       │   └── memory/
│       │
│       ├── tools/               # Tool Layer
│       │   ├── __init__.py
│       │   ├── registry.py
│       │   ├── data/
│       │   ├── technical/
│       │   ├── ml/
│       │   └── nlp/
│       │
│       ├── core/                # Domain Layer
│       │   ├── __init__.py
│       │   ├── models/
│       │   ├── technical/
│       │   ├── predictor/
│       │   └── sentiment/
│       │
│       ├── data/                # Data Layer
│       │   ├── __init__.py
│       │   ├── sources/
│       │   ├── storage/
│       │   └── processing/
│       │
│       └── web/                 # Web Layer (Post-MVP)
│           ├── __init__.py
│           ├── app.py
│           ├── routers/
│           └── templates/
│
├── data/                        # Local data storage
│   ├── stockai.db              # SQLite database
│   ├── cache/                  # Price data cache
│   └── models/                 # Trained ML models
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_cli/
│   ├── test_agent/
│   ├── test_tools/
│   └── test_core/
│
└── docs/
    ├── getting-started.md
    ├── architecture.md
    └── api-reference.md
```

---

## Security Considerations

### API Key Management

```python
# stockai/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    tavily_api_key: str = ""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
```

### Data Privacy

- All user data stored locally
- No telemetry or analytics
- Portfolio data never transmitted
- API keys in environment variables only

---

## Scalability Path

### Phase 1: Single User CLI
- SQLite database
- Local ML models
- Single-threaded execution

### Phase 2: Web Dashboard
- Add FastAPI server
- Async data fetching
- Plotly visualizations

### Phase 3: Multi-user (Future)
- PostgreSQL migration
- Redis caching
- User authentication

---

## Integration Points

### External APIs

| API | Purpose | Rate Limit | Fallback |
|-----|---------|------------|----------|
| Yahoo Finance | Price data | ~2000/hr | Local cache |
| Sectors.app | Fundamentals | 100/day | Cached data |
| Tavily | Web search | 1000/mo | None |
| OpenAI | Agent reasoning | Pay per use | Anthropic |

### LLM Provider Switching

```python
# Support multiple providers (from kai-code pattern)
MODEL_MAP = {
    "gpt-4o": "openai:gpt-4o",
    "gpt-4": "openai:gpt-4",
    "claude-sonnet": "anthropic:claude-3-5-sonnet-20241022",
    "claude-opus": "anthropic:claude-3-opus-20240229",
}

def get_model(model_name: str):
    model_id = MODEL_MAP.get(model_name, model_name)
    return init_chat_model(model_id)
```

---

**Document Status:** APPROVED
**Next Step:** UX Design
