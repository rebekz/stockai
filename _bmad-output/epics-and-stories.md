# StockAI - Epics and User Stories

**Version:** 1.0.0
**Date:** 2026-01-02
**Author:** Bob (Scrum Master Agent) with BMAD Team
**Status:** APPROVED (YOLO Mode)

---

## Epic Overview

| Epic | Title | Stories | Priority | Status |
|------|-------|---------|----------|--------|
| E1 | Core CLI & Data Layer | 8 | P0 | Ready |
| E2 | Agent Engine | 7 | P0 | Ready |
| E3 | Prediction Models | 6 | P0 | Ready |
| E4 | Portfolio Management | 5 | P1 | Ready |
| E5 | Sentiment Analysis | 5 | P1 | Ready |
| E6 | Web Dashboard | 5 | P2 | Ready |

---

## Epic 1: Core CLI & Data Layer

**Goal:** Establish the foundation - CLI interface and data infrastructure

### Story 1.1: Project Setup & Configuration

**As a** developer
**I want** a properly structured Python project with all dependencies
**So that** I can start building features immediately

**Acceptance Criteria:**
- [ ] AC1.1.1: `pyproject.toml` with all dependencies as specified in architecture
- [ ] AC1.1.2: Project structure matches architecture document exactly
- [ ] AC1.1.3: `.env.example` with all required environment variables documented
- [ ] AC1.1.4: `pip install -e .` works without errors
- [ ] AC1.1.5: `stock --help` displays help text
- [ ] AC1.1.6: `.gitignore` excludes appropriate files (venv, .env, __pycache__, etc.)

**Tasks:**
1. Create `pyproject.toml` with project metadata and dependencies
2. Create directory structure per architecture
3. Create `__init__.py` files with version info
4. Create `.env.example` template
5. Create basic `cli/main.py` with Typer app
6. Test installation in fresh virtual environment

---

### Story 1.2: Database Models & Initialization

**As a** developer
**I want** SQLAlchemy models for all data entities
**So that** I can persist and query stock data efficiently

**Acceptance Criteria:**
- [ ] AC1.2.1: `Stock` model with ticker, name, sector, market_cap fields
- [ ] AC1.2.2: `Price` model with OHLCV fields and foreign key to Stock
- [ ] AC1.2.3: `Prediction` model with direction, confidence, features_json
- [ ] AC1.2.4: `Portfolio` model with ticker, quantity, avg_cost
- [ ] AC1.2.5: `IndexMember` model for IDX30/LQ45 constituents
- [ ] AC1.2.6: `init_db()` function creates all tables
- [ ] AC1.2.7: Database file created at `data/stockai.db`

**Tasks:**
1. Create `stockai/data/storage/models.py` with all models
2. Create `stockai/data/storage/database.py` with session factory
3. Implement `init_db()` function
4. Add database initialization to CLI startup
5. Write unit tests for model creation

---

### Story 1.3: Yahoo Finance Data Connector

**As a** user
**I want** to fetch Indonesian stock data from Yahoo Finance
**So that** I have access to historical price information

**Acceptance Criteria:**
- [ ] AC1.3.1: `YahooFinanceConnector` class implemented
- [ ] AC1.3.2: `get_ticker()` adds `.JK` suffix to IDX tickers
- [ ] AC1.3.3: `fetch_prices()` returns DataFrame with date, open, high, low, close, volume
- [ ] AC1.3.4: `fetch_info()` returns stock metadata dictionary
- [ ] AC1.3.5: `fetch_multiple()` handles multiple tickers with error tolerance
- [ ] AC1.3.6: Rate limiting prevents API abuse (max 5 requests/second)
- [ ] AC1.3.7: Proper error handling for invalid tickers

**Tasks:**
1. Create `stockai/data/sources/yahoo.py`
2. Implement `YahooFinanceConnector` class
3. Add `.JK` suffix handling
4. Implement rate limiting decorator
5. Write integration tests (with mocking)

---

### Story 1.4: IDX Index Data

**As a** user
**I want** to know which stocks belong to IDX30 and LQ45
**So that** I can focus on liquid, quality stocks

**Acceptance Criteria:**
- [ ] AC1.4.1: `IDX_INDICES` dictionary contains IDX30, LQ45, banking, consumer, energy, telco
- [ ] AC1.4.2: `get_index_members(code)` returns list of tickers
- [ ] AC1.4.3: `list_available_indices()` returns all index codes
- [ ] AC1.4.4: Index data stored in database with effective dates
- [ ] AC1.4.5: `stock index LQ45` displays all members

**Tasks:**
1. Create `stockai/data/sources/idx.py` with index data
2. Implement index member functions
3. Create database seeding for index members
4. Add `index` command to CLI

---

### Story 1.5: Stock Info Command

**As a** user
**I want** to run `stock info BBCA`
**So that** I can see basic information about a stock

**Acceptance Criteria:**
- [ ] AC1.5.1: Command `stock info <ticker>` works
- [ ] AC1.5.2: Displays: Name, Sector, Market Cap, Listing Date
- [ ] AC1.5.3: Uses Rich Panel for formatting per UX design
- [ ] AC1.5.4: Shows loading spinner during fetch
- [ ] AC1.5.5: Shows helpful error for invalid tickers
- [ ] AC1.5.6: Response time < 3 seconds

**Tasks:**
1. Create `stockai/cli/commands/stock.py`
2. Implement `info` command with Typer
3. Create Rich UI components for display
4. Add error handling with suggestions
5. Write end-to-end test

---

### Story 1.6: Price History Command

**As a** user
**I want** to run `stock price BBCA -d 30`
**So that** I can see recent price history

**Acceptance Criteria:**
- [ ] AC1.6.1: Command `stock price <ticker>` works
- [ ] AC1.6.2: `--days` flag controls history length (default: 30)
- [ ] AC1.6.3: Displays current price panel with change
- [ ] AC1.6.4: Shows ASCII chart of price trend
- [ ] AC1.6.5: Shows table of last 10 days OHLCV
- [ ] AC1.6.6: Green/red colors for up/down movement
- [ ] AC1.6.7: Prices formatted with thousands separator

**Tasks:**
1. Extend `stock.py` with `price` command
2. Create `stockai/cli/ui/charts.py` for ASCII charts
3. Create `stockai/cli/ui/tables.py` for data tables
4. Implement color coding logic
5. Write tests for formatting

---

### Story 1.7: Data Caching Layer

**As a** developer
**I want** automatic caching of fetched data
**So that** repeated queries are fast and don't hit API limits

**Acceptance Criteria:**
- [ ] AC1.7.1: Price data cached in database on fetch
- [ ] AC1.7.2: Cache hit returns data without API call
- [ ] AC1.7.3: Cache expiry: 15 minutes for intraday, 1 day for historical
- [ ] AC1.7.4: `--refresh` flag forces cache bypass
- [ ] AC1.7.5: Cache stats viewable via `stock cache stats`

**Tasks:**
1. Create `stockai/data/storage/cache.py`
2. Implement cache decorator for connector methods
3. Add cache management commands
4. Configure TTL settings

---

### Story 1.8: Configuration System

**As a** user
**I want** persistent configuration for defaults
**So that** I don't have to specify options every time

**Acceptance Criteria:**
- [ ] AC1.8.1: Config file at `~/.stockai/config.toml`
- [ ] AC1.8.2: Project-level config at `.stockai/config.toml`
- [ ] AC1.8.3: Settings: default_model, default_index, watchlist
- [ ] AC1.8.4: `stock config set <key> <value>` works
- [ ] AC1.8.5: `stock config show` displays current config
- [ ] AC1.8.6: Environment variables override config file

**Tasks:**
1. Create `stockai/config.py` with Pydantic Settings
2. Implement config file loading hierarchy
3. Add config CLI commands
4. Document all configuration options

---

## Epic 2: Agent Engine

**Goal:** Implement the autonomous AI agent using LangChain DeepAgents

### Story 2.1: Base Agent Setup

**As a** developer
**I want** the core DeepAgent infrastructure
**So that** I can build autonomous analysis capabilities

**Acceptance Criteria:**
- [ ] AC2.1.1: `StockAIAgent` class created using `create_deep_agent`
- [ ] AC2.1.2: Agent initializes with configurable model (gpt-4o, claude-sonnet)
- [ ] AC2.1.3: System prompt loaded from `prompts/system.py`
- [ ] AC2.1.4: Agent has access to all registered tools
- [ ] AC2.1.5: Conversation state persisted to `.stockai/session.json`
- [ ] AC2.1.6: `agent.run(prompt)` returns structured result

**Tasks:**
1. Create `stockai/agent/orchestrator.py`
2. Implement `StockAIAgent` class
3. Create system prompt with domain expertise
4. Set up tool registration
5. Implement state persistence

---

### Story 2.2: Planning Phase

**As an** AI agent
**I want** to decompose complex queries into tasks
**So that** I can systematically research any question

**Acceptance Criteria:**
- [ ] AC2.2.1: Planning prompt generates structured task list
- [ ] AC2.2.2: Tasks have: id, description, required_tools, dependencies
- [ ] AC2.2.3: "Analyze BBCA" decomposes into 5+ subtasks
- [ ] AC2.2.4: Planning considers task dependencies (price before technical)
- [ ] AC2.2.5: Plan displayed to user with progress tracking
- [ ] AC2.2.6: Plan stored in agent state for execution

**Tasks:**
1. Create `stockai/agent/phases/planning.py`
2. Write planning prompt template
3. Implement task schema with Pydantic
4. Add dependency resolution logic
5. Create progress display component

---

### Story 2.3: Action Phase with Tools

**As an** AI agent
**I want** to execute tools based on my plan
**So that** I can gather data and perform analysis

**Acceptance Criteria:**
- [ ] AC2.3.1: Agent selects appropriate tool for each task
- [ ] AC2.3.2: Tool results stored in agent state
- [ ] AC2.3.3: Errors handled gracefully with retry logic
- [ ] AC2.3.4: Tool execution displayed in real-time (verbose mode)
- [ ] AC2.3.5: Maximum 10 tool calls per query (safety limit)
- [ ] AC2.3.6: Timeout of 30 seconds per tool call

**Tasks:**
1. Create `stockai/agent/phases/action.py`
2. Implement tool selection logic
3. Add retry mechanism (3 attempts)
4. Create verbose execution display
5. Add safety limits

---

### Story 2.4: Validation Phase

**As an** AI agent
**I want** to validate my work before responding
**So that** I provide accurate and complete answers

**Acceptance Criteria:**
- [ ] AC2.4.1: Validation checks all planned tasks completed
- [ ] AC2.4.2: Validates data quality (no empty results)
- [ ] AC2.4.3: Can request re-execution of failed tasks
- [ ] AC2.4.4: Maximum 2 validation loops
- [ ] AC2.4.5: Logs validation decisions for transparency

**Tasks:**
1. Create `stockai/agent/phases/validation.py`
2. Implement completion checking
3. Add data quality validation
4. Create re-execution request logic
5. Add validation logging

---

### Story 2.5: Answer Synthesis

**As an** AI agent
**I want** to synthesize collected data into a coherent response
**So that** users get actionable insights

**Acceptance Criteria:**
- [ ] AC2.5.1: Answer prompt combines all tool results
- [ ] AC2.5.2: Response follows UX design format
- [ ] AC2.5.3: Key findings highlighted with bullet points
- [ ] AC2.5.4: Prediction included with confidence
- [ ] AC2.5.5: Disclaimer automatically appended
- [ ] AC2.5.6: Response structured for both CLI and JSON output

**Tasks:**
1. Create `stockai/agent/phases/answer.py`
2. Write answer synthesis prompt
3. Create response formatting utilities
4. Add disclaimer insertion
5. Support multiple output formats

---

### Story 2.6: Tool Registration System

**As a** developer
**I want** a clean tool registration pattern
**So that** I can easily add new capabilities

**Acceptance Criteria:**
- [ ] AC2.6.1: `@stockai_tool` decorator for tool definition
- [ ] AC2.6.2: Tools auto-registered on import
- [ ] AC2.6.3: Tool list viewable via `stock tools`
- [ ] AC2.6.4: Tools have docstrings for LLM understanding
- [ ] AC2.6.5: Permission system for dangerous tools

**Tasks:**
1. Create `stockai/tools/registry.py`
2. Implement tool decorator
3. Create tool listing command
4. Document tool creation pattern

---

### Story 2.7: Analyze Command Integration

**As a** user
**I want** to run `stock analyze BBCA`
**So that** I get comprehensive AI analysis

**Acceptance Criteria:**
- [ ] AC2.7.1: `stock analyze <ticker>` invokes agent
- [ ] AC2.7.2: Shows planning progress in real-time
- [ ] AC2.7.3: Displays tool execution (optional with `--verbose`)
- [ ] AC2.7.4: Final output matches UX design analysis format
- [ ] AC2.7.5: Total time < 60 seconds
- [ ] AC2.7.6: Works offline with cached data (degraded mode)

**Tasks:**
1. Create `stockai/cli/commands/analyze.py`
2. Integrate agent with CLI
3. Create progress display
4. Add timing metrics
5. Implement degraded mode

---

## Epic 3: Prediction Models

**Goal:** Implement ML models for direction prediction

### Story 3.1: Feature Engineering Pipeline

**As a** ML system
**I want** to generate ~105 features from raw data
**So that** models have rich inputs for prediction

**Acceptance Criteria:**
- [ ] AC3.1.1: Price features: returns (1d, 5d, 20d), volatility, range
- [ ] AC3.1.2: Technical features: RSI, MACD, BB, Stochastic, ATR
- [ ] AC3.1.3: Volume features: volume ratio, OBV, accumulation
- [ ] AC3.1.4: Market features: IHSG correlation, sector performance
- [ ] AC3.1.5: All features normalized appropriately
- [ ] AC3.1.6: Feature pipeline returns DataFrame with consistent columns
- [ ] AC3.1.7: Missing values handled (forward fill, then drop)

**Tasks:**
1. Create `stockai/core/predictor/features.py`
2. Implement price feature calculators
3. Implement technical feature calculators
4. Implement volume feature calculators
5. Create feature normalization
6. Write comprehensive unit tests

---

### Story 3.2: XGBoost Classifier

**As a** prediction system
**I want** an XGBoost model for quick predictions
**So that** I can provide fast baseline predictions

**Acceptance Criteria:**
- [ ] AC3.2.1: Binary classification: UP (1) vs DOWN (0)
- [ ] AC3.2.2: Model trained on 3-year historical data
- [ ] AC3.2.3: Validation accuracy > 55% (better than random)
- [ ] AC3.2.4: Feature importance extractable
- [ ] AC3.2.5: Model serialized to `data/models/xgboost_v1.json`
- [ ] AC3.2.6: Inference time < 100ms

**Tasks:**
1. Create `stockai/core/predictor/xgboost_model.py`
2. Implement training pipeline
3. Add walk-forward cross-validation
4. Create model serialization
5. Implement inference method

---

### Story 3.3: LSTM Sequence Model

**As a** prediction system
**I want** an LSTM model for pattern recognition
**So that** I capture sequential dependencies

**Acceptance Criteria:**
- [ ] AC3.3.1: Sequence length: 20 days
- [ ] AC3.3.2: Architecture: 2 LSTM layers + Dense head
- [ ] AC3.3.3: Output: probability of UP direction
- [ ] AC3.3.4: Training with early stopping
- [ ] AC3.3.5: Model saved in PyTorch format
- [ ] AC3.3.6: GPU acceleration if available

**Tasks:**
1. Create `stockai/core/predictor/lstm_model.py`
2. Implement PyTorch model class
3. Create sequence data loader
4. Implement training loop
5. Add model checkpointing

---

### Story 3.4: Ensemble Predictor

**As a** prediction system
**I want** to combine XGBoost and LSTM predictions
**So that** I get more robust predictions

**Acceptance Criteria:**
- [ ] AC3.4.1: Weighted ensemble: XGBoost (0.4), LSTM (0.4), Sentiment (0.2 modifier)
- [ ] AC3.4.2: Final confidence = weighted average
- [ ] AC3.4.3: Direction = majority vote with confidence threshold
- [ ] AC3.4.4: Calibration: predicted 60% confidence should hit ~60%
- [ ] AC3.4.5: Model contributions visible in output

**Tasks:**
1. Create `stockai/core/predictor/ensemble.py`
2. Implement ensemble class
3. Add calibration logic
4. Create output structure
5. Write ensemble tests

---

### Story 3.5: Predict Command

**As a** user
**I want** to run `stock predict BBCA --horizon 3`
**So that** I get AI direction prediction

**Acceptance Criteria:**
- [ ] AC3.5.1: `stock predict <ticker>` works
- [ ] AC3.5.2: `--horizon` flag for 1, 3, 7 days (default: 3)
- [ ] AC3.5.3: Output matches UX design prediction format
- [ ] AC3.5.4: Shows confidence bar visualization
- [ ] AC3.5.5: Shows historical accuracy for this model+horizon
- [ ] AC3.5.6: Prediction logged to database

**Tasks:**
1. Create prediction command in CLI
2. Integrate ensemble predictor
3. Create prediction display
4. Add accuracy tracking
5. Implement logging

---

### Story 3.6: Model Training Pipeline

**As a** developer
**I want** automated model training
**So that** models stay up-to-date

**Acceptance Criteria:**
- [ ] AC3.6.1: `stock train` command initiates training
- [ ] AC3.6.2: Training data: last 3 years for all IDX30 stocks
- [ ] AC3.6.3: Walk-forward validation with 3-month test windows
- [ ] AC3.6.4: Training metrics logged and displayed
- [ ] AC3.6.5: Models versioned (v1, v2, etc.)
- [ ] AC3.6.6: Automatic model selection based on validation performance

**Tasks:**
1. Create training command
2. Implement data preparation
3. Add walk-forward validation
4. Create metrics logging
5. Implement model versioning

---

## Epic 4: Portfolio Management

**Goal:** Track user's stock positions and performance

### Story 4.1: Portfolio Data Model

**As a** user
**I want** to store my stock positions
**So that** I can track my investments

**Acceptance Criteria:**
- [ ] AC4.1.1: Portfolio table: ticker, quantity, avg_cost, added_date
- [ ] AC4.1.2: Transaction history tracked
- [ ] AC4.1.3: Support partial sells
- [ ] AC4.1.4: Multiple portfolios supported (optional)

**Tasks:**
1. Extend database models for portfolio
2. Create transaction tracking
3. Add portfolio CRUD operations

---

### Story 4.2: Add/Remove Positions

**As a** user
**I want** to run `stock add BBCA 100 @9500`
**So that** I can record my purchases

**Acceptance Criteria:**
- [ ] AC4.2.1: `stock add <ticker> <qty> [@price]` works
- [ ] AC4.2.2: Price defaults to current market if not specified
- [ ] AC4.2.3: `stock remove <ticker> [qty]` removes position
- [ ] AC4.2.4: Confirmation prompt before action
- [ ] AC4.2.5: Success message shows updated position

**Tasks:**
1. Create portfolio commands
2. Implement add logic with price lookup
3. Implement remove logic
4. Add confirmation prompts

---

### Story 4.3: Portfolio View

**As a** user
**I want** to run `stock portfolio`
**So that** I can see all my positions

**Acceptance Criteria:**
- [ ] AC4.3.1: Displays table of all positions
- [ ] AC4.3.2: Shows: ticker, qty, avg_cost, current_price, P&L, P&L%
- [ ] AC4.3.3: Color-coded P&L (green/red)
- [ ] AC4.3.4: Summary row with total value and total P&L
- [ ] AC4.3.5: Optional sector grouping with `--by-sector`

**Tasks:**
1. Implement portfolio view command
2. Create portfolio value calculation
3. Build Rich table display
4. Add sector grouping option

---

### Story 4.4: P&L Tracking

**As a** user
**I want** to see my profit/loss over time
**So that** I can evaluate my performance

**Acceptance Criteria:**
- [ ] AC4.4.1: `stock pnl` shows P&L summary
- [ ] AC4.4.2: Daily, weekly, monthly, YTD breakdowns
- [ ] AC4.4.3: Best and worst performers listed
- [ ] AC4.4.4: Historical P&L chart (ASCII)

**Tasks:**
1. Implement P&L calculation
2. Create time-based breakdowns
3. Add performer analysis
4. Create historical chart

---

### Story 4.5: Portfolio Analysis

**As a** user
**I want** AI analysis of my portfolio
**So that** I get rebalancing suggestions

**Acceptance Criteria:**
- [ ] AC4.5.1: `stock portfolio analyze` triggers agent
- [ ] AC4.5.2: Analysis includes sector concentration
- [ ] AC4.5.3: Risk assessment (volatility, correlation)
- [ ] AC4.5.4: Suggestions for rebalancing
- [ ] AC4.5.5: Comparison to IDX30 benchmark

**Tasks:**
1. Create portfolio analysis command
2. Implement concentration analysis
3. Add risk metrics
4. Create rebalancing suggestions

---

## Epic 5: Sentiment Analysis

**Goal:** Analyze Indonesian news sentiment

### Story 5.1: IndoBERT Integration

**As a** ML system
**I want** IndoBERT for Indonesian text classification
**So that** I can analyze local news sentiment

**Acceptance Criteria:**
- [ ] AC5.1.1: IndoBERT model loaded from HuggingFace
- [ ] AC5.1.2: Fine-tuned on ID-SMSA dataset (or similar)
- [ ] AC5.1.3: Sentiment output: positive, negative, neutral + score
- [ ] AC5.1.4: Inference time < 500ms per headline
- [ ] AC5.1.5: Model cached locally after first download

**Tasks:**
1. Create `stockai/core/sentiment/indobert.py`
2. Implement model loading
3. Create fine-tuning pipeline (if needed)
4. Add inference method
5. Implement caching

---

### Story 5.2: News Fetcher

**As a** data system
**I want** to fetch Indonesian financial news
**So that** I have content for sentiment analysis

**Acceptance Criteria:**
- [ ] AC5.2.1: RSS feeds: Kontan, Bisnis, CNBC Indonesia
- [ ] AC5.2.2: Filter by ticker mentions
- [ ] AC5.2.3: Last 7 days of news fetched
- [ ] AC5.2.4: Deduplication of similar articles
- [ ] AC5.2.5: Article storage in database

**Tasks:**
1. Create `stockai/core/sentiment/news_fetcher.py`
2. Implement RSS parsing
3. Add ticker extraction
4. Create deduplication logic
5. Add database storage

---

### Story 5.3: Sentiment Analysis Tool

**As an** AI agent
**I want** a sentiment analysis tool
**So that** I can include sentiment in analysis

**Acceptance Criteria:**
- [ ] AC5.3.1: `analyze_sentiment(ticker)` tool created
- [ ] AC5.3.2: Returns: overall_score, headlines[], source_breakdown
- [ ] AC5.3.3: Scores aggregated across sources
- [ ] AC5.3.4: Time decay: recent news weighted higher
- [ ] AC5.3.5: Tool registered with agent

**Tasks:**
1. Create sentiment tool in tools layer
2. Implement aggregation logic
3. Add time decay weighting
4. Register with agent

---

### Story 5.4: Sentiment Command

**As a** user
**I want** to run `stock sentiment BBCA`
**So that** I can see news sentiment

**Acceptance Criteria:**
- [ ] AC5.4.1: `stock sentiment <ticker>` works
- [ ] AC5.4.2: Displays overall sentiment score with icon
- [ ] AC5.4.3: Lists top headlines with individual scores
- [ ] AC5.4.4: Source breakdown (Kontan, Bisnis, etc.)
- [ ] AC5.4.5: Trend indicator (improving/worsening)

**Tasks:**
1. Create sentiment command
2. Implement display formatting
3. Add source breakdown
4. Create trend calculation

---

### Story 5.5: Sentiment Integration in Prediction

**As a** prediction system
**I want** sentiment as a prediction feature
**So that** predictions account for market mood

**Acceptance Criteria:**
- [ ] AC5.5.1: Sentiment score added to feature vector
- [ ] AC5.5.2: Sentiment acts as modifier (0.2 weight)
- [ ] AC5.5.3: Positive sentiment boosts UP confidence
- [ ] AC5.5.4: Model contribution visible in output
- [ ] AC5.5.5: Graceful degradation if sentiment unavailable

**Tasks:**
1. Add sentiment to feature engineering
2. Modify ensemble to include sentiment
3. Update prediction output
4. Handle missing sentiment

---

## Epic 6: Web Dashboard

**Goal:** Generate comprehensive web reports

### Story 6.1: FastAPI Setup

**As a** developer
**I want** FastAPI server infrastructure
**So that** I can serve web reports

**Acceptance Criteria:**
- [ ] AC6.1.1: `stock web serve` starts server on port 8000
- [ ] AC6.1.2: Health check endpoint at `/health`
- [ ] AC6.1.3: Static files served from `web/static`
- [ ] AC6.1.4: Jinja2 templates configured

**Tasks:**
1. Create `stockai/web/app.py`
2. Set up FastAPI app
3. Configure static files
4. Add Jinja2 templates

---

### Story 6.2: Analysis Report Page

**As a** user
**I want** web-based analysis reports
**So that** I can view detailed analysis with charts

**Acceptance Criteria:**
- [ ] AC6.2.1: `/report/<ticker>` generates analysis page
- [ ] AC6.2.2: Layout matches UX design
- [ ] AC6.2.3: Plotly candlestick chart embedded
- [ ] AC6.2.4: Technical indicators overlaid on chart
- [ ] AC6.2.5: Responsive design (mobile-friendly)

**Tasks:**
1. Create report router
2. Design HTML template
3. Integrate Plotly charts
4. Add responsive CSS

---

### Story 6.3: Interactive Charts

**As a** user
**I want** interactive price charts
**So that** I can explore data visually

**Acceptance Criteria:**
- [ ] AC6.3.1: Candlestick chart with zoom/pan
- [ ] AC6.3.2: Toggle technical indicators
- [ ] AC6.3.3: Volume bars below chart
- [ ] AC6.3.4: Date range selector
- [ ] AC6.3.5: Chart exports to PNG

**Tasks:**
1. Create Plotly chart component
2. Add indicator toggles
3. Implement range selector
4. Add export functionality

---

### Story 6.4: PDF Export

**As a** user
**I want** to export reports as PDF
**So that** I can share or archive analysis

**Acceptance Criteria:**
- [ ] AC6.4.1: `/report/<ticker>/pdf` downloads PDF
- [ ] AC6.4.2: PDF matches web report layout
- [ ] AC6.4.3: Charts rendered as images
- [ ] AC6.4.4: Proper page breaks

**Tasks:**
1. Implement PDF generation
2. Configure chart rendering
3. Add page layout
4. Create download endpoint

---

### Story 6.5: CLI Web Integration

**As a** user
**I want** `stock analyze BBCA --web`
**So that** I can open analysis in browser

**Acceptance Criteria:**
- [ ] AC6.5.1: `--web` flag opens browser with report
- [ ] AC6.5.2: Server starts automatically if not running
- [ ] AC6.5.3: Report URL displayed in CLI

**Tasks:**
1. Add `--web` flag to analyze command
2. Implement browser opening
3. Add server management

---

## Story Template

```markdown
### Story X.X: [Title]

**As a** [role]
**I want** [feature]
**So that** [benefit]

**Acceptance Criteria:**
- [ ] ACX.X.1: [Criterion 1]
- [ ] ACX.X.2: [Criterion 2]
- [ ] ACX.X.3: [Criterion 3]

**Tasks:**
1. [Task 1]
2. [Task 2]
3. [Task 3]
```

---

## Definition of Done

A story is DONE when:

1. All acceptance criteria met
2. Unit tests written and passing
3. Integration tests passing (if applicable)
4. Code reviewed (self-review in solo project)
5. Documentation updated
6. No regressions in existing tests

---

**Document Status:** APPROVED
**Total Stories:** 36
**Ready for Implementation:** Yes
