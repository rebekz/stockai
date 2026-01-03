# StockAI - Product Requirements Document

**Version:** 1.0.0
**Date:** 2026-01-02
**Author:** John (PM Agent) with BMAD Team
**Status:** APPROVED (YOLO Mode)

---

## Executive Summary

StockAI is an **autonomous financial research agent** built specifically for Indonesian stock market analysis. Think "Claude Code, but for IDX investing." It combines AI-powered task planning, self-reflection, and real-time market data to deliver actionable stock predictions and comprehensive analysis.

### The Opportunity

| Metric | Value | Significance |
|--------|-------|--------------|
| Indonesian Retail Investors | 20.32M | 37% YoY growth |
| AI Stock Prediction Platforms | **0** | Blue ocean opportunity |
| Target Demographic | 54% under 30 | Tech-savvy, mobile-first |
| Market Gap | Complete | No IndoBERT sentiment, no ML predictions |

---

## Problem Statement

Indonesian retail investors face a critical gap:

1. **No AI-powered predictions** - Existing platforms (Stockbit, Ajaib, Bibit) offer charts and news, but ZERO predictive AI
2. **Language barrier** - International AI tools don't understand Bahasa Indonesia for sentiment analysis
3. **Information overload** - Investors drown in data without intelligent synthesis
4. **Manual research** - Hours spent on repetitive analysis that could be automated

### User Pain Points

- "I spend 3+ hours daily reading news and analyzing charts"
- "I can't tell if a stock will go up or down based on sentiment"
- "There's no tool that thinks like a professional analyst"
- "I want AI that understands Indonesian market nuances"

---

## Product Vision

> **StockAI: Your autonomous AI research partner for Indonesian stocks**
>
> An intelligent agent that thinks, plans, and learns as it analyzes the market - delivering confident, data-backed predictions with full transparency.

### Core Value Proposition

| For | Who | StockAI Provides |
|-----|-----|------------------|
| Retail investors | Trade IDX stocks | Autonomous AI analysis with UP/DOWN signals |
| Busy professionals | Have limited time | Morning briefings and automated research |
| Data-driven traders | Want evidence-based decisions | Transparent ML predictions with confidence scores |
| Indonesian speakers | Need local context | IndoBERT-powered Indonesian sentiment analysis |

---

## User Personas

### Primary: Andi (Active Retail Trader)
- **Age:** 28, Jakarta
- **Portfolio:** Rp 50-200M across 10-15 IDX stocks
- **Behavior:** Checks market daily, trades 2-3x per week
- **Pain:** Spends 2-3 hours on research, still uncertain about decisions
- **Goal:** Confident buy/sell signals with supporting evidence

### Secondary: Dewi (Part-time Investor)
- **Age:** 35, Surabaya
- **Portfolio:** Rp 20-50M in blue-chips (IDX30)
- **Behavior:** Long-term hold strategy, checks weekly
- **Pain:** Misses opportunities, no time for deep analysis
- **Goal:** Morning briefings and alerts when action needed

### Tertiary: Budi (Learning Investor)
- **Age:** 24, Bandung
- **Portfolio:** Rp 5-20M, starting out
- **Behavior:** Learning TA, follows influencers
- **Pain:** Information overload, doesn't know what to trust
- **Goal:** Educational AI that explains its reasoning

---

## Features & Requirements

### MVP Features (Epic 1-3)

#### 1. Core CLI Interface
**Priority:** P0 (Must Have)

```
stock info BBCA          # Stock metadata
stock price BBRI -d 30   # Price history
stock technical TLKM     # Technical analysis
stock analyze ASII       # Full AI analysis
stock predict BBCA       # UP/DOWN prediction
stock morning            # Daily briefing
```

**Requirements:**
- FR-001: Display stock info (name, sector, market cap)
- FR-002: Show OHLCV price history with formatting
- FR-003: Calculate and display technical indicators
- FR-004: Generate AI-powered analysis reports
- FR-005: Predict directional movement with confidence

#### 2. Autonomous Agent Engine
**Priority:** P0 (Must Have)

Based on Dexter + Kai-Code architecture:

| Component | Function |
|-----------|----------|
| Planning Agent | Decomposes complex queries into research tasks |
| Action Agent | Selects and executes appropriate tools |
| Validation Agent | Verifies task completion and data quality |
| Reflection Agent | Self-critiques and iterates on findings |
| Answer Agent | Synthesizes findings into final response |

**Requirements:**
- FR-006: Intelligent task decomposition from natural language
- FR-007: Tool selection based on task requirements
- FR-008: Self-validation with retry logic
- FR-009: Loop detection and safety limits
- FR-010: Transparent reasoning chain display

#### 3. Prediction Engine
**Priority:** P0 (Must Have)

| Model | Purpose | Target Accuracy |
|-------|---------|-----------------|
| XGBoost | Feature importance, quick predictions | 58-62% |
| LSTM | Sequential pattern recognition | 60-65% |
| Ensemble | Combined prediction | 62-67% |
| IndoBERT | Indonesian sentiment analysis | 82-86% sentiment |

**Requirements:**
- FR-011: Binary direction prediction (UP/DOWN)
- FR-012: Confidence score (0-100%)
- FR-013: Feature importance explanation
- FR-014: Prediction horizon (1d, 3d, 7d)
- FR-015: Historical accuracy tracking

### Post-MVP Features (Epic 4-6)

#### 4. Portfolio Management
- FR-016: Add/remove positions
- FR-017: Track P&L
- FR-018: Portfolio analysis vs benchmark
- FR-019: Rebalancing suggestions

#### 5. Sentiment Analysis
- FR-020: Indonesian news sentiment (Kontan, Bisnis, CNBC ID)
- FR-021: Social sentiment aggregation
- FR-022: Sentiment trend visualization
- FR-023: Alert on sentiment shifts

#### 6. Web Dashboard
- FR-024: Comprehensive analysis reports
- FR-025: Interactive charts
- FR-026: Multi-stock comparison
- FR-027: Export to PDF

---

## Technical Constraints

### Must Use
- **Python 3.11+** - Primary language
- **LangChain DeepAgents** - Agent framework (from kai-code-1)
- **Typer + Rich** - CLI framework
- **SQLite/DuckDB** - Local database
- **Yahoo Finance API** - Price data (free)
- **IndoBERT** - Indonesian NLP

### Performance Requirements
- NFR-001: CLI response < 3 seconds for basic queries
- NFR-002: Prediction generation < 30 seconds
- NFR-003: Full analysis < 60 seconds
- NFR-004: Morning briefing < 45 seconds
- NFR-005: Support offline mode for cached data

### Security Requirements
- NFR-006: No API keys in code
- NFR-007: Local-first data storage
- NFR-008: No transmission of portfolio data externally

---

## Success Metrics

### MVP Success Criteria
| Metric | Target | Measurement |
|--------|--------|-------------|
| Prediction Accuracy | >60% directional | Backtest on 6 months data |
| CLI Response Time | <3s average | Performance monitoring |
| User Satisfaction | >4/5 rating | Self-assessment |
| Daily Usage | 5+ commands/day | Local logging |

### Long-term Goals
- Sharpe Ratio > 1.0 on paper portfolio
- 90% user retention (weekly active)
- Coverage of all IDX30 + LQ45 stocks

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Yahoo Finance rate limits | Data gaps | Caching layer, multiple sources |
| Model overfit | Poor live performance | Walk-forward validation, ensemble |
| Indonesian NLP quality | Wrong sentiment | Fine-tune IndoBERT on ID-SMSA dataset |
| Regulatory concerns | Legal issues | Disclaimer: not financial advice |

---

## Timeline (YOLO Mode - Aggressive)

| Epic | Focus | Duration |
|------|-------|----------|
| Epic 1 | Core CLI + Data Layer | 2 weeks |
| Epic 2 | Agent Engine | 2 weeks |
| Epic 3 | Prediction Models | 2 weeks |
| Epic 4 | Portfolio Management | 1 week |
| Epic 5 | Sentiment Analysis | 2 weeks |
| Epic 6 | Web Dashboard | 2 weeks |

**Total MVP:** 6 weeks
**Full Product:** 11 weeks

---

## Appendix

### Competitor Analysis

| Platform | Users | AI Features | Gap |
|----------|-------|-------------|-----|
| Stockbit | 900K | None | No predictions |
| Ajaib | 3M+ | None | No analysis |
| Bibit | 3M+ | Portfolio allocation only | No stock-level AI |
| International | N/A | Yes | No IDX coverage |

### Data Source Priority

1. Yahoo Finance (yfinance) - Free, reliable OHLCV
2. Sectors.app - Free tier, fundamentals
3. GOAPI.io - Real-time (paid tier later)
4. Kontan/Bisnis RSS - News sentiment
5. IDX Website - Index constituents (scrape)

---

**Document Status:** APPROVED
**Next Step:** Architecture Design
