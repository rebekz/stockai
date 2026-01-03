"""Sentiment Analysis Module for StockAI.

Provides news sentiment analysis for Indonesian stocks using
Gemini LLM with fallback to transformer models.
"""

from stockai.core.sentiment.analyzer import SentimentAnalyzer
from stockai.core.sentiment.gemini_analyzer import GeminiSentimentAnalyzer, get_sentiment_analyzer
from stockai.core.sentiment.news import NewsAggregator
from stockai.core.sentiment.models import SentimentResult, NewsArticle, AggregatedSentiment

__all__ = [
    "SentimentAnalyzer",
    "GeminiSentimentAnalyzer",
    "get_sentiment_analyzer",
    "NewsAggregator",
    "SentimentResult",
    "NewsArticle",
    "AggregatedSentiment",
]
