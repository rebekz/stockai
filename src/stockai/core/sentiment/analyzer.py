"""Sentiment Analyzer using transformer models.

Supports multiple models:
1. FinBERT (financial sentiment - English)
2. IndoBERT (Indonesian language)
3. Multilingual sentiment models
"""

import logging
import re
from typing import Any

from stockai.core.sentiment.models import (
    AggregatedSentiment,
    NewsArticle,
    SentimentLabel,
    SentimentResult,
)

logger = logging.getLogger(__name__)

# Lazy imports for transformers to avoid slow startup
_pipeline = None
_tokenizer = None
_model = None


def _get_pipeline():
    """Lazy load transformers pipeline."""
    global _pipeline
    if _pipeline is None:
        try:
            from transformers import pipeline

            # Use multilingual sentiment model that works for Indonesian
            # distilbert-base-multilingual-cased-sentiments-student is fast and multilingual
            _pipeline = pipeline(
                "sentiment-analysis",
                model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
                top_k=None,  # Return all scores
            )
            logger.info("Loaded multilingual sentiment model")
        except Exception as e:
            logger.warning(f"Could not load transformer model: {e}")
            _pipeline = "fallback"
    return _pipeline


class SentimentAnalyzer:
    """Analyzes text sentiment for stock-related news.

    Features:
    - Multilingual support (English + Indonesian)
    - Financial keyword boosting
    - Confidence calibration
    - Batch processing
    """

    # Financial keywords that indicate sentiment
    BULLISH_KEYWORDS = {
        # English
        "profit", "growth", "surge", "gain", "rally", "rise", "up", "high",
        "bullish", "positive", "strong", "beat", "exceed", "outperform",
        "buy", "upgrade", "target", "dividend", "acquisition", "expansion",
        # Indonesian
        "naik", "untung", "laba", "positif", "bagus", "kuat", "tumbuh",
        "meningkat", "melonjak", "rekomendasi beli", "potensi", "ekspansi",
        "akuisisi", "dividen", "target", "tertinggi",
    }

    BEARISH_KEYWORDS = {
        # English
        "loss", "decline", "drop", "fall", "crash", "down", "low",
        "bearish", "negative", "weak", "miss", "underperform",
        "sell", "downgrade", "cut", "layoff", "debt", "risk",
        # Indonesian
        "turun", "rugi", "negatif", "buruk", "lemah", "menurun",
        "merosot", "jatuh", "rekomendasi jual", "risiko", "utang",
        "phk", "kerugian", "terendah", "anjlok",
    }

    def __init__(self, use_gpu: bool = False):
        """Initialize sentiment analyzer.

        Args:
            use_gpu: Whether to use GPU if available
        """
        self._use_gpu = use_gpu
        self._model_name = "multilingual-distilbert"

    def _keyword_sentiment(self, text: str) -> tuple[float, float, float]:
        """Calculate keyword-based sentiment scores.

        Args:
            text: Text to analyze

        Returns:
            Tuple of (bullish, bearish, neutral) scores
        """
        text_lower = text.lower()
        words = set(re.findall(r'\b\w+\b', text_lower))

        bullish_count = len(words & self.BULLISH_KEYWORDS)
        bearish_count = len(words & self.BEARISH_KEYWORDS)
        total_keywords = bullish_count + bearish_count

        if total_keywords == 0:
            return 0.33, 0.33, 0.34

        bullish_score = bullish_count / total_keywords
        bearish_score = bearish_count / total_keywords
        neutral_score = 1 - max(bullish_score, bearish_score)

        return bullish_score, bearish_score, neutral_score

    def _model_sentiment(self, text: str) -> tuple[float, float, float]:
        """Get sentiment from transformer model.

        Args:
            text: Text to analyze

        Returns:
            Tuple of (bullish/positive, bearish/negative, neutral) scores
        """
        pipeline = _get_pipeline()

        if pipeline == "fallback":
            return self._keyword_sentiment(text)

        try:
            # Truncate long text
            text = text[:512]
            result = pipeline(text)[0]

            # Model returns: positive, negative, neutral
            scores = {r["label"].lower(): r["score"] for r in result}

            positive = scores.get("positive", 0.33)
            negative = scores.get("negative", 0.33)
            neutral = scores.get("neutral", 0.34)

            return positive, negative, neutral
        except Exception as e:
            logger.warning(f"Model inference failed: {e}, using fallback")
            return self._keyword_sentiment(text)

    def analyze(self, text: str, use_keywords: bool = True) -> SentimentResult:
        """Analyze sentiment of text.

        Args:
            text: Text to analyze
            use_keywords: Whether to boost with keyword analysis

        Returns:
            SentimentResult with scores and label
        """
        if not text or not text.strip():
            return SentimentResult(
                text=text,
                label=SentimentLabel.NEUTRAL,
                score=0.5,
                bullish_prob=0.33,
                bearish_prob=0.33,
                neutral_prob=0.34,
                model_name=self._model_name,
            )

        # Get model sentiment
        model_bullish, model_bearish, model_neutral = self._model_sentiment(text)

        if use_keywords:
            # Get keyword sentiment
            kw_bullish, kw_bearish, kw_neutral = self._keyword_sentiment(text)

            # Weighted combination (70% model, 30% keywords for financial context)
            bullish = 0.7 * model_bullish + 0.3 * kw_bullish
            bearish = 0.7 * model_bearish + 0.3 * kw_bearish
            neutral = 0.7 * model_neutral + 0.3 * kw_neutral
        else:
            bullish, bearish, neutral = model_bullish, model_bearish, model_neutral

        # Normalize
        total = bullish + bearish + neutral
        if total > 0:
            bullish /= total
            bearish /= total
            neutral /= total

        # Determine label
        max_score = max(bullish, bearish, neutral)
        if bullish == max_score:
            label = SentimentLabel.BULLISH
        elif bearish == max_score:
            label = SentimentLabel.BEARISH
        else:
            label = SentimentLabel.NEUTRAL

        return SentimentResult(
            text=text,
            label=label,
            score=max_score,
            bullish_prob=bullish,
            bearish_prob=bearish,
            neutral_prob=neutral,
            model_name=self._model_name,
        )

    def analyze_batch(self, texts: list[str]) -> list[SentimentResult]:
        """Analyze multiple texts.

        Args:
            texts: List of texts to analyze

        Returns:
            List of SentimentResult
        """
        return [self.analyze(text) for text in texts]

    def analyze_articles(
        self,
        articles: list[NewsArticle],
    ) -> list[NewsArticle]:
        """Analyze sentiment of news articles.

        Args:
            articles: List of NewsArticle objects

        Returns:
            Articles with sentiment filled in
        """
        for article in articles:
            sentiment = self.analyze(article.full_text)
            article.sentiment = sentiment
        return articles

    def aggregate_sentiment(
        self,
        articles: list[NewsArticle],
        symbol: str,
    ) -> AggregatedSentiment:
        """Aggregate sentiment from multiple articles.

        Args:
            articles: List of analyzed articles
            symbol: Stock symbol

        Returns:
            AggregatedSentiment summary
        """
        if not articles:
            return AggregatedSentiment(
                symbol=symbol,
                article_count=0,
                avg_sentiment_score=0,
                bullish_count=0,
                bearish_count=0,
                neutral_count=0,
                confidence=0,
                dominant_label=SentimentLabel.NEUTRAL,
                articles=[],
            )

        # Ensure all articles have sentiment
        articles = self.analyze_articles(articles)

        # Count by label
        bullish_count = sum(
            1 for a in articles
            if a.sentiment and a.sentiment.label == SentimentLabel.BULLISH
        )
        bearish_count = sum(
            1 for a in articles
            if a.sentiment and a.sentiment.label == SentimentLabel.BEARISH
        )
        neutral_count = len(articles) - bullish_count - bearish_count

        # Calculate average sentiment score (-1 to 1)
        sentiment_scores = [
            a.sentiment.sentiment_score
            for a in articles
            if a.sentiment
        ]
        avg_score = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0

        # Determine dominant label
        if bullish_count > bearish_count and bullish_count > neutral_count:
            dominant = SentimentLabel.BULLISH
        elif bearish_count > bullish_count and bearish_count > neutral_count:
            dominant = SentimentLabel.BEARISH
        else:
            dominant = SentimentLabel.NEUTRAL

        # Calculate confidence based on agreement
        total = len(articles)
        agreement = max(bullish_count, bearish_count, neutral_count) / total
        confidence = agreement * 0.8 + 0.2  # Scale to 0.2-1.0

        return AggregatedSentiment(
            symbol=symbol,
            article_count=len(articles),
            avg_sentiment_score=avg_score,
            bullish_count=bullish_count,
            bearish_count=bearish_count,
            neutral_count=neutral_count,
            confidence=confidence,
            dominant_label=dominant,
            articles=articles,
        )

    def get_prediction_feature(
        self,
        articles: list[NewsArticle],
    ) -> dict[str, float]:
        """Get sentiment features for prediction model.

        Args:
            articles: List of analyzed articles

        Returns:
            Dict with sentiment features for ML model
        """
        if not articles:
            return {
                "sentiment_score": 0.0,
                "sentiment_confidence": 0.0,
                "bullish_ratio": 0.0,
                "bearish_ratio": 0.0,
                "news_volume": 0.0,
            }

        # Ensure analyzed
        articles = self.analyze_articles(articles)

        total = len(articles)
        bullish = sum(1 for a in articles if a.sentiment and a.sentiment.label == SentimentLabel.BULLISH)
        bearish = sum(1 for a in articles if a.sentiment and a.sentiment.label == SentimentLabel.BEARISH)

        scores = [a.sentiment.sentiment_score for a in articles if a.sentiment]
        avg_score = sum(scores) / len(scores) if scores else 0

        confidences = [a.sentiment.score for a in articles if a.sentiment]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0

        # Normalize news volume (log scale, capped)
        import math
        news_volume = min(1.0, math.log1p(total) / math.log1p(20))  # Cap at 20 articles

        return {
            "sentiment_score": avg_score,
            "sentiment_confidence": avg_confidence,
            "bullish_ratio": bullish / total if total > 0 else 0,
            "bearish_ratio": bearish / total if total > 0 else 0,
            "news_volume": news_volume,
        }
