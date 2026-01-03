"""Gemini LLM-based Sentiment Analyzer.

Uses Google's Gemini LLM for financial sentiment analysis,
following the dexter project pattern for financial research.
"""

import json
import logging
from typing import Any

from stockai.config import get_settings
from stockai.core.sentiment.models import (
    AggregatedSentiment,
    NewsArticle,
    SentimentLabel,
    SentimentResult,
)

logger = logging.getLogger(__name__)

# Lazy import for LangChain
_llm = None


def _get_gemini_llm():
    """Lazy load Gemini LLM."""
    global _llm
    if _llm is None:
        settings = get_settings()
        if not settings.has_google_api:
            logger.warning("GOOGLE_API_KEY not set, Gemini sentiment unavailable")
            return None

        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            _llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=settings.google_api_key,
                temperature=0.1,  # Low temperature for consistent analysis
                max_output_tokens=500,
            )
            logger.info("Loaded Gemini LLM for sentiment analysis")
        except Exception as e:
            logger.warning(f"Could not load Gemini LLM: {e}")
            return None
    return _llm


# Prompt template for sentiment analysis
SENTIMENT_PROMPT = """You are a financial sentiment analyst specializing in Indonesian stock market news.

Analyze the following news article and determine the sentiment for stock trading purposes.

Article:
{text}

Stock Symbol: {symbol}

Analyze the sentiment and provide your response in this exact JSON format:
{{
    "label": "BULLISH" or "BEARISH" or "NEUTRAL",
    "bullish_prob": 0.0-1.0,
    "bearish_prob": 0.0-1.0,
    "neutral_prob": 0.0-1.0,
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation in 1-2 sentences"
}}

Guidelines:
- BULLISH: News suggests stock price will go up (positive earnings, expansion, upgrades, acquisitions)
- BEARISH: News suggests stock price will go down (losses, layoffs, downgrades, debt issues)
- NEUTRAL: News has no clear impact on stock price
- Probabilities should sum to 1.0
- Consider Indonesian market context and local economic factors
- Be objective and focus on facts that affect stock price

Return ONLY the JSON object, no other text."""


class GeminiSentimentAnalyzer:
    """Sentiment analyzer using Gemini LLM.

    Follows the dexter pattern of using LLM for financial analysis
    with structured output for reliable parsing.
    """

    def __init__(self):
        """Initialize Gemini sentiment analyzer."""
        self._model_name = "gemini-2.0-flash"
        self._fallback = None  # Lazy load fallback

    def _get_fallback_analyzer(self):
        """Get keyword-based fallback analyzer."""
        if self._fallback is None:
            from stockai.core.sentiment.analyzer import SentimentAnalyzer
            self._fallback = SentimentAnalyzer()
        return self._fallback

    def _parse_llm_response(self, response_text: str) -> dict[str, Any]:
        """Parse LLM response to extract sentiment data.

        Args:
            response_text: Raw LLM response

        Returns:
            Parsed sentiment dict
        """
        try:
            # Try to extract JSON from response
            text = response_text.strip()

            # Handle markdown code blocks
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse LLM response: {response_text[:100]}")
            return None

    def analyze(
        self,
        text: str,
        symbol: str = "STOCK",
    ) -> SentimentResult:
        """Analyze sentiment using Gemini LLM.

        Args:
            text: Text to analyze
            symbol: Stock symbol for context

        Returns:
            SentimentResult with Gemini analysis
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

        llm = _get_gemini_llm()
        if llm is None:
            # Fallback to keyword-based analyzer
            logger.debug("Using fallback keyword analyzer")
            return self._get_fallback_analyzer().analyze(text)

        try:
            # Truncate long text
            truncated_text = text[:1500]  # Gemini can handle more context

            prompt = SENTIMENT_PROMPT.format(
                text=truncated_text,
                symbol=symbol,
            )

            response = llm.invoke(prompt)
            parsed = self._parse_llm_response(response.content)

            if parsed is None:
                return self._get_fallback_analyzer().analyze(text)

            # Extract values with defaults
            label_str = parsed.get("label", "NEUTRAL").upper()
            try:
                label = SentimentLabel(label_str)
            except ValueError:
                label = SentimentLabel.NEUTRAL

            bullish = float(parsed.get("bullish_prob", 0.33))
            bearish = float(parsed.get("bearish_prob", 0.33))
            neutral = float(parsed.get("neutral_prob", 0.34))
            confidence = float(parsed.get("confidence", 0.5))

            # Normalize probabilities
            total = bullish + bearish + neutral
            if total > 0:
                bullish /= total
                bearish /= total
                neutral /= total

            return SentimentResult(
                text=text,
                label=label,
                score=confidence,
                bullish_prob=bullish,
                bearish_prob=bearish,
                neutral_prob=neutral,
                model_name=self._model_name,
            )

        except Exception as e:
            logger.warning(f"Gemini analysis failed: {e}, using fallback")
            return self._get_fallback_analyzer().analyze(text)

    def analyze_batch(
        self,
        texts: list[str],
        symbol: str = "STOCK",
    ) -> list[SentimentResult]:
        """Analyze multiple texts.

        Args:
            texts: List of texts to analyze
            symbol: Stock symbol for context

        Returns:
            List of SentimentResult
        """
        return [self.analyze(text, symbol) for text in texts]

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
            symbol = article.symbol or "STOCK"
            sentiment = self.analyze(article.full_text, symbol)
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

        # Calculate confidence based on agreement and model confidence
        total = len(articles)
        agreement = max(bullish_count, bearish_count, neutral_count) / total

        model_confidences = [a.sentiment.score for a in articles if a.sentiment]
        avg_model_conf = sum(model_confidences) / len(model_confidences) if model_confidences else 0.5

        confidence = agreement * 0.6 + avg_model_conf * 0.4

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
        news_volume = min(1.0, math.log1p(total) / math.log1p(20))

        return {
            "sentiment_score": avg_score,
            "sentiment_confidence": avg_confidence,
            "bullish_ratio": bullish / total if total > 0 else 0,
            "bearish_ratio": bearish / total if total > 0 else 0,
            "news_volume": news_volume,
        }


def get_sentiment_analyzer():
    """Get the best available sentiment analyzer.

    Returns GeminiSentimentAnalyzer if Google API is configured,
    otherwise falls back to keyword-based SentimentAnalyzer.
    """
    settings = get_settings()
    if settings.has_google_api:
        return GeminiSentimentAnalyzer()
    else:
        from stockai.core.sentiment.analyzer import SentimentAnalyzer
        return SentimentAnalyzer()
