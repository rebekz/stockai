"""Data models for sentiment analysis."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class SentimentLabel(str, Enum):
    """Sentiment classification labels."""

    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


@dataclass
class SentimentResult:
    """Result of sentiment analysis."""

    text: str
    label: SentimentLabel
    score: float  # 0-1 confidence score
    bullish_prob: float  # Probability of bullish sentiment
    bearish_prob: float  # Probability of bearish sentiment
    neutral_prob: float  # Probability of neutral sentiment
    model_name: str = "default"
    analyzed_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def sentiment_score(self) -> float:
        """Return sentiment score from -1 (bearish) to 1 (bullish)."""
        return self.bullish_prob - self.bearish_prob

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text[:100] + "..." if len(self.text) > 100 else self.text,
            "label": self.label.value,
            "score": round(self.score, 4),
            "sentiment_score": round(self.sentiment_score, 4),
            "bullish_prob": round(self.bullish_prob, 4),
            "bearish_prob": round(self.bearish_prob, 4),
            "neutral_prob": round(self.neutral_prob, 4),
            "model_name": self.model_name,
            "analyzed_at": self.analyzed_at.isoformat(),
        }


@dataclass
class NewsArticle:
    """News article data model."""

    title: str
    content: str
    source: str
    url: str
    published_at: datetime | None = None
    symbol: str | None = None
    sentiment: SentimentResult | None = None

    @property
    def full_text(self) -> str:
        """Get title + content for analysis."""
        return f"{self.title}. {self.content}"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "content": self.content[:200] + "..." if len(self.content) > 200 else self.content,
            "source": self.source,
            "url": self.url,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "symbol": self.symbol,
            "sentiment": self.sentiment.to_dict() if self.sentiment else None,
        }


@dataclass
class AggregatedSentiment:
    """Aggregated sentiment from multiple sources."""

    symbol: str
    article_count: int
    avg_sentiment_score: float  # -1 to 1
    bullish_count: int
    bearish_count: int
    neutral_count: int
    confidence: float  # Overall confidence based on agreement
    dominant_label: SentimentLabel
    articles: list[NewsArticle] = field(default_factory=list)
    analyzed_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def signal_strength(self) -> str:
        """Return signal strength based on sentiment and confidence."""
        score = abs(self.avg_sentiment_score)
        if score > 0.5 and self.confidence > 0.7:
            return "STRONG"
        elif score > 0.3 and self.confidence > 0.5:
            return "MODERATE"
        elif score > 0.1:
            return "WEAK"
        return "NEUTRAL"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "article_count": self.article_count,
            "avg_sentiment_score": round(self.avg_sentiment_score, 4),
            "bullish_count": self.bullish_count,
            "bearish_count": self.bearish_count,
            "neutral_count": self.neutral_count,
            "confidence": round(self.confidence, 4),
            "dominant_label": self.dominant_label.value,
            "signal_strength": self.signal_strength,
            "analyzed_at": self.analyzed_at.isoformat(),
            "articles": [a.to_dict() for a in self.articles[:5]],  # Top 5 articles
        }
