"""E2E Tests for Sentiment Analysis Module.

Tests the sentiment analyzer, news aggregator, and CLI commands.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner

from stockai.core.sentiment.models import (
    SentimentLabel,
    SentimentResult,
    NewsArticle,
    AggregatedSentiment,
)
from stockai.core.sentiment.analyzer import SentimentAnalyzer
from stockai.core.sentiment.news import NewsAggregator, MockNewsSource


class TestSentimentModels:
    """Test sentiment data models."""

    def test_sentiment_result_creation(self):
        """Test SentimentResult creation and methods."""
        result = SentimentResult(
            text="Bank BCA reports strong profits",
            label=SentimentLabel.BULLISH,
            score=0.85,
            bullish_prob=0.7,
            bearish_prob=0.2,
            neutral_prob=0.1,
            model_name="test",
        )

        assert result.label == SentimentLabel.BULLISH
        assert result.score == 0.85
        assert result.sentiment_score == pytest.approx(0.5, abs=0.01)  # 0.7 - 0.2

    def test_sentiment_result_to_dict(self):
        """Test SentimentResult serialization."""
        result = SentimentResult(
            text="Short text",
            label=SentimentLabel.NEUTRAL,
            score=0.5,
            bullish_prob=0.33,
            bearish_prob=0.33,
            neutral_prob=0.34,
        )

        d = result.to_dict()
        assert "label" in d
        assert d["label"] == "NEUTRAL"
        assert "sentiment_score" in d
        assert "analyzed_at" in d

    def test_news_article_creation(self):
        """Test NewsArticle creation."""
        article = NewsArticle(
            title="Test News Title",
            content="Test content about stocks",
            source="Test Source",
            url="https://example.com/news/1",
            published_at=datetime.utcnow(),
            symbol="BBCA",
        )

        assert article.title == "Test News Title"
        assert article.symbol == "BBCA"
        assert article.full_text == "Test News Title. Test content about stocks"

    def test_news_article_to_dict(self):
        """Test NewsArticle serialization."""
        article = NewsArticle(
            title="Title",
            content="Content",
            source="Source",
            url="https://example.com",
        )

        d = article.to_dict()
        assert "title" in d
        assert "source" in d
        assert d["sentiment"] is None

    def test_aggregated_sentiment_creation(self):
        """Test AggregatedSentiment creation."""
        agg = AggregatedSentiment(
            symbol="BBCA",
            article_count=10,
            avg_sentiment_score=0.4,  # > 0.3 for MODERATE
            bullish_count=5,
            bearish_count=2,
            neutral_count=3,
            confidence=0.7,
            dominant_label=SentimentLabel.BULLISH,
        )

        assert agg.symbol == "BBCA"
        assert agg.article_count == 10
        assert agg.signal_strength == "MODERATE"  # score > 0.3, conf > 0.5

    def test_aggregated_sentiment_signal_strength(self):
        """Test signal strength calculation."""
        # Strong signal
        strong = AggregatedSentiment(
            symbol="TEST",
            article_count=5,
            avg_sentiment_score=0.6,
            bullish_count=4,
            bearish_count=1,
            neutral_count=0,
            confidence=0.8,
            dominant_label=SentimentLabel.BULLISH,
        )
        assert strong.signal_strength == "STRONG"

        # Weak signal
        weak = AggregatedSentiment(
            symbol="TEST",
            article_count=5,
            avg_sentiment_score=0.15,
            bullish_count=2,
            bearish_count=1,
            neutral_count=2,
            confidence=0.4,
            dominant_label=SentimentLabel.NEUTRAL,
        )
        assert weak.signal_strength == "WEAK"


class TestSentimentAnalyzer:
    """Test sentiment analyzer."""

    def setup_method(self):
        """Setup for each test."""
        self.analyzer = SentimentAnalyzer()

    def test_analyzer_initialization(self):
        """Test analyzer creation."""
        assert self.analyzer is not None
        assert self.analyzer._model_name == "multilingual-distilbert"

    def test_keyword_sentiment_bullish(self):
        """Test keyword-based bullish sentiment."""
        text = "Company reports strong profit growth and positive outlook"
        bullish, bearish, neutral = self.analyzer._keyword_sentiment(text)

        assert bullish > bearish
        assert bullish > 0

    def test_keyword_sentiment_bearish(self):
        """Test keyword-based bearish sentiment."""
        text = "Company reports loss and decline in revenue due to weak demand"
        bullish, bearish, neutral = self.analyzer._keyword_sentiment(text)

        assert bearish > bullish
        assert bearish > 0

    def test_keyword_sentiment_neutral(self):
        """Test keyword-based neutral sentiment."""
        text = "The weather is nice today"  # No financial keywords
        bullish, bearish, neutral = self.analyzer._keyword_sentiment(text)

        assert bullish == pytest.approx(0.33, abs=0.01)
        assert bearish == pytest.approx(0.33, abs=0.01)

    def test_keyword_sentiment_indonesian(self):
        """Test Indonesian keyword detection."""
        text = "Laba perusahaan naik signifikan, potensi pertumbuhan kuat"
        bullish, bearish, neutral = self.analyzer._keyword_sentiment(text)

        assert bullish > bearish  # "naik", "laba", "potensi", "kuat" are bullish

    def test_analyze_empty_text(self):
        """Test analysis of empty text."""
        result = self.analyzer.analyze("")

        assert result.label == SentimentLabel.NEUTRAL
        assert result.score == 0.5

    def test_analyze_bullish_text(self):
        """Test analysis of bullish text."""
        text = "Bank BCA reports record profit growth, dividend increase announced"

        # Mock the model to ensure consistent test
        with patch.object(
            self.analyzer, '_model_sentiment',
            return_value=(0.7, 0.2, 0.1)
        ):
            result = self.analyzer.analyze(text)

        assert result.label == SentimentLabel.BULLISH
        assert result.bullish_prob > result.bearish_prob

    def test_analyze_bearish_text(self):
        """Test analysis of bearish text."""
        text = "Company announces massive layoffs, stock crashes amid losses"

        with patch.object(
            self.analyzer, '_model_sentiment',
            return_value=(0.1, 0.8, 0.1)
        ):
            result = self.analyzer.analyze(text)

        assert result.label == SentimentLabel.BEARISH
        assert result.bearish_prob > result.bullish_prob

    def test_analyze_batch(self):
        """Test batch analysis."""
        texts = [
            "Profit up by 20%",
            "Sales decline sharply",
            "Market remains stable",
        ]

        with patch.object(
            self.analyzer, '_model_sentiment',
            side_effect=[
                (0.7, 0.2, 0.1),  # Bullish
                (0.2, 0.7, 0.1),  # Bearish
                (0.3, 0.3, 0.4),  # Neutral
            ]
        ):
            results = self.analyzer.analyze_batch(texts)

        assert len(results) == 3
        assert results[0].label == SentimentLabel.BULLISH
        assert results[1].label == SentimentLabel.BEARISH
        assert results[2].label == SentimentLabel.NEUTRAL

    def test_analyze_articles(self):
        """Test article sentiment analysis."""
        articles = [
            NewsArticle(
                title="Good news",
                content="Company profits soar",
                source="Test",
                url="https://test.com/1",
            ),
            NewsArticle(
                title="Bad news",
                content="Stock crashes",
                source="Test",
                url="https://test.com/2",
            ),
        ]

        with patch.object(
            self.analyzer, '_model_sentiment',
            side_effect=[
                (0.8, 0.1, 0.1),  # Good news
                (0.1, 0.8, 0.1),  # Bad news
            ]
        ):
            analyzed = self.analyzer.analyze_articles(articles)

        assert all(a.sentiment is not None for a in analyzed)
        assert analyzed[0].sentiment.label == SentimentLabel.BULLISH
        assert analyzed[1].sentiment.label == SentimentLabel.BEARISH

    def test_aggregate_sentiment(self):
        """Test sentiment aggregation."""
        articles = [
            NewsArticle(title="Good 1", content="profit", source="A", url="1"),
            NewsArticle(title="Good 2", content="growth", source="B", url="2"),
            NewsArticle(title="Bad 1", content="loss", source="C", url="3"),
        ]

        with patch.object(
            self.analyzer, '_model_sentiment',
            side_effect=[
                (0.8, 0.1, 0.1),
                (0.7, 0.2, 0.1),
                (0.2, 0.7, 0.1),
            ]
        ):
            aggregated = self.analyzer.aggregate_sentiment(articles, "BBCA")

        assert aggregated.symbol == "BBCA"
        assert aggregated.article_count == 3
        assert aggregated.bullish_count == 2
        assert aggregated.bearish_count == 1
        assert aggregated.dominant_label == SentimentLabel.BULLISH

    def test_aggregate_sentiment_empty(self):
        """Test aggregation with no articles."""
        aggregated = self.analyzer.aggregate_sentiment([], "TEST")

        assert aggregated.article_count == 0
        assert aggregated.dominant_label == SentimentLabel.NEUTRAL
        assert aggregated.confidence == 0

    def test_get_prediction_feature(self):
        """Test feature extraction for prediction."""
        articles = [
            NewsArticle(title="Good", content="profit", source="A", url="1"),
            NewsArticle(title="Bad", content="loss", source="B", url="2"),
        ]

        with patch.object(
            self.analyzer, '_model_sentiment',
            side_effect=[
                (0.8, 0.1, 0.1),
                (0.2, 0.7, 0.1),
            ]
        ):
            features = self.analyzer.get_prediction_feature(articles)

        assert "sentiment_score" in features
        assert "bullish_ratio" in features
        assert "bearish_ratio" in features
        assert "news_volume" in features
        assert 0 <= features["news_volume"] <= 1

    def test_get_prediction_feature_empty(self):
        """Test feature extraction with no articles."""
        features = self.analyzer.get_prediction_feature([])

        assert features["sentiment_score"] == 0.0
        assert features["news_volume"] == 0.0


class TestNewsAggregator:
    """Test news aggregation."""

    def setup_method(self):
        """Setup for each test."""
        self.aggregator = NewsAggregator()

    def test_aggregator_initialization(self):
        """Test aggregator creation."""
        assert self.aggregator is not None
        assert self.aggregator._timeout == 10

    def test_get_search_query_known_stock(self):
        """Test search query for known stock."""
        query = self.aggregator._get_search_query("BBCA")

        assert "BBCA" in query
        assert "Bank Central Asia" in query
        assert "saham" in query

    def test_get_search_query_unknown_stock(self):
        """Test search query for unknown stock."""
        query = self.aggregator._get_search_query("XXXX")

        assert "XXXX" in query
        assert "saham" in query
        assert "IDX" in query

    def test_clean_html(self):
        """Test HTML cleaning."""
        html = "<p>This is <b>bold</b> text</p><script>alert('x')</script>"
        clean = self.aggregator._clean_html(html)

        assert "bold" in clean
        assert "<" not in clean
        assert "script" not in clean or "alert" not in clean

    def test_parse_date_rss_format(self):
        """Test RSS date parsing."""
        date_str = "Fri, 03 Jan 2025 10:30:00 GMT"
        parsed = self.aggregator._parse_date(date_str)

        assert parsed is not None
        assert parsed.year == 2025
        assert parsed.month == 1
        assert parsed.day == 3

    def test_parse_date_iso_format(self):
        """Test ISO date parsing."""
        date_str = "2025-01-03T10:30:00Z"
        parsed = self.aggregator._parse_date(date_str)

        assert parsed is not None
        assert parsed.year == 2025

    def test_parse_date_invalid(self):
        """Test invalid date parsing."""
        parsed = self.aggregator._parse_date("not a date")
        assert parsed is None

    @patch('feedparser.parse')
    def test_fetch_google_news(self, mock_parse):
        """Test Google News fetching."""
        mock_parse.return_value = MagicMock(
            entries=[
                {
                    "title": "BBCA Stock Rises - News Source",
                    "summary": "<p>Content here</p>",
                    "link": "https://news.google.com/123",
                    "published": "2025-01-03",
                }
            ]
        )

        articles = self.aggregator.fetch_google_news("BBCA", max_articles=5)

        assert len(articles) == 1
        assert articles[0].title == "BBCA Stock Rises"
        assert articles[0].source == "News Source"
        assert articles[0].symbol == "BBCA"

    @patch('feedparser.parse')
    def test_fetch_yahoo_news(self, mock_parse):
        """Test Yahoo Finance news fetching."""
        mock_parse.return_value = MagicMock(
            entries=[
                {
                    "title": "Yahoo News Title",
                    "summary": "News content",
                    "link": "https://finance.yahoo.com/news/123",
                    "published": "2025-01-03",
                }
            ]
        )

        articles = self.aggregator.fetch_yahoo_news("BBCA", max_articles=5)

        assert len(articles) == 1
        assert articles[0].source == "Yahoo Finance"

    @patch.object(NewsAggregator, 'fetch_firecrawl_news')
    @patch.object(NewsAggregator, 'fetch_kontan_news')
    @patch.object(NewsAggregator, 'fetch_bisnis_news')
    @patch.object(NewsAggregator, 'fetch_cnbc_indonesia_news')
    @patch.object(NewsAggregator, 'fetch_detik_finance_news')
    @patch.object(NewsAggregator, 'fetch_google_news')
    @patch.object(NewsAggregator, 'fetch_yahoo_news')
    def test_fetch_all(self, mock_yahoo, mock_google, mock_detik, mock_cnbc, mock_bisnis, mock_kontan, mock_firecrawl):
        """Test fetching from all sources."""
        # Mock all scrapers to return empty (simulating unavailable sources)
        mock_firecrawl.return_value = []
        mock_kontan.return_value = []
        mock_bisnis.return_value = []
        mock_cnbc.return_value = []
        mock_detik.return_value = []

        mock_google.return_value = [
            NewsArticle(
                title="Google News 1",
                content="Content",
                source="Google",
                url="https://google.com/1",
                published_at=datetime.utcnow(),
                symbol="BBCA",
            )
        ]
        mock_yahoo.return_value = [
            NewsArticle(
                title="Yahoo News 1",
                content="Content",
                source="Yahoo",
                url="https://yahoo.com/1",
                published_at=datetime.utcnow(),
                symbol="BBCA",
            )
        ]

        articles = self.aggregator.fetch_all("BBCA", max_articles=10)

        assert len(articles) == 2
        assert mock_google.called
        assert mock_yahoo.called

    @patch.object(NewsAggregator, 'fetch_firecrawl_news')
    @patch.object(NewsAggregator, 'fetch_kontan_news')
    @patch.object(NewsAggregator, 'fetch_bisnis_news')
    @patch.object(NewsAggregator, 'fetch_cnbc_indonesia_news')
    @patch.object(NewsAggregator, 'fetch_detik_finance_news')
    @patch.object(NewsAggregator, 'fetch_google_news')
    @patch.object(NewsAggregator, 'fetch_yahoo_news')
    def test_fetch_all_deduplication(self, mock_yahoo, mock_google, mock_detik, mock_cnbc, mock_bisnis, mock_kontan, mock_firecrawl):
        """Test deduplication in fetch_all."""
        # Mock all scrapers to return empty
        mock_firecrawl.return_value = []
        mock_kontan.return_value = []
        mock_bisnis.return_value = []
        mock_cnbc.return_value = []
        mock_detik.return_value = []

        # Same title from different sources
        mock_google.return_value = [
            NewsArticle(
                title="Same Title Here",
                content="Google version",
                source="Google",
                url="https://google.com/1",
                published_at=datetime.utcnow(),
                symbol="BBCA",
            )
        ]
        mock_yahoo.return_value = [
            NewsArticle(
                title="Same Title Here",  # Duplicate title
                content="Yahoo version",
                source="Yahoo",
                url="https://yahoo.com/1",
                published_at=datetime.utcnow(),
                symbol="BBCA",
            )
        ]

        articles = self.aggregator.fetch_all("BBCA", max_articles=10)

        # Should deduplicate to 1 article
        assert len(articles) == 1

    @patch.object(NewsAggregator, 'fetch_firecrawl_news')
    @patch.object(NewsAggregator, 'fetch_kontan_news')
    @patch.object(NewsAggregator, 'fetch_bisnis_news')
    @patch.object(NewsAggregator, 'fetch_cnbc_indonesia_news')
    @patch.object(NewsAggregator, 'fetch_detik_finance_news')
    @patch.object(NewsAggregator, 'fetch_google_news')
    @patch.object(NewsAggregator, 'fetch_yahoo_news')
    def test_fetch_all_date_filter(self, mock_yahoo, mock_google, mock_detik, mock_cnbc, mock_bisnis, mock_kontan, mock_firecrawl):
        """Test date filtering in fetch_all."""
        # Mock all scrapers to return empty
        mock_firecrawl.return_value = []
        mock_kontan.return_value = []
        mock_bisnis.return_value = []
        mock_cnbc.return_value = []
        mock_detik.return_value = []

        old_date = datetime.utcnow() - timedelta(days=30)

        mock_google.return_value = [
            NewsArticle(
                title="Old News",
                content="Content",
                source="Google",
                url="https://google.com/1",
                published_at=old_date,  # Too old
                symbol="BBCA",
            ),
            NewsArticle(
                title="New News",
                content="Content",
                source="Google",
                url="https://google.com/2",
                published_at=datetime.utcnow(),  # Recent
                symbol="BBCA",
            ),
        ]
        mock_yahoo.return_value = []

        articles = self.aggregator.fetch_all("BBCA", max_articles=10, days_back=7)

        # Old news should be filtered out
        assert len(articles) == 1
        assert articles[0].title == "New News"


class TestMockNewsSource:
    """Test mock news source for testing."""

    def test_fetch_news_known_stock(self):
        """Test mock news for known stock."""
        mock = MockNewsSource()
        articles = mock.fetch_news("BBCA", max_articles=5)

        assert len(articles) > 0
        assert all(a.symbol == "BBCA" for a in articles)

    def test_fetch_news_unknown_stock(self):
        """Test mock news for unknown stock."""
        mock = MockNewsSource()
        articles = mock.fetch_news("XXXX", max_articles=5)

        # Should return default market news
        assert len(articles) > 0

    def test_fetch_news_limit(self):
        """Test article limit."""
        mock = MockNewsSource()
        articles = mock.fetch_news("BBCA", max_articles=2)

        assert len(articles) <= 2


class TestSentimentCLI:
    """Test sentiment CLI commands."""

    def setup_method(self):
        """Setup for each test."""
        from stockai.cli.main import app
        self.runner = CliRunner()
        self.app = app

    @patch('stockai.core.sentiment.news.NewsAggregator.fetch_all')
    @patch('stockai.core.sentiment.get_sentiment_analyzer')
    def test_sentiment_analyze_command(self, mock_get_analyzer, mock_fetch):
        """Test sentiment analyze command."""
        # Setup mocks
        mock_fetch.return_value = [
            NewsArticle(
                title="Test News",
                content="Content",
                source="Test",
                url="https://test.com",
                published_at=datetime.utcnow(),
                symbol="BBCA",
            )
        ]

        # Mock the analyzer returned by get_sentiment_analyzer
        mock_analyzer = MagicMock()
        mock_analyzer.aggregate_sentiment.return_value = AggregatedSentiment(
            symbol="BBCA",
            article_count=1,
            avg_sentiment_score=0.3,
            bullish_count=1,
            bearish_count=0,
            neutral_count=0,
            confidence=0.7,
            dominant_label=SentimentLabel.BULLISH,
        )
        mock_get_analyzer.return_value = mock_analyzer

        result = self.runner.invoke(self.app, ["sentiment", "analyze", "BBCA"])

        assert result.exit_code == 0
        assert "BBCA" in result.stdout
        assert "BULLISH" in result.stdout

    @patch('stockai.core.sentiment.news.NewsAggregator.fetch_all')
    def test_sentiment_analyze_no_news(self, mock_fetch):
        """Test sentiment analyze with no news."""
        mock_fetch.return_value = []

        result = self.runner.invoke(self.app, ["sentiment", "analyze", "XXXX"])

        assert result.exit_code == 0
        assert "No recent news" in result.stdout

    @patch('stockai.core.sentiment.news.NewsAggregator.fetch_all')
    def test_sentiment_news_command(self, mock_fetch):
        """Test sentiment news command."""
        mock_fetch.return_value = [
            NewsArticle(
                title="News Title 1",
                content="Content",
                source="Source A",
                url="https://test.com/1",
                published_at=datetime.utcnow(),
                symbol="BBCA",
            ),
            NewsArticle(
                title="News Title 2",
                content="Content",
                source="Source B",
                url="https://test.com/2",
                published_at=datetime.utcnow(),
                symbol="BBCA",
            ),
        ]

        result = self.runner.invoke(self.app, ["sentiment", "news", "BBCA"])

        assert result.exit_code == 0
        assert "News Title 1" in result.stdout
        assert "News Title 2" in result.stdout

    @patch('stockai.core.sentiment.news.NewsAggregator.get_market_news')
    @patch('stockai.core.sentiment.get_sentiment_analyzer')
    def test_sentiment_market_command(self, mock_get_analyzer, mock_fetch):
        """Test sentiment market command."""
        mock_fetch.return_value = [
            NewsArticle(
                title="Market News",
                content="Content",
                source="Test",
                url="https://test.com",
                symbol="IHSG",
            )
        ]

        # Mock the analyzer returned by get_sentiment_analyzer
        mock_analyzer = MagicMock()
        mock_analyzer.aggregate_sentiment.return_value = AggregatedSentiment(
            symbol="IHSG",
            article_count=1,
            avg_sentiment_score=0.2,
            bullish_count=1,
            bearish_count=0,
            neutral_count=0,
            confidence=0.6,
            dominant_label=SentimentLabel.BULLISH,
        )
        mock_get_analyzer.return_value = mock_analyzer

        result = self.runner.invoke(self.app, ["sentiment", "market"])

        assert result.exit_code == 0
        assert "IHSG" in result.stdout


class TestSentimentIntegration:
    """Integration tests for sentiment with prediction."""

    @patch('stockai.core.sentiment.news.NewsAggregator.fetch_all')
    @patch('stockai.core.sentiment.analyzer._get_pipeline')
    def test_ensemble_with_sentiment(self, mock_pipeline, mock_fetch):
        """Test ensemble predictor with sentiment integration."""
        import pandas as pd
        import numpy as np

        # Mock news
        mock_fetch.return_value = [
            NewsArticle(
                title="Bullish news",
                content="Profit growth strong",
                source="Test",
                url="https://test.com",
                symbol="BBCA",
            )
        ]

        # Mock sentiment pipeline to use fallback
        mock_pipeline.return_value = "fallback"

        # Create test data
        dates = pd.date_range(end=datetime.utcnow(), periods=100, freq='D')
        df = pd.DataFrame({
            'date': dates,
            'open': np.random.uniform(9000, 10000, 100),
            'high': np.random.uniform(10000, 11000, 100),
            'low': np.random.uniform(8500, 9000, 100),
            'close': np.random.uniform(9000, 10000, 100),
            'volume': np.random.randint(1000000, 5000000, 100),
        })

        from stockai.core.predictor.ensemble import EnsemblePredictor

        # Create ensemble without trained models
        ensemble = EnsemblePredictor()

        # Test prediction with sentiment
        result = ensemble.predict_with_sentiment(df, "BBCA", use_news=True)

        # Should return a result even without trained models
        assert "direction" in result
        assert "probability" in result

    def test_analyzer_get_prediction_feature_integration(self):
        """Test that prediction features work with mock data."""
        mock = MockNewsSource()
        articles = mock.fetch_news("BBCA", max_articles=5)

        analyzer = SentimentAnalyzer()

        # Use keyword-based sentiment (no model)
        with patch.object(analyzer, '_model_sentiment') as mock_model:
            mock_model.side_effect = lambda x: analyzer._keyword_sentiment(x)
            features = analyzer.get_prediction_feature(articles)

        assert "sentiment_score" in features
        assert -1 <= features["sentiment_score"] <= 1
        assert 0 <= features["bullish_ratio"] <= 1
        assert 0 <= features["bearish_ratio"] <= 1
