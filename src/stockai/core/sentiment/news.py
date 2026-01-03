"""News Aggregator for Indonesian Stock News.

Fetches news from multiple sources:
1. Google News RSS feeds
2. Yahoo Finance news
3. Indonesian financial news sites (mock/placeholder)
"""

import logging
import re
from datetime import datetime, timedelta
from typing import Any

import feedparser
import requests
from bs4 import BeautifulSoup

from stockai.core.sentiment.models import NewsArticle

logger = logging.getLogger(__name__)


class NewsAggregator:
    """Aggregates news from multiple sources.

    Sources:
    - Google News RSS (search-based)
    - Yahoo Finance news API
    - Indonesian financial sites (kontan.co.id, bisnis.com)
    """

    GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=id&gl=ID&ceid=ID:id"

    # Indonesian stock company names for better search
    IDX_COMPANY_NAMES = {
        "BBCA": "Bank Central Asia",
        "BBRI": "Bank Rakyat Indonesia",
        "BMRI": "Bank Mandiri",
        "TLKM": "Telkom Indonesia",
        "ASII": "Astra International",
        "UNVR": "Unilever Indonesia",
        "ICBP": "Indofood CBP",
        "INDF": "Indofood Sukses Makmur",
        "GGRM": "Gudang Garam",
        "HMSP": "HM Sampoerna",
        "KLBF": "Kalbe Farma",
        "PGAS": "Perusahaan Gas Negara",
        "JSMR": "Jasa Marga",
        "ADRO": "Adaro Energy",
        "PTBA": "Bukit Asam",
        "ANTM": "Aneka Tambang",
        "INCO": "Vale Indonesia",
        "SMGR": "Semen Indonesia",
        "CPIN": "Charoen Pokphand",
        "UNTR": "United Tractors",
    }

    def __init__(self, timeout: int = 10):
        """Initialize news aggregator.

        Args:
            timeout: Request timeout in seconds
        """
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })

    def _get_search_query(self, symbol: str) -> str:
        """Get search query for symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Search query string
        """
        symbol = symbol.upper().replace(".JK", "")

        # Get company name if known
        company_name = self.IDX_COMPANY_NAMES.get(symbol)

        if company_name:
            # Search for both symbol and company name
            return f'"{symbol}" OR "{company_name}" saham'
        else:
            # Generic Indonesian stock search
            return f'"{symbol}" saham IDX'

    def _parse_date(self, date_str: str) -> datetime | None:
        """Parse date from various formats.

        Args:
            date_str: Date string

        Returns:
            Datetime or None
        """
        formats = [
            "%a, %d %b %Y %H:%M:%S %Z",
            "%a, %d %b %Y %H:%M:%S %z",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue

        return None

    def _clean_html(self, html: str) -> str:
        """Clean HTML content to plain text.

        Args:
            html: HTML string

        Returns:
            Clean text
        """
        if not html:
            return ""

        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(separator=" ", strip=True)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def fetch_google_news(
        self,
        symbol: str,
        max_articles: int = 10,
    ) -> list[NewsArticle]:
        """Fetch news from Google News RSS.

        Args:
            symbol: Stock symbol
            max_articles: Maximum articles to fetch

        Returns:
            List of NewsArticle
        """
        query = self._get_search_query(symbol)
        url = self.GOOGLE_NEWS_RSS.format(query=query.replace(" ", "+"))

        articles = []

        try:
            feed = feedparser.parse(url)

            for entry in feed.entries[:max_articles]:
                title = entry.get("title", "")
                content = self._clean_html(entry.get("summary", ""))
                link = entry.get("link", "")
                published = entry.get("published", "")

                # Parse source from title (Google News format: "Title - Source")
                source = "Google News"
                if " - " in title:
                    parts = title.rsplit(" - ", 1)
                    title = parts[0]
                    source = parts[1] if len(parts) > 1 else source

                article = NewsArticle(
                    title=title,
                    content=content or title,  # Use title if no content
                    source=source,
                    url=link,
                    published_at=self._parse_date(published),
                    symbol=symbol.upper(),
                )
                articles.append(article)

            logger.info(f"Fetched {len(articles)} articles from Google News for {symbol}")

        except Exception as e:
            logger.warning(f"Error fetching Google News: {e}")

        return articles

    def fetch_yahoo_news(
        self,
        symbol: str,
        max_articles: int = 5,
    ) -> list[NewsArticle]:
        """Fetch news from Yahoo Finance.

        Args:
            symbol: Stock symbol
            max_articles: Maximum articles to fetch

        Returns:
            List of NewsArticle
        """
        # Add .JK suffix for Indonesian stocks
        yahoo_symbol = symbol.upper()
        if not yahoo_symbol.endswith(".JK"):
            yahoo_symbol = f"{yahoo_symbol}.JK"

        articles = []

        try:
            # Yahoo Finance news RSS
            url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={yahoo_symbol}"
            feed = feedparser.parse(url)

            for entry in feed.entries[:max_articles]:
                title = entry.get("title", "")
                content = self._clean_html(entry.get("summary", ""))
                link = entry.get("link", "")
                published = entry.get("published", "")

                article = NewsArticle(
                    title=title,
                    content=content or title,
                    source="Yahoo Finance",
                    url=link,
                    published_at=self._parse_date(published),
                    symbol=symbol.upper(),
                )
                articles.append(article)

            logger.info(f"Fetched {len(articles)} articles from Yahoo Finance for {symbol}")

        except Exception as e:
            logger.warning(f"Error fetching Yahoo Finance news: {e}")

        return articles

    def fetch_all(
        self,
        symbol: str,
        max_articles: int = 15,
        days_back: int = 7,
    ) -> list[NewsArticle]:
        """Fetch news from all sources.

        Args:
            symbol: Stock symbol
            max_articles: Maximum total articles
            days_back: Only include articles from last N days

        Returns:
            List of NewsArticle, deduplicated and sorted
        """
        all_articles = []

        # Fetch from each source
        google_articles = self.fetch_google_news(symbol, max_articles=10)
        yahoo_articles = self.fetch_yahoo_news(symbol, max_articles=5)

        all_articles.extend(google_articles)
        all_articles.extend(yahoo_articles)

        # Filter by date (use timezone-aware cutoff for comparison)
        from datetime import timezone
        cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
        filtered = []
        for a in all_articles:
            if a.published_at is None:
                filtered.append(a)
            else:
                # Make naive datetime timezone-aware for comparison
                pub_dt = a.published_at
                if pub_dt.tzinfo is None:
                    pub_dt = pub_dt.replace(tzinfo=timezone.utc)
                if pub_dt > cutoff:
                    filtered.append(a)

        # Deduplicate by title similarity
        seen_titles = set()
        unique_articles = []

        for article in filtered:
            # Simple dedup by normalized title
            title_key = article.title.lower().strip()[:50]
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_articles.append(article)

        # Sort by date (newest first)
        def get_sort_date(article):
            if article.published_at is None:
                return datetime.min.replace(tzinfo=timezone.utc)
            dt = article.published_at
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt

        unique_articles.sort(key=get_sort_date, reverse=True)

        # Limit total
        result = unique_articles[:max_articles]

        logger.info(f"Total {len(result)} unique articles for {symbol}")
        return result

    def get_market_news(
        self,
        max_articles: int = 10,
    ) -> list[NewsArticle]:
        """Fetch general Indonesian market news.

        Args:
            max_articles: Maximum articles to fetch

        Returns:
            List of NewsArticle
        """
        # Search for general IDX/market news
        queries = [
            "IHSG saham Indonesia",
            "Bursa Efek Indonesia BEI",
        ]

        all_articles = []

        for query in queries:
            url = self.GOOGLE_NEWS_RSS.format(query=query.replace(" ", "+"))

            try:
                feed = feedparser.parse(url)

                for entry in feed.entries[:5]:
                    title = entry.get("title", "")
                    content = self._clean_html(entry.get("summary", ""))
                    link = entry.get("link", "")
                    published = entry.get("published", "")

                    source = "Google News"
                    if " - " in title:
                        parts = title.rsplit(" - ", 1)
                        title = parts[0]
                        source = parts[1] if len(parts) > 1 else source

                    article = NewsArticle(
                        title=title,
                        content=content or title,
                        source=source,
                        url=link,
                        published_at=self._parse_date(published),
                        symbol="IHSG",
                    )
                    all_articles.append(article)

            except Exception as e:
                logger.warning(f"Error fetching market news: {e}")

        # Deduplicate
        seen = set()
        unique = []
        for a in all_articles:
            key = a.title.lower()[:50]
            if key not in seen:
                seen.add(key)
                unique.append(a)

        return unique[:max_articles]


class MockNewsSource:
    """Mock news source for testing without network.

    Provides realistic sample news articles.
    """

    SAMPLE_NEWS = {
        "BBCA": [
            {
                "title": "Bank BCA Catat Laba Bersih Rp 40 Triliun, Naik 15%",
                "content": "PT Bank Central Asia Tbk (BBCA) mencatat pertumbuhan laba bersih yang solid pada tahun ini. Kinerja positif didorong oleh peningkatan kredit dan fee-based income.",
                "source": "Kontan",
                "sentiment": "bullish",
            },
            {
                "title": "Analis: Saham BBCA Masih Overvalued, Target Harga Diturunkan",
                "content": "Beberapa analis menurunkan target harga saham BBCA setelah valuasi dinilai terlalu tinggi dibanding peers.",
                "source": "Bisnis Indonesia",
                "sentiment": "bearish",
            },
            {
                "title": "BCA Ekspansi Kredit UMKM di Kuartal IV",
                "content": "Bank BCA menargetkan pertumbuhan kredit UMKM hingga 12% tahun ini sebagai bagian dari strategi diversifikasi portofolio.",
                "source": "CNBC Indonesia",
                "sentiment": "neutral",
            },
        ],
        "BBRI": [
            {
                "title": "BRI Bagikan Dividen Rp 288 per Saham",
                "content": "PT Bank Rakyat Indonesia Tbk (BBRI) mengumumkan pembagian dividen tunai sebesar Rp 288 per saham, setara dengan 85% dari laba bersih.",
                "source": "IDN Financials",
                "sentiment": "bullish",
            },
            {
                "title": "Kredit Macet BRI Naik, NPL Tembus 3%",
                "content": "Rasio kredit bermasalah atau Non-Performing Loan (NPL) BRI naik menjadi 3.2% akibat tekanan ekonomi.",
                "source": "Kontan",
                "sentiment": "bearish",
            },
        ],
        "DEFAULT": [
            {
                "title": "IHSG Menguat Didorong Sentimen Global Positif",
                "content": "Indeks Harga Saham Gabungan (IHSG) ditutup menguat 0.8% didorong oleh sentimen positif dari pasar global.",
                "source": "Detik Finance",
                "sentiment": "bullish",
            },
            {
                "title": "Investor Asing Net Sell Rp 500 Miliar",
                "content": "Investor asing tercatat melakukan aksi jual bersih sebesar Rp 500 miliar di pasar saham Indonesia.",
                "source": "Bisnis.com",
                "sentiment": "bearish",
            },
        ],
    }

    def fetch_news(
        self,
        symbol: str,
        max_articles: int = 5,
    ) -> list[NewsArticle]:
        """Fetch mock news articles.

        Args:
            symbol: Stock symbol
            max_articles: Maximum articles

        Returns:
            List of mock NewsArticle
        """
        symbol = symbol.upper().replace(".JK", "")
        news_data = self.SAMPLE_NEWS.get(symbol, self.SAMPLE_NEWS["DEFAULT"])

        articles = []
        for i, item in enumerate(news_data[:max_articles]):
            article = NewsArticle(
                title=item["title"],
                content=item["content"],
                source=item["source"],
                url=f"https://example.com/news/{symbol.lower()}/{i+1}",
                published_at=datetime.utcnow() - timedelta(hours=i*6),
                symbol=symbol,
            )
            articles.append(article)

        return articles
