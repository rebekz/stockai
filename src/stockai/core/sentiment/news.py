"""News Aggregator for Indonesian Stock News.

Fetches news from multiple sources:
1. Google News RSS feeds
2. Yahoo Finance news
3. Indonesian financial news sites via Firecrawl (Kontan, Bisnis, CNBC Indonesia, Detik Finance)
"""

import logging
import re
from datetime import datetime, timedelta
from typing import Any

import feedparser
import requests
from bs4 import BeautifulSoup

from stockai.config import get_settings
from stockai.core.sentiment.models import NewsArticle

logger = logging.getLogger(__name__)

# Lazy import Firecrawl
_firecrawl_client = None


def _get_firecrawl_client():
    """Lazy load Firecrawl client."""
    global _firecrawl_client
    if _firecrawl_client is None:
        settings = get_settings()
        if not settings.has_firecrawl_api:
            logger.debug("Firecrawl API key not configured")
            return None

        try:
            from firecrawl import FirecrawlApp
            _firecrawl_client = FirecrawlApp(api_key=settings.firecrawl_api_key)
            logger.info("Initialized Firecrawl client")
        except ImportError:
            logger.warning("Firecrawl not installed, using fallback scrapers")
            return None
        except Exception as e:
            logger.warning(f"Could not initialize Firecrawl: {e}")
            return None

    return _firecrawl_client


class NewsAggregator:
    """Aggregates news from multiple sources.

    Sources:
    - Google News RSS (search-based)
    - Yahoo Finance news API
    - Indonesian financial sites (kontan.co.id, bisnis.com)
    """

    GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=id&gl=ID&ceid=ID:id"

    # Indonesian financial news sources
    KONTAN_SEARCH_URL = "https://www.kontan.co.id/search/?search={query}"
    BISNIS_SEARCH_URL = "https://www.bisnis.com/index.php/search?q={query}"
    CNBC_SEARCH_URL = "https://www.cnbcindonesia.com/search?query={query}"
    DETIK_SEARCH_URL = "https://www.detik.com/search/searchall?query={query}&siteid=3"  # siteid=3 = finance

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

    def fetch_kontan_news(
        self,
        symbol: str,
        max_articles: int = 5,
    ) -> list[NewsArticle]:
        """Fetch news from Kontan.co.id.

        Args:
            symbol: Stock symbol
            max_articles: Maximum articles to fetch

        Returns:
            List of NewsArticle
        """
        symbol = symbol.upper().replace(".JK", "")
        company_name = self.IDX_COMPANY_NAMES.get(symbol, symbol)
        query = f"{symbol} {company_name}".replace(" ", "+")
        url = self.KONTAN_SEARCH_URL.format(query=query)

        articles = []

        try:
            response = self._session.get(url, timeout=self._timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Kontan search results structure
            news_items = soup.select("div.isi-news-plg, div.list-news li, article.news-item")[:max_articles]

            for item in news_items:
                # Try different selectors for title/link
                title_elem = item.select_one("a.news-title, h2 a, h3 a, a")
                if not title_elem:
                    continue

                title = title_elem.get_text(strip=True)
                link = title_elem.get("href", "")

                # Make absolute URL
                if link and not link.startswith("http"):
                    link = f"https://www.kontan.co.id{link}"

                # Get summary/content
                content_elem = item.select_one("p, div.news-desc, span.desc")
                content = content_elem.get_text(strip=True) if content_elem else title

                # Get date
                date_elem = item.select_one("span.date, time, div.date")
                published_at = None
                if date_elem:
                    date_text = date_elem.get_text(strip=True)
                    published_at = self._parse_indonesian_date(date_text)

                if title and len(title) > 10:
                    article = NewsArticle(
                        title=title,
                        content=content or title,
                        source="Kontan",
                        url=link,
                        published_at=published_at,
                        symbol=symbol,
                    )
                    articles.append(article)

            logger.info(f"Fetched {len(articles)} articles from Kontan for {symbol}")

        except Exception as e:
            logger.warning(f"Error fetching Kontan news: {e}")

        return articles

    def fetch_bisnis_news(
        self,
        symbol: str,
        max_articles: int = 5,
    ) -> list[NewsArticle]:
        """Fetch news from Bisnis.com.

        Args:
            symbol: Stock symbol
            max_articles: Maximum articles to fetch

        Returns:
            List of NewsArticle
        """
        symbol = symbol.upper().replace(".JK", "")
        company_name = self.IDX_COMPANY_NAMES.get(symbol, symbol)
        query = f"{symbol} {company_name}".replace(" ", "+")
        url = self.BISNIS_SEARCH_URL.format(query=query)

        articles = []

        try:
            response = self._session.get(url, timeout=self._timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Bisnis.com search results structure
            news_items = soup.select("div.col-sm-8 article, div.list-news li, div.search-result-item")[:max_articles]

            for item in news_items:
                title_elem = item.select_one("h4 a, h3 a, h2 a, a.title")
                if not title_elem:
                    continue

                title = title_elem.get_text(strip=True)
                link = title_elem.get("href", "")

                if link and not link.startswith("http"):
                    link = f"https://www.bisnis.com{link}"

                content_elem = item.select_one("p, div.description, span.lead")
                content = content_elem.get_text(strip=True) if content_elem else title

                date_elem = item.select_one("span.date, time, div.time")
                published_at = None
                if date_elem:
                    date_text = date_elem.get_text(strip=True)
                    published_at = self._parse_indonesian_date(date_text)

                if title and len(title) > 10:
                    article = NewsArticle(
                        title=title,
                        content=content or title,
                        source="Bisnis.com",
                        url=link,
                        published_at=published_at,
                        symbol=symbol,
                    )
                    articles.append(article)

            logger.info(f"Fetched {len(articles)} articles from Bisnis.com for {symbol}")

        except Exception as e:
            logger.warning(f"Error fetching Bisnis.com news: {e}")

        return articles

    def fetch_cnbc_indonesia_news(
        self,
        symbol: str,
        max_articles: int = 5,
    ) -> list[NewsArticle]:
        """Fetch news from CNBC Indonesia.

        Args:
            symbol: Stock symbol
            max_articles: Maximum articles to fetch

        Returns:
            List of NewsArticle
        """
        symbol = symbol.upper().replace(".JK", "")
        company_name = self.IDX_COMPANY_NAMES.get(symbol, symbol)
        query = f"{symbol} {company_name}".replace(" ", "+")
        url = self.CNBC_SEARCH_URL.format(query=query)

        articles = []

        try:
            response = self._session.get(url, timeout=self._timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # CNBC Indonesia search results structure
            news_items = soup.select("article, div.list li, div.media-object")[:max_articles]

            for item in news_items:
                title_elem = item.select_one("h2 a, h3 a, h4 a, a.title, a")
                if not title_elem:
                    continue

                title = title_elem.get_text(strip=True)
                link = title_elem.get("href", "")

                if link and not link.startswith("http"):
                    link = f"https://www.cnbcindonesia.com{link}"

                content_elem = item.select_one("p, div.desc, span.description")
                content = content_elem.get_text(strip=True) if content_elem else title

                date_elem = item.select_one("span.date, time, div.date, span.time")
                published_at = None
                if date_elem:
                    date_text = date_elem.get_text(strip=True)
                    published_at = self._parse_indonesian_date(date_text)

                if title and len(title) > 10:
                    article = NewsArticle(
                        title=title,
                        content=content or title,
                        source="CNBC Indonesia",
                        url=link,
                        published_at=published_at,
                        symbol=symbol,
                    )
                    articles.append(article)

            logger.info(f"Fetched {len(articles)} articles from CNBC Indonesia for {symbol}")

        except Exception as e:
            logger.warning(f"Error fetching CNBC Indonesia news: {e}")

        return articles

    def fetch_detik_finance_news(
        self,
        symbol: str,
        max_articles: int = 5,
    ) -> list[NewsArticle]:
        """Fetch news from Detik Finance.

        Args:
            symbol: Stock symbol
            max_articles: Maximum articles to fetch

        Returns:
            List of NewsArticle
        """
        symbol = symbol.upper().replace(".JK", "")
        company_name = self.IDX_COMPANY_NAMES.get(symbol, symbol)
        query = f"{symbol} {company_name} saham".replace(" ", "+")
        url = self.DETIK_SEARCH_URL.format(query=query)

        articles = []

        try:
            response = self._session.get(url, timeout=self._timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Detik search results structure
            news_items = soup.select("article, div.list-content__item, div.media")[:max_articles]

            for item in news_items:
                title_elem = item.select_one("h2 a, h3 a, a.media__link, a")
                if not title_elem:
                    continue

                title = title_elem.get_text(strip=True)
                link = title_elem.get("href", "")

                content_elem = item.select_one("p, div.media__desc, span.media__summary")
                content = content_elem.get_text(strip=True) if content_elem else title

                date_elem = item.select_one("span.date, span.media__date, time")
                published_at = None
                if date_elem:
                    date_text = date_elem.get_text(strip=True)
                    published_at = self._parse_indonesian_date(date_text)

                if title and len(title) > 10:
                    article = NewsArticle(
                        title=title,
                        content=content or title,
                        source="Detik Finance",
                        url=link,
                        published_at=published_at,
                        symbol=symbol,
                    )
                    articles.append(article)

            logger.info(f"Fetched {len(articles)} articles from Detik Finance for {symbol}")

        except Exception as e:
            logger.warning(f"Error fetching Detik Finance news: {e}")

        return articles

    def _parse_indonesian_date(self, date_str: str) -> datetime | None:
        """Parse Indonesian date formats.

        Args:
            date_str: Date string in Indonesian format

        Returns:
            Datetime or None
        """
        if not date_str:
            return None

        # Clean the date string
        date_str = date_str.strip()

        # Indonesian month names
        id_months = {
            "januari": "01", "februari": "02", "maret": "03", "april": "04",
            "mei": "05", "juni": "06", "juli": "07", "agustus": "08",
            "september": "09", "oktober": "10", "november": "11", "desember": "12",
            "jan": "01", "feb": "02", "mar": "03", "apr": "04",
            "jun": "06", "jul": "07", "agu": "08", "ags": "08",
            "sep": "09", "okt": "10", "nov": "11", "des": "12",
        }

        try:
            # Handle relative dates
            date_lower = date_str.lower()
            if "hari ini" in date_lower or "today" in date_lower:
                return datetime.now()
            if "kemarin" in date_lower or "yesterday" in date_lower:
                return datetime.now() - timedelta(days=1)
            if "jam lalu" in date_lower or "hour ago" in date_lower:
                match = re.search(r"(\d+)", date_str)
                if match:
                    hours = int(match.group(1))
                    return datetime.now() - timedelta(hours=hours)
            if "menit lalu" in date_lower or "minute ago" in date_lower:
                match = re.search(r"(\d+)", date_str)
                if match:
                    minutes = int(match.group(1))
                    return datetime.now() - timedelta(minutes=minutes)

            # Replace Indonesian month names with numbers
            for id_month, num in id_months.items():
                date_str = re.sub(rf"\b{id_month}\b", num, date_str, flags=re.IGNORECASE)

            # Try various date patterns
            patterns = [
                r"(\d{1,2})[/\-\s](\d{1,2})[/\-\s](\d{4})",  # DD/MM/YYYY
                r"(\d{4})[/\-\s](\d{1,2})[/\-\s](\d{1,2})",  # YYYY/MM/DD
                r"(\d{1,2})\s+(\d{2})\s+(\d{4})",  # DD MM YYYY
            ]

            for pattern in patterns:
                match = re.search(pattern, date_str)
                if match:
                    groups = match.groups()
                    if len(groups[0]) == 4:  # YYYY first
                        year, month, day = groups
                    else:  # DD first
                        day, month, year = groups
                    return datetime(int(year), int(month), int(day))

            # Try standard parsing as fallback
            return self._parse_date(date_str)

        except Exception:
            return None

    def fetch_firecrawl_news(
        self,
        symbol: str,
        max_articles: int = 10,
    ) -> list[NewsArticle]:
        """Fetch news using Firecrawl search API.

        Uses Firecrawl to search Indonesian financial news sites
        for articles about the given stock.

        Args:
            symbol: Stock symbol
            max_articles: Maximum articles to fetch

        Returns:
            List of NewsArticle
        """
        client = _get_firecrawl_client()
        if client is None:
            logger.debug("Firecrawl not available, skipping")
            return []

        symbol = symbol.upper().replace(".JK", "")
        company_name = self.IDX_COMPANY_NAMES.get(symbol, "")

        # Build search query
        if company_name:
            query = f"{symbol} OR {company_name} saham Indonesia"
        else:
            query = f"{symbol} saham IDX Indonesia"

        articles = []

        try:
            # Use Firecrawl search with Indonesian financial sites
            search_result = client.search(
                query=query,
                limit=max_articles,
                scrape_options={
                    "formats": ["markdown"],
                    "onlyMainContent": True,
                },
            )

            if not search_result or "data" not in search_result:
                return []

            for item in search_result.get("data", [])[:max_articles]:
                title = item.get("title", "")
                url = item.get("url", "")
                content = item.get("markdown", "") or item.get("description", "")

                # Extract source from URL
                source = "Firecrawl"
                if "kontan.co.id" in url:
                    source = "Kontan"
                elif "bisnis.com" in url:
                    source = "Bisnis.com"
                elif "cnbcindonesia.com" in url:
                    source = "CNBC Indonesia"
                elif "detik.com" in url:
                    source = "Detik Finance"
                elif "kompas.com" in url:
                    source = "Kompas"
                elif "idnfinancials" in url:
                    source = "IDN Financials"
                elif "yahoo" in url:
                    source = "Yahoo Finance"

                # Parse published date if available
                published_at = None
                if item.get("publishedDate"):
                    published_at = self._parse_date(item["publishedDate"])

                # Clean content - take first 500 chars for summary
                if content and len(content) > 500:
                    content = content[:500] + "..."

                if title and len(title) > 10:
                    article = NewsArticle(
                        title=title,
                        content=content or title,
                        source=source,
                        url=url,
                        published_at=published_at,
                        symbol=symbol,
                    )
                    articles.append(article)

            logger.info(f"Fetched {len(articles)} articles from Firecrawl for {symbol}")

        except Exception as e:
            logger.warning(f"Error fetching Firecrawl news: {e}")

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

        # Try Firecrawl first (best quality, searches all Indonesian sources)
        firecrawl_articles = self.fetch_firecrawl_news(symbol, max_articles=max_articles)
        all_articles.extend(firecrawl_articles)

        # Supplement with RSS feeds for broader coverage
        google_articles = self.fetch_google_news(symbol, max_articles=10)
        yahoo_articles = self.fetch_yahoo_news(symbol, max_articles=5)
        all_articles.extend(google_articles)
        all_articles.extend(yahoo_articles)

        # If Firecrawl didn't return enough, use fallback scrapers
        if len(firecrawl_articles) < 5:
            logger.debug("Firecrawl returned few articles, using fallback scrapers")
            kontan_articles = self.fetch_kontan_news(symbol, max_articles=3)
            bisnis_articles = self.fetch_bisnis_news(symbol, max_articles=3)
            cnbc_articles = self.fetch_cnbc_indonesia_news(symbol, max_articles=3)
            detik_articles = self.fetch_detik_finance_news(symbol, max_articles=3)

            all_articles.extend(kontan_articles)
            all_articles.extend(bisnis_articles)
            all_articles.extend(cnbc_articles)
            all_articles.extend(detik_articles)

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
