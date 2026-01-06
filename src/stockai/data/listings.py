"""IDX Stock Listings Database.

Provides comprehensive list of Indonesian stocks with search capabilities.
Updated with IDX30, LQ45, and other major stocks as of 2025.
"""

import logging
import re
from difflib import SequenceMatcher
from typing import Any

logger = logging.getLogger(__name__)


# IDX30 stocks (most liquid 30 stocks)
IDX30_STOCKS = [
    {"symbol": "ACES", "name": "Ace Hardware Indonesia", "sector": "Consumer Cyclicals"},
    {"symbol": "ADRO", "name": "Adaro Energy Indonesia", "sector": "Energy"},
    {"symbol": "AMRT", "name": "Sumber Alfaria Trijaya", "sector": "Consumer Cyclicals"},
    {"symbol": "ANTM", "name": "Aneka Tambang", "sector": "Basic Materials"},
    {"symbol": "ASII", "name": "Astra International", "sector": "Consumer Cyclicals"},
    {"symbol": "BBCA", "name": "Bank Central Asia", "sector": "Finance"},
    {"symbol": "BBNI", "name": "Bank Negara Indonesia", "sector": "Finance"},
    {"symbol": "BBRI", "name": "Bank Rakyat Indonesia", "sector": "Finance"},
    {"symbol": "BBTN", "name": "Bank Tabungan Negara", "sector": "Finance"},
    {"symbol": "BMRI", "name": "Bank Mandiri", "sector": "Finance"},
    {"symbol": "BRPT", "name": "Barito Pacific", "sector": "Basic Materials"},
    {"symbol": "BUKA", "name": "Bukalapak.com", "sector": "Technology"},
    {"symbol": "CPIN", "name": "Charoen Pokphand Indonesia", "sector": "Consumer Non-Cyclicals"},
    {"symbol": "EMTK", "name": "Elang Mahkota Teknologi", "sector": "Technology"},
    {"symbol": "ESSA", "name": "Surya Esa Perkasa", "sector": "Energy"},
    {"symbol": "GOTO", "name": "GoTo Gojek Tokopedia", "sector": "Technology"},
    {"symbol": "HRUM", "name": "Harum Energy", "sector": "Energy"},
    {"symbol": "ICBP", "name": "Indofood CBP Sukses Makmur", "sector": "Consumer Non-Cyclicals"},
    {"symbol": "INCO", "name": "Vale Indonesia", "sector": "Basic Materials"},
    {"symbol": "INDF", "name": "Indofood Sukses Makmur", "sector": "Consumer Non-Cyclicals"},
    {"symbol": "INKP", "name": "Indah Kiat Pulp & Paper", "sector": "Basic Materials"},
    {"symbol": "ITMG", "name": "Indo Tambangraya Megah", "sector": "Energy"},
    {"symbol": "KLBF", "name": "Kalbe Farma", "sector": "Consumer Non-Cyclicals"},
    {"symbol": "MDKA", "name": "Merdeka Copper Gold", "sector": "Basic Materials"},
    {"symbol": "PGAS", "name": "Perusahaan Gas Negara", "sector": "Energy"},
    {"symbol": "SMGR", "name": "Semen Indonesia", "sector": "Basic Materials"},
    {"symbol": "TBIG", "name": "Tower Bersama Infrastructure", "sector": "Infrastructures"},
    {"symbol": "TLKM", "name": "Telkom Indonesia", "sector": "Infrastructures"},
    {"symbol": "TOWR", "name": "Sarana Menara Nusantara", "sector": "Infrastructures"},
    {"symbol": "UNTR", "name": "United Tractors", "sector": "Industrials"},
]

# LQ45 stocks (45 most liquid - includes IDX30 plus 15 more)
LQ45_ADDITIONAL = [
    {"symbol": "AKRA", "name": "AKR Corporindo", "sector": "Energy"},
    {"symbol": "BRIS", "name": "Bank Syariah Indonesia", "sector": "Finance"},
    {"symbol": "BSDE", "name": "Bumi Serpong Damai", "sector": "Property & Real Estate"},
    {"symbol": "CTRA", "name": "Ciputra Development", "sector": "Property & Real Estate"},
    {"symbol": "ERAA", "name": "Erajaya Swasembada", "sector": "Consumer Cyclicals"},
    {"symbol": "EXCL", "name": "XL Axiata", "sector": "Infrastructures"},
    {"symbol": "HMSP", "name": "HM Sampoerna", "sector": "Consumer Non-Cyclicals"},
    {"symbol": "INTP", "name": "Indocement Tunggal Prakarsa", "sector": "Basic Materials"},
    {"symbol": "ISAT", "name": "Indosat Ooredoo Hutchison", "sector": "Infrastructures"},
    {"symbol": "JPFA", "name": "Japfa Comfeed Indonesia", "sector": "Consumer Non-Cyclicals"},
    {"symbol": "JSMR", "name": "Jasa Marga", "sector": "Infrastructures"},
    {"symbol": "MAPI", "name": "Mitra Adiperkasa", "sector": "Consumer Cyclicals"},
    {"symbol": "MEDC", "name": "Medco Energi Internasional", "sector": "Energy"},
    {"symbol": "MYOR", "name": "Mayora Indah", "sector": "Consumer Non-Cyclicals"},
    {"symbol": "PTBA", "name": "Bukit Asam", "sector": "Energy"},
]

# JII70 stocks (Jakarta Islamic Index 70 - 70 most liquid sharia-compliant stocks)
# Note: JII70 excludes conventional banks and non-halal businesses
JII70_STOCKS = [
    # Top tier by market cap
    {"symbol": "AMMN", "name": "Amman Mineral Internasional", "sector": "Basic Materials"},
    {"symbol": "TLKM", "name": "Telkom Indonesia", "sector": "Infrastructures"},
    {"symbol": "BYAN", "name": "Bayan Resources", "sector": "Energy"},
    {"symbol": "TPIA", "name": "Chandra Asri Pacific", "sector": "Basic Materials"},
    {"symbol": "ASII", "name": "Astra International", "sector": "Consumer Cyclicals"},
    {"symbol": "GOTO", "name": "GoTo Gojek Tokopedia", "sector": "Technology"},
    {"symbol": "DSSA", "name": "Dian Swastatika Sentosa", "sector": "Energy"},
    {"symbol": "UNTR", "name": "United Tractors", "sector": "Industrials"},
    {"symbol": "INDF", "name": "Indofood Sukses Makmur", "sector": "Consumer Non-Cyclicals"},
    {"symbol": "PANI", "name": "Pantai Indah Kapuk Dua", "sector": "Property & Real Estate"},
    # High market cap sharia stocks
    {"symbol": "BRPT", "name": "Barito Pacific", "sector": "Basic Materials"},
    {"symbol": "BRMS", "name": "Bumi Resources Minerals", "sector": "Basic Materials"},
    {"symbol": "BUMI", "name": "Bumi Resources", "sector": "Energy"},
    {"symbol": "UNVR", "name": "Unilever Indonesia", "sector": "Consumer Non-Cyclicals"},
    {"symbol": "BRIS", "name": "Bank Syariah Indonesia", "sector": "Finance"},
    {"symbol": "ICBP", "name": "Indofood CBP Sukses Makmur", "sector": "Consumer Non-Cyclicals"},
    {"symbol": "ANTM", "name": "Aneka Tambang", "sector": "Basic Materials"},
    {"symbol": "ISAT", "name": "Indosat Ooredoo Hutchison", "sector": "Infrastructures"},
    {"symbol": "CPIN", "name": "Charoen Pokphand Indonesia", "sector": "Consumer Non-Cyclicals"},
    {"symbol": "ADRO", "name": "Adaro Energy Indonesia", "sector": "Energy"},
    # Mid-large cap sharia stocks
    {"symbol": "INCO", "name": "Vale Indonesia", "sector": "Basic Materials"},
    {"symbol": "MDKA", "name": "Merdeka Copper Gold", "sector": "Basic Materials"},
    {"symbol": "KLBF", "name": "Kalbe Farma", "sector": "Consumer Non-Cyclicals"},
    {"symbol": "PTBA", "name": "Bukit Asam", "sector": "Energy"},
    {"symbol": "SMGR", "name": "Semen Indonesia", "sector": "Basic Materials"},
    {"symbol": "INTP", "name": "Indocement Tunggal Prakarsa", "sector": "Basic Materials"},
    {"symbol": "MYOR", "name": "Mayora Indah", "sector": "Consumer Non-Cyclicals"},
    {"symbol": "JPFA", "name": "Japfa Comfeed Indonesia", "sector": "Consumer Non-Cyclicals"},
    {"symbol": "EXCL", "name": "XL Axiata", "sector": "Infrastructures"},
    {"symbol": "JSMR", "name": "Jasa Marga", "sector": "Infrastructures"},
    # Infrastructure & Property sharia stocks
    {"symbol": "TBIG", "name": "Tower Bersama Infrastructure", "sector": "Infrastructures"},
    {"symbol": "TOWR", "name": "Sarana Menara Nusantara", "sector": "Infrastructures"},
    {"symbol": "WIKA", "name": "Wijaya Karya", "sector": "Infrastructures"},
    {"symbol": "WSKT", "name": "Waskita Karya", "sector": "Infrastructures"},
    {"symbol": "CTRA", "name": "Ciputra Development", "sector": "Property & Real Estate"},
    {"symbol": "SMRA", "name": "Summarecon Agung", "sector": "Property & Real Estate"},
    {"symbol": "BSDE", "name": "Bumi Serpong Damai", "sector": "Property & Real Estate"},
    {"symbol": "PWON", "name": "Pakuwon Jati", "sector": "Property & Real Estate"},
    # Consumer & Healthcare sharia stocks
    {"symbol": "SIDO", "name": "Industri Jamu dan Farmasi Sido Muncul", "sector": "Consumer Non-Cyclicals"},
    {"symbol": "MIKA", "name": "Mitra Keluarga Karyasehat", "sector": "Healthcare"},
    {"symbol": "ERAA", "name": "Erajaya Swasembada", "sector": "Consumer Cyclicals"},
    {"symbol": "MAPI", "name": "Mitra Adiperkasa", "sector": "Consumer Cyclicals"},
    {"symbol": "ACES", "name": "Ace Hardware Indonesia", "sector": "Consumer Cyclicals"},
    {"symbol": "AMRT", "name": "Sumber Alfaria Trijaya", "sector": "Consumer Cyclicals"},
    # Energy & Mining sharia stocks
    {"symbol": "MEDC", "name": "Medco Energi Internasional", "sector": "Energy"},
    {"symbol": "HRUM", "name": "Harum Energy", "sector": "Energy"},
    {"symbol": "ITMG", "name": "Indo Tambangraya Megah", "sector": "Energy"},
    {"symbol": "AKRA", "name": "AKR Corporindo", "sector": "Energy"},
    {"symbol": "ESSA", "name": "Surya Esa Perkasa", "sector": "Energy"},
    {"symbol": "PGAS", "name": "Perusahaan Gas Negara", "sector": "Energy"},
    {"symbol": "TINS", "name": "Timah", "sector": "Basic Materials"},
    # Technology & Media sharia stocks
    {"symbol": "EMTK", "name": "Elang Mahkota Teknologi", "sector": "Technology"},
    {"symbol": "BUKA", "name": "Bukalapak.com", "sector": "Technology"},
    {"symbol": "MNCN", "name": "Media Nusantara Citra", "sector": "Consumer Cyclicals"},
    {"symbol": "SCMA", "name": "Surya Citra Media", "sector": "Consumer Cyclicals"},
    # Paper & Materials sharia stocks
    {"symbol": "INKP", "name": "Indah Kiat Pulp & Paper", "sector": "Basic Materials"},
    {"symbol": "TKIM", "name": "Pabrik Kertas Tjiwi Kimia", "sector": "Basic Materials"},
    # Additional liquid sharia stocks
    {"symbol": "BTPS", "name": "Bank BTPN Syariah", "sector": "Finance"},
    {"symbol": "SRTG", "name": "Saratoga Investama Sedaya", "sector": "Finance"},
    {"symbol": "MLPT", "name": "Multipolar Technology", "sector": "Technology"},
    {"symbol": "KPIG", "name": "MNC Land", "sector": "Property & Real Estate"},
    {"symbol": "BMTR", "name": "Global Mediacom", "sector": "Consumer Cyclicals"},
    {"symbol": "LSIP", "name": "PP London Sumatra Indonesia", "sector": "Consumer Non-Cyclicals"},
    {"symbol": "AALI", "name": "Astra Agro Lestari", "sector": "Consumer Non-Cyclicals"},
    {"symbol": "SSMS", "name": "Sawit Sumbermas Sarana", "sector": "Consumer Non-Cyclicals"},
    {"symbol": "SILO", "name": "Siloam International Hospitals", "sector": "Healthcare"},
    {"symbol": "AUTO", "name": "Astra Otoparts", "sector": "Consumer Cyclicals"},
    {"symbol": "SMDR", "name": "Samudera Indonesia", "sector": "Transportation & Logistics"},
    # Note: INKA removed (delisted from Yahoo Finance)
]

# Other notable IDX stocks
OTHER_IDX_STOCKS = [
    {"symbol": "AALI", "name": "Astra Agro Lestari", "sector": "Consumer Non-Cyclicals"},
    {"symbol": "ADMF", "name": "Adira Dinamika Multi Finance", "sector": "Finance"},
    {"symbol": "AGII", "name": "Aneka Gas Industri", "sector": "Industrials"},
    {"symbol": "ANJT", "name": "Austindo Nusantara Jaya", "sector": "Consumer Non-Cyclicals"},
    {"symbol": "APLN", "name": "Agung Podomoro Land", "sector": "Property & Real Estate"},
    {"symbol": "ARTO", "name": "Bank Jago", "sector": "Finance"},
    {"symbol": "ASRI", "name": "Alam Sutera Realty", "sector": "Property & Real Estate"},
    {"symbol": "AUTO", "name": "Astra Otoparts", "sector": "Consumer Cyclicals"},
    {"symbol": "BDMN", "name": "Bank Danamon Indonesia", "sector": "Finance"},
    {"symbol": "BJBR", "name": "Bank Pembangunan Daerah Jawa Barat dan Banten", "sector": "Finance"},
    {"symbol": "BNGA", "name": "Bank CIMB Niaga", "sector": "Finance"},
    {"symbol": "BTPS", "name": "Bank BTPN Syariah", "sector": "Finance"},
    {"symbol": "DMAS", "name": "Puradelta Lestari", "sector": "Property & Real Estate"},
    {"symbol": "DOID", "name": "Delta Dunia Makmur", "sector": "Energy"},
    {"symbol": "GGRM", "name": "Gudang Garam", "sector": "Consumer Non-Cyclicals"},
    {"symbol": "GIAA", "name": "Garuda Indonesia", "sector": "Transportation & Logistics"},
    {"symbol": "HEXA", "name": "Hexindo Adiperkasa", "sector": "Industrials"},
    {"symbol": "INDY", "name": "Indika Energy", "sector": "Energy"},
    {"symbol": "KAEF", "name": "Kimia Farma", "sector": "Consumer Non-Cyclicals"},
    {"symbol": "LPKR", "name": "Lippo Karawaci", "sector": "Property & Real Estate"},
    {"symbol": "LPPF", "name": "Matahari Department Store", "sector": "Consumer Cyclicals"},
    {"symbol": "LSIP", "name": "PP London Sumatra Indonesia", "sector": "Consumer Non-Cyclicals"},
    {"symbol": "MAIN", "name": "Malindo Feedmill", "sector": "Consumer Non-Cyclicals"},
    {"symbol": "MEGA", "name": "Bank Mega", "sector": "Finance"},
    {"symbol": "MIKA", "name": "Mitra Keluarga Karyasehat", "sector": "Healthcare"},
    {"symbol": "MNCN", "name": "Media Nusantara Citra", "sector": "Consumer Cyclicals"},
    {"symbol": "NISP", "name": "Bank OCBC NISP", "sector": "Finance"},
    {"symbol": "PNBN", "name": "Bank Pan Indonesia", "sector": "Finance"},
    {"symbol": "PNLF", "name": "Panin Financial", "sector": "Finance"},
    {"symbol": "PWON", "name": "Pakuwon Jati", "sector": "Property & Real Estate"},
    {"symbol": "SCMA", "name": "Surya Citra Media", "sector": "Consumer Cyclicals"},
    {"symbol": "SIDO", "name": "Industri Jamu dan Farmasi Sido Muncul", "sector": "Consumer Non-Cyclicals"},
    {"symbol": "SIMP", "name": "Salim Ivomas Pratama", "sector": "Consumer Non-Cyclicals"},
    {"symbol": "SMDR", "name": "Samudera Indonesia", "sector": "Transportation & Logistics"},
    {"symbol": "SMRA", "name": "Summarecon Agung", "sector": "Property & Real Estate"},
    {"symbol": "SRIL", "name": "Sri Rejeki Isman", "sector": "Industrials"},
    {"symbol": "SSMS", "name": "Sawit Sumbermas Sarana", "sector": "Consumer Non-Cyclicals"},
    {"symbol": "TINS", "name": "Timah", "sector": "Basic Materials"},
    {"symbol": "TKIM", "name": "Pabrik Kertas Tjiwi Kimia", "sector": "Basic Materials"},
    {"symbol": "TPIA", "name": "Chandra Asri Petrochemical", "sector": "Basic Materials"},
    {"symbol": "UNVR", "name": "Unilever Indonesia", "sector": "Consumer Non-Cyclicals"},
    {"symbol": "WIKA", "name": "Wijaya Karya", "sector": "Infrastructures"},
    {"symbol": "WSKT", "name": "Waskita Karya", "sector": "Infrastructures"},
    {"symbol": "WTON", "name": "Wijaya Karya Beton", "sector": "Infrastructures"},
]

# Combine all stocks
ALL_IDX_STOCKS = IDX30_STOCKS + LQ45_ADDITIONAL + OTHER_IDX_STOCKS


class IDXStockDatabase:
    """Local database for IDX stock listings with search capabilities."""

    def __init__(self):
        """Initialize the stock database."""
        self._stocks = {s["symbol"]: s for s in ALL_IDX_STOCKS}
        self._name_index = self._build_name_index()

    def _build_name_index(self) -> dict[str, str]:
        """Build index of stock names to symbols."""
        index = {}
        for stock in ALL_IDX_STOCKS:
            # Add full name
            name_lower = stock["name"].lower()
            index[name_lower] = stock["symbol"]

            # Add individual words from name
            for word in name_lower.split():
                if len(word) > 2:
                    if word not in index:
                        index[word] = stock["symbol"]

        return index

    def get_stock(self, symbol: str) -> dict[str, Any] | None:
        """Get stock info by symbol.

        Args:
            symbol: Stock symbol (e.g., BBCA)

        Returns:
            Stock info dict or None
        """
        symbol = symbol.upper().replace(".JK", "")
        return self._stocks.get(symbol)

    def get_idx30_stocks(self) -> list[dict[str, Any]]:
        """Get all IDX30 stocks."""
        return IDX30_STOCKS.copy()

    def get_lq45_stocks(self) -> list[dict[str, Any]]:
        """Get all LQ45 stocks."""
        return (IDX30_STOCKS + LQ45_ADDITIONAL).copy()

    def get_jii70_stocks(self) -> list[dict[str, Any]]:
        """Get all JII70 stocks (Jakarta Islamic Index 70)."""
        return JII70_STOCKS.copy()

    def get_all_stocks(self) -> list[dict[str, Any]]:
        """Get all known IDX stocks."""
        return ALL_IDX_STOCKS.copy()

    def get_stocks_by_sector(self, sector: str) -> list[dict[str, Any]]:
        """Get stocks by sector.

        Args:
            sector: Sector name

        Returns:
            List of stocks in sector
        """
        return [s for s in ALL_IDX_STOCKS if s["sector"].lower() == sector.lower()]

    def search(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.3,
    ) -> list[dict[str, Any]]:
        """Search for stocks by symbol or name.

        Uses fuzzy matching to find stocks matching the query.

        Args:
            query: Search query (symbol or name)
            limit: Maximum results
            min_score: Minimum similarity score (0-1)

        Returns:
            List of matching stocks with scores
        """
        if not query:
            return []

        query = query.strip().upper()
        query_lower = query.lower()
        results = []

        # First: Exact symbol match
        if query in self._stocks:
            stock = self._stocks[query].copy()
            stock["score"] = 1.0
            stock["match_type"] = "exact_symbol"
            return [stock]

        # Second: Symbol prefix match
        for symbol, stock in self._stocks.items():
            if symbol.startswith(query):
                result = stock.copy()
                result["score"] = 0.9
                result["match_type"] = "symbol_prefix"
                results.append(result)

        # Third: Name contains query
        for stock in ALL_IDX_STOCKS:
            if query_lower in stock["name"].lower():
                if not any(r["symbol"] == stock["symbol"] for r in results):
                    result = stock.copy()
                    result["score"] = 0.7
                    result["match_type"] = "name_contains"
                    results.append(result)

        # Fourth: Fuzzy match on names
        for stock in ALL_IDX_STOCKS:
            if any(r["symbol"] == stock["symbol"] for r in results):
                continue

            # Check symbol similarity
            symbol_score = SequenceMatcher(None, query, stock["symbol"]).ratio()
            name_score = SequenceMatcher(None, query_lower, stock["name"].lower()).ratio()

            best_score = max(symbol_score, name_score)
            if best_score >= min_score:
                result = stock.copy()
                result["score"] = best_score
                result["match_type"] = "fuzzy"
                results.append(result)

        # Sort by score and limit
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]


# Singleton instance
_db: IDXStockDatabase | None = None


def get_stock_database() -> IDXStockDatabase:
    """Get singleton stock database instance."""
    global _db
    if _db is None:
        _db = IDXStockDatabase()
    return _db


def search_stocks(query: str, limit: int = 10) -> list[dict[str, Any]]:
    """Search for stocks by symbol or name."""
    return get_stock_database().search(query, limit)


def get_stock_info(symbol: str) -> dict[str, Any] | None:
    """Get stock info by symbol."""
    return get_stock_database().get_stock(symbol)


def get_idx30_list() -> list[str]:
    """Get list of IDX30 symbols."""
    return [s["symbol"] for s in IDX30_STOCKS]


def get_lq45_list() -> list[str]:
    """Get list of LQ45 symbols."""
    return [s["symbol"] for s in IDX30_STOCKS + LQ45_ADDITIONAL]


def get_jii70_list() -> list[str]:
    """Get list of JII70 symbols (Jakarta Islamic Index 70)."""
    return [s["symbol"] for s in JII70_STOCKS]
