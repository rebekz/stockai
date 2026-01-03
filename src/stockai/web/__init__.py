"""StockAI Web Dashboard.

FastAPI-based web interface for stock analysis.
"""

from stockai.web.app import create_app

__all__ = ["create_app"]
