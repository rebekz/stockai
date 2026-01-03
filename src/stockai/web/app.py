"""FastAPI Application for StockAI Web Dashboard.

Provides web interface for stock analysis, portfolio management,
and sentiment tracking.
"""

import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from stockai import __version__
from stockai.config import get_settings

logger = logging.getLogger(__name__)

# Get templates directory
TEMPLATES_DIR = Path(__file__).parent / "templates"


def create_app() -> FastAPI:
    """Create and configure FastAPI application.

    Returns:
        Configured FastAPI instance
    """
    app = FastAPI(
        title="StockAI Dashboard",
        description="AI-Powered Indonesian Stock Analysis Dashboard",
        version=__version__,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
    )

    # Setup templates
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    # Store templates in app state
    app.state.templates = templates

    # Include routers
    from stockai.web.routes import api_router, pages_router

    app.include_router(api_router, prefix="/api")
    app.include_router(pages_router)

    # Exception handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        if request.headers.get("accept") == "application/json":
            return JSONResponse(
                status_code=exc.status_code,
                content={"error": exc.detail},
            )
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "error": exc.detail, "status_code": exc.status_code},
            status_code=exc.status_code,
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"},
        )

    # Health check endpoint
    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "version": __version__}

    logger.info("StockAI Web Dashboard initialized")
    return app


# Create app instance
app = create_app()
