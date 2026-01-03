"""Pydantic Schemas for StockAI API.

Request validation and response serialization models.
"""

from datetime import datetime
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ============ Stock Schemas ============


class StockInfo(BaseModel):
    """Stock information nested in responses."""

    id: int
    symbol: str
    name: str
    sector: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


# ============ Watchlist Schemas ============


class WatchlistItemCreate(BaseModel):
    """Schema for creating a new watchlist item."""

    symbol: Optional[str] = Field(
        None,
        description="Stock symbol to add to watchlist",
        min_length=1,
        max_length=10,
    )
    stock_id: Optional[int] = Field(
        None,
        description="Stock ID to add to watchlist (alternative to symbol)",
        gt=0,
    )
    alert_price_above: Optional[float] = Field(
        None,
        description="Trigger alert when price goes above this value",
        gt=0,
    )
    alert_price_below: Optional[float] = Field(
        None,
        description="Trigger alert when price goes below this value",
        gt=0,
    )
    notes: Optional[str] = Field(
        None,
        description="User notes for this watchlist item",
        max_length=1000,
    )

    @model_validator(mode="after")
    def validate_stock_identifier(self) -> "WatchlistItemCreate":
        """Ensure either symbol or stock_id is provided, but not both."""
        if self.symbol is None and self.stock_id is None:
            raise ValueError("Either 'symbol' or 'stock_id' must be provided")
        if self.symbol is not None and self.stock_id is not None:
            raise ValueError("Provide either 'symbol' or 'stock_id', not both")
        return self

    @field_validator("symbol", mode="before")
    @classmethod
    def uppercase_symbol(cls, v: Optional[str]) -> Optional[str]:
        """Convert symbol to uppercase."""
        if v is not None:
            return v.upper().strip()
        return v

    @model_validator(mode="after")
    def validate_alert_prices(self) -> "WatchlistItemCreate":
        """Validate alert price logic if both are set."""
        if (
            self.alert_price_above is not None
            and self.alert_price_below is not None
            and self.alert_price_below >= self.alert_price_above
        ):
            raise ValueError(
                "'alert_price_below' must be less than 'alert_price_above'"
            )
        return self

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "symbol": "BBCA",
                    "alert_price_above": 10000.0,
                    "alert_price_below": 8500.0,
                    "notes": "Bank stock to watch",
                }
            ]
        }
    )


class WatchlistItemUpdate(BaseModel):
    """Schema for updating a watchlist item (partial updates supported)."""

    alert_price_above: Optional[float] = Field(
        None,
        description="Trigger alert when price goes above this value (set to 0 to clear)",
        ge=0,
    )
    alert_price_below: Optional[float] = Field(
        None,
        description="Trigger alert when price goes below this value (set to 0 to clear)",
        ge=0,
    )
    notes: Optional[str] = Field(
        None,
        description="User notes for this watchlist item (set to empty string to clear)",
        max_length=1000,
    )

    @model_validator(mode="after")
    def validate_alert_prices(self) -> "WatchlistItemUpdate":
        """Validate alert price logic if both are set and non-zero."""
        if (
            self.alert_price_above is not None
            and self.alert_price_below is not None
            and self.alert_price_above > 0
            and self.alert_price_below > 0
            and self.alert_price_below >= self.alert_price_above
        ):
            raise ValueError(
                "'alert_price_below' must be less than 'alert_price_above'"
            )
        return self

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "alert_price_above": 11000.0,
                    "notes": "Updated target price",
                }
            ]
        }
    )


class WatchlistItemResponse(BaseModel):
    """Schema for watchlist item response."""

    id: int = Field(..., description="Watchlist item ID")
    stock: StockInfo = Field(..., description="Associated stock information")
    alert_price_above: Optional[float] = Field(
        None, description="Alert trigger for price above"
    )
    alert_price_below: Optional[float] = Field(
        None, description="Alert trigger for price below"
    )
    notes: Optional[str] = Field(None, description="User notes")
    created_at: datetime = Field(..., description="When the item was added to watchlist")

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "examples": [
                {
                    "id": 1,
                    "stock": {
                        "id": 1,
                        "symbol": "BBCA",
                        "name": "Bank Central Asia Tbk",
                        "sector": "Financials",
                    },
                    "alert_price_above": 10000.0,
                    "alert_price_below": 8500.0,
                    "notes": "Bank stock to watch",
                    "created_at": "2024-01-15T10:30:00Z",
                }
            ]
        },
    )

    @field_validator("alert_price_above", "alert_price_below", mode="before")
    @classmethod
    def convert_decimal_to_float(cls, v):
        """Convert Decimal to float for JSON serialization."""
        if isinstance(v, Decimal):
            return float(v)
        return v


class WatchlistItemListResponse(BaseModel):
    """Schema for listing watchlist items."""

    count: int = Field(..., description="Total number of watchlist items")
    items: list[WatchlistItemResponse] = Field(
        ..., description="List of watchlist items"
    )


class WatchlistDeleteResponse(BaseModel):
    """Schema for watchlist item deletion response."""

    message: str = Field(..., description="Success message")
    deleted_item: WatchlistItemResponse = Field(
        ..., description="The deleted watchlist item"
    )
