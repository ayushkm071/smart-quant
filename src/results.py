"""Explicit result schemas for price and volatility forecasts."""
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class PriceForecastResult:
    ticker: str
    model: str
    forecast_origin_date: str
    target_date: str | None
    horizon_trading_days: int
    origin_close: float | None
    predicted_return_pct: float | None
    predicted_close: float | None
    status: str = "success"
    execution_timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    duration_seconds: float | None = None
    model_version: str | None = None
    target_date_note: str = "Exact future date requires an exchange trading calendar."
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VolatilityForecastResult:
    ticker: str
    model: str
    forecast_origin_date: str
    target_date: str | None
    horizon_trading_days: int
    forecast_variance: float | None
    forecast_volatility: float | None
    units: str = "cumulative return percentage points"
    status: str = "success"
    execution_timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    duration_seconds: float | None = None
    model_version: str | None = None
    target_date_note: str = "Exact future date requires an exchange trading calendar."
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
