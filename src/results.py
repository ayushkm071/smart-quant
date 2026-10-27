"""Stable response shape shared by all forecasting adapters."""
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ForecastResult:
    model: str
    forecast_type: str
    as_of_date: str
    horizon_trading_days: int
    current_close: float
    forecast_value: float
    unit: str
    forecast_close: float | None = None
    forecast_return_pct: float | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
