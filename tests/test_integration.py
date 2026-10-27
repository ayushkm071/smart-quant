"""Fast checks for shared data, response shape, and API routing."""
from __future__ import annotations

import json
import io
import unittest
from pathlib import Path

import pandas as pd

from src.data_loader import DEFAULT_DATA_PATH, load_daily_data
from src.forecasting import forecast, forecast_all
from src.results import ForecastResult
import main as app


class InProcessHandler(app.Handler):
    """Run GET handlers directly without opening a sandbox-blocked socket."""
    def __init__(self, path):
        self.path = path
        self.wfile = io.BytesIO()
        self.headers = []
        self.status = None
        self.payload = None

    def send_response(self, status):
        self.status = status

    def send_header(self, key, value):
        self.headers.append((key, value))

    def end_headers(self):
        pass

    def _json(self, status, payload):
        self.status = status
        self.payload = payload


class SharedDataLoaderTests(unittest.TestCase):
    def test_project_csv_loads_with_ordered_ohlcv_schema(self):
        frame = load_daily_data(DEFAULT_DATA_PATH)
        self.assertGreater(len(frame), 1000)
        self.assertEqual(list(frame.columns), ["Open", "High", "Low", "Close", "Volume"])
        self.assertTrue(frame.index.is_monotonic_increasing)
        self.assertTrue((frame["Close"] > 0).all())

    def test_missing_required_columns_raise_clear_error(self):
        temporary = Path(__file__).resolve().parent / ".test_tmp"
        temporary.mkdir(exist_ok=True)
        path = temporary / "bad.csv"
        try:
            pd.DataFrame({"Date": ["2024-01-01"], "Close": [100]}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "missing required OHLCV"):
                load_daily_data(path)
        finally:
            path.unlink(missing_ok=True)
            temporary.rmdir()


class UnifiedResultTests(unittest.TestCase):
    def test_result_serializes_both_price_and_volatility_fields(self):
        result = ForecastResult("ARIMA", "price", "2024-01-01", 5, 100.0,
                                103.0, "INR", 103.0, 3.0, {"order": [1, 1, 1]})
        encoded = json.loads(json.dumps(result.to_dict()))
        self.assertEqual(encoded["forecast_type"], "price")
        self.assertEqual(encoded["forecast_close"], 103.0)
        self.assertEqual(encoded["horizon_trading_days"], 5)


class ForecastValidationTests(unittest.TestCase):
    def test_unsupported_model_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "model must be one of"):
            forecast("unknown", 1)

    def test_all_models_reject_unsupported_horizon(self):
        with self.assertRaisesRegex(ValueError, "horizon must be one of"):
            forecast_all(2)


class HttpRouteTests(unittest.TestCase):
    def test_models_route_lists_expected_models_and_horizons(self):
        handler = InProcessHandler("/api/models")
        handler.do_GET()
        payload = handler.payload
        self.assertEqual(handler.status, 200)
        self.assertEqual(payload["models"], ["arima", "xgboost", "lstm", "garch"])
        self.assertEqual(payload["horizons"], [1, 5, 21, 30, 45])

    def test_invalid_forecast_request_returns_bad_request(self):
        handler = InProcessHandler("/api/forecast?model=unknown&horizon=1")
        handler.do_GET()
        self.assertEqual(handler.status, 400)
        self.assertIn("model must be one of", handler.payload["error"])

    def test_dashboard_route_serves_html(self):
        handler = InProcessHandler("/")
        handler.do_GET()
        self.assertEqual(handler.status, 200)
        self.assertIn(("Content-Type", "text/html; charset=utf-8"), handler.headers)
        self.assertIn(b"TCS Forecast Dashboard", handler.wfile.getvalue())


if __name__ == "__main__":
    unittest.main()
