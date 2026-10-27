"""Unit checks for data integrity, result schemas, and saved evaluation readers."""
from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.data_loader import DEFAULT_DATA_PATH, inspect_data_file, load_daily_data
from src.dashboard_data import load_price_evaluations, load_volatility_evaluations
from src.forecasting import HORIZONS, validate_horizon, run_forecasts
from src.results import PriceForecastResult, VolatilityForecastResult

ROOT = Path(__file__).resolve().parents[1]


class DataIntegrityTests(unittest.TestCase):
    def test_project_csv_loads_sorted_positive_adjusted_ohlcv(self):
        frame = load_daily_data(DEFAULT_DATA_PATH)
        self.assertGreater(len(frame), 1000)
        self.assertEqual(list(frame.columns), ["Open", "High", "Low", "Close", "Volume"])
        self.assertTrue(frame.index.is_monotonic_increasing)
        self.assertTrue((frame["Close"] > 0).all())

    def test_incomplete_csv_reports_missing_columns(self):
        path = Path(__file__).resolve().parent / ".test_bad_data.csv"
        try:
            pd.DataFrame({"Date": ["2026-01-01"], "Close": [100]}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "missing required OHLCV"):
                load_daily_data(path)
        finally:
            path.unlink(missing_ok=True)

    def test_data_quality_report_counts_rows_and_dates(self):
        report = inspect_data_file(DEFAULT_DATA_PATH)
        self.assertTrue(report["exists"])
        self.assertGreater(report["rows_valid"], 1000)
        self.assertEqual(report["duplicate_dates"], 0)
        self.assertTrue(report["latest_date"])


class ResultSchemaTests(unittest.TestCase):
    def test_price_result_has_required_fields_and_nullable_target_date(self):
        result = PriceForecastResult("TCS.NS", "ARIMA", "2026-10-26", None, 5,
                                     3000.0, 1.25, 3037.5)
        payload = json.loads(json.dumps(result.to_dict()))
        for key in ("ticker", "model", "forecast_origin_date", "target_date",
                    "horizon_trading_days", "origin_close", "predicted_return_pct",
                    "predicted_close", "status", "execution_timestamp"):
            self.assertIn(key, payload)
        self.assertIsNone(payload["target_date"])

    def test_volatility_schema_keeps_distinct_units(self):
        result = VolatilityForecastResult("TCS.NS", "GARCH(1,1)-Student-t",
            "2026-10-26", None, 5, 2.5, 1.581, units="cumulative return percentage points")
        self.assertEqual(result.to_dict()["units"], "cumulative return percentage points")
        self.assertNotIn("predicted_close", result.to_dict())


class ForecastRunTests(unittest.TestCase):
    def test_horizon_whitelist(self):
        for horizon in HORIZONS:
            self.assertEqual(validate_horizon(horizon), horizon)
        with self.assertRaises(ValueError):
            validate_horizon(2)

    def test_model_failure_is_returned_without_hiding_error(self):
        data = load_daily_data(DEFAULT_DATA_PATH)
        def fail(_data, _horizon):
            raise RuntimeError("test failure")
        with patch("src.forecasting.FORECASTERS", {"arima": fail}):
            result = run_forecasts(data, 1, ("arima",))["arima"]
        self.assertEqual(result["status"], "error")
        self.assertIn("test failure", result["error"])
        self.assertIsNone(result["predicted_close"])


class EvaluationReaderTests(unittest.TestCase):
    def test_price_evaluations_use_saved_metrics_and_predictions(self):
        metrics, predictions = load_price_evaluations()
        self.assertFalse(metrics.empty)
        self.assertFalse(predictions.empty)
        self.assertTrue({"ARIMA", "XGBoost", "LSTM"}.issubset(set(metrics["model"])))
        self.assertTrue({"actual_close", "predicted_close", "target_date", "error"}.issubset(predictions.columns))

    def test_volatility_evaluations_keep_garch_and_historical_benchmark(self):
        metrics, predictions = load_volatility_evaluations()
        self.assertFalse(metrics.empty)
        self.assertFalse(predictions.empty)
        self.assertTrue(metrics["unit"].str.contains("percentage points").all())
        self.assertIn("GARCH(1,1)-Student-t", set(metrics["model"]))


class DashboardSourceTests(unittest.TestCase):
    def test_streamlit_entry_and_all_five_pages_are_present(self):
        app_source = (ROOT / "front end" / "dashboard_app.py").read_text(encoding="utf-8-sig")
        ast.parse(app_source)
        for page in ("page_overview", "page_forecasts", "page_performance", "page_volatility", "page_diagnostics"):
            self.assertIn(f"def {page}", app_source)
        entry = (ROOT / "main.py").read_text(encoding="utf-8-sig")
        ast.parse(entry)
        self.assertIn("run_dashboard()", entry)


if __name__ == "__main__":
    unittest.main()
