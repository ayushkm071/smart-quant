"""Readers and schema normalization for saved model evaluations."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
EVALUATION_DIRS = {
    "ARIMA": RESULTS / "arima_v2",
    "XGBoost": RESULTS / "xgboost_return_walkforward_v4",
    "LSTM": RESULTS / "lstm_direct_walkforward_v2",
    # Latest completed volatility evaluation; its v4 source remains archived.
    "GARCH": RESULTS / "garch_v4",
}


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def _date_minmax(frame: pd.DataFrame, column: str) -> tuple[str | None, str | None]:
    if column not in frame:
        return None, None
    dates = pd.to_datetime(frame[column], errors="coerce").dropna()
    if dates.empty:
        return None, None
    return dates.min().date().isoformat(), dates.max().date().isoformat()


def load_price_evaluations() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return normalized price-model metrics and historical predictions."""
    metric_rows: list[dict] = []
    prediction_parts: list[pd.DataFrame] = []

    arima_dir = EVALUATION_DIRS["ARIMA"]
    arima_metrics = _read_csv(arima_dir / "metrics.csv")
    arima_predictions = _read_csv(arima_dir / "predictions.csv")
    for _, row in arima_metrics.iterrows():
        h = int(row.get("horizon", 0))
        subset = arima_predictions.loc[pd.to_numeric(arima_predictions.get("horizon", pd.Series(dtype=float)), errors="coerce") == h]
        start, end = _date_minmax(subset, "target_date")
        metric_rows.append({"model": "ARIMA", "horizon": h, "unit": "INR",
                            "mae": row.get("ARIMA_MAE"), "rmse": row.get("ARIMA_RMSE"),
                            "directional_accuracy_pct": row.get("ARIMA_directional_accuracy_pct"),
                            "return_correlation": np.nan,
                            "valid_predictions": row.get("n_test_targets_valid"),
                            "failed_forecasts": row.get("n_failed_arima_forecasts"),
                            "evaluation_start": start, "evaluation_end": end,
                            "naive_mae": row.get("Naive_MAE"), "momentum_mae": row.get("Momentum_MAE"),
                            "source": "results/arima_v2/metrics.csv"})
    if not arima_predictions.empty:
        part = pd.DataFrame({
            "model": "ARIMA", "horizon": arima_predictions["horizon"],
            "origin_date": arima_predictions["origin_date"], "target_date": arima_predictions["target_date"],
            "origin_close": arima_predictions["origin_price"], "actual_close": arima_predictions["actual_price"],
            "predicted_close": arima_predictions["arima_price"], "naive_close": arima_predictions["naive_price"],
            "momentum_close": arima_predictions["momentum_price"],
        })
        part["actual_return_pct"] = (part["actual_close"] / part["origin_close"] - 1) * 100
        part["predicted_return_pct"] = (part["predicted_close"] / part["origin_close"] - 1) * 100
        part["error"] = part["predicted_close"] - part["actual_close"]
        prediction_parts.append(part)

    xgb_dir = EVALUATION_DIRS["XGBoost"]
    xgb_metrics = _read_csv(xgb_dir / "metrics.csv")
    xgb_predictions = _read_csv(xgb_dir / "all_test_predictions.csv")
    for _, row in xgb_metrics.iterrows():
        h = int(row.get("horizon_trading_days", 0))
        subset = xgb_predictions.loc[pd.to_numeric(xgb_predictions.get("horizon_trading_days", pd.Series(dtype=float)), errors="coerce") == h]
        start, end = _date_minmax(subset, "target_date")
        metric_rows.append({"model": "XGBoost", "horizon": h, "unit": "INR",
                            "mae": row.get("xgb_price_mae"), "rmse": row.get("xgb_price_rmse"),
                            "directional_accuracy_pct": row.get("xgb_directional_accuracy_pct"),
                            "return_correlation": row.get("xgb_deviation_corr"),
                            "valid_predictions": row.get("test_predictions"), "failed_forecasts": 0,
                            "evaluation_start": start, "evaluation_end": end,
                            "naive_mae": row.get("naive_price_mae"), "momentum_mae": row.get("momentum_price_mae"),
                            "source": "results/xgboost_return_walkforward_v4/metrics.csv"})
    if not xgb_predictions.empty:
        part = pd.DataFrame({
            "model": "XGBoost", "horizon": xgb_predictions["horizon_trading_days"],
            "origin_date": xgb_predictions["forecast_origin_date"], "target_date": xgb_predictions["target_date"],
            "origin_close": xgb_predictions["origin_close"], "actual_close": xgb_predictions["actual_close"],
            "predicted_close": xgb_predictions["xgb_predicted_close"], "naive_close": xgb_predictions["naive_predicted_close"],
            "momentum_close": xgb_predictions["momentum_predicted_close"],
            "actual_return_pct": xgb_predictions["actual_return_pct"],
            "predicted_return_pct": xgb_predictions["xgb_predicted_return_pct"],
        })
        part["error"] = part["predicted_close"] - part["actual_close"]
        prediction_parts.append(part)

    lstm_dir = EVALUATION_DIRS["LSTM"]
    lstm_metrics = _read_csv(lstm_dir / "metrics.csv")
    lstm_files = [lstm_dir / f"predictions_{h}d.csv" for h in (1, 5, 21, 30, 45)]
    lstm_predictions = pd.concat([_read_csv(path) for path in lstm_files], ignore_index=True)
    for _, row in lstm_metrics.iterrows():
        h = int(row.get("horizon_trading_days", 0))
        subset = lstm_predictions.loc[pd.to_numeric(lstm_predictions.get("horizon_trading_days", pd.Series(dtype=float)), errors="coerce") == h]
        start, end = _date_minmax(subset, "target_date")
        metric_rows.append({"model": "LSTM", "horizon": h, "unit": "INR",
                            "mae": row.get("lstm_price_mae"), "rmse": row.get("lstm_price_rmse"),
                            "directional_accuracy_pct": row.get("lstm_directional_accuracy_pct"),
                            "return_correlation": row.get("lstm_raw_return_corr"),
                            "valid_predictions": row.get("validation_predictions"), "failed_forecasts": 0,
                            "evaluation_start": start, "evaluation_end": end,
                            "naive_mae": row.get("naive_price_mae"), "momentum_mae": row.get("momentum_price_mae"),
                            "source": "results/lstm_direct_walkforward_v2/metrics.csv"})
    if not lstm_predictions.empty:
        part = pd.DataFrame({
            "model": "LSTM", "horizon": lstm_predictions["horizon_trading_days"],
            "origin_date": lstm_predictions["forecast_origin_date"], "target_date": lstm_predictions["target_date"],
            "origin_close": lstm_predictions["origin_close"], "actual_close": lstm_predictions["actual_close"],
            "predicted_close": lstm_predictions["lstm_predicted_close"], "naive_close": lstm_predictions["naive_predicted_close"],
            "momentum_close": lstm_predictions["momentum_predicted_close"],
            "actual_return_pct": lstm_predictions["actual_return_pct"],
            "predicted_return_pct": lstm_predictions["lstm_return_pct"],
        })
        part["error"] = part["predicted_close"] - part["actual_close"]
        prediction_parts.append(part)

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_parts, ignore_index=True) if prediction_parts else pd.DataFrame()
    if not metrics.empty:
        for column in ("horizon", "mae", "rmse", "directional_accuracy_pct", "return_correlation",
                       "valid_predictions", "failed_forecasts", "naive_mae", "momentum_mae"):
            metrics[column] = pd.to_numeric(metrics[column], errors="coerce")
    if not predictions.empty:
        predictions["target_date"] = pd.to_datetime(predictions["target_date"], errors="coerce")
        predictions["origin_date"] = pd.to_datetime(predictions["origin_date"], errors="coerce")
        for column in ("horizon", "origin_close", "actual_close", "predicted_close", "actual_return_pct",
                       "predicted_return_pct", "error", "naive_close", "momentum_close"):
            if column in predictions:
                predictions[column] = pd.to_numeric(predictions[column], errors="coerce")
    return metrics, predictions


def load_volatility_evaluations() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the latest saved GARCH v4 comparison; fall back to active v2 outputs."""
    v4 = EVALUATION_DIRS["GARCH"]
    metric_path = v4 / "metrics.csv"
    prediction_path = v4 / "predictions.csv"
    metrics = _read_csv(metric_path)
    predictions = _read_csv(prediction_path)
    source = "results/garch_v4 (archived GARCH v4 evaluation)"
    using_v2 = metrics.empty or predictions.empty
    if using_v2:
        v2 = RESULTS / "garch_v2"
        metrics = _read_csv(v2 / "metrics.csv")
        predictions = _read_csv(v2 / "predictions.csv")
        source = "results/garch_v2 (active GARCH v2 evaluation)"
        if not metrics.empty:
            metrics = metrics.rename(columns={"volatility_MAE_pct_points": "MAE",
                                              "volatility_RMSE_pct_points": "RMSE",
                                              "mean_QLIKE": "mean_QLIKE"})
    if metrics.empty:
        return pd.DataFrame(), pd.DataFrame()
    if "MAE" not in metrics and "volatility_MAE_pct_points" in metrics:
        metrics = metrics.rename(columns={"volatility_MAE_pct_points": "MAE",
                                          "volatility_RMSE_pct_points": "RMSE"})
    if "model" not in metrics.columns:
        return pd.DataFrame(), pd.DataFrame()
    normalized_metrics = metrics.copy()
    normalized_metrics["source"] = source
    normalized_metrics["unit"] = "cumulative return percentage points"
    normalized_metrics["model"] = normalized_metrics["model"].replace({
        "GARCH(1,1)-Student-t": "GARCH(1,1)-Student-t",
        "Rolling historical volatility": "Rolling historical volatility",
    })
    if using_v2 and not predictions.empty:
        base = ["horizon", "origin_date", "target_date", "realized_variance", "realized_volatility"]
        if all(column in predictions for column in base):
            garch = predictions[base + ["garch_forecast_variance", "garch_forecast_volatility", "garch_qlike"]].copy()
            garch = garch.rename(columns={"garch_forecast_variance": "forecast_variance", "garch_forecast_volatility": "forecast_volatility", "garch_qlike": "QLIKE"})
            garch["model"] = "GARCH(1,1)-Student-t"
            historical = predictions[base + ["historical_forecast_variance", "historical_forecast_volatility", "historical_qlike"]].copy()
            historical = historical.rename(columns={"historical_forecast_variance": "forecast_variance", "historical_forecast_volatility": "forecast_volatility", "historical_qlike": "QLIKE"})
            historical["model"] = "Rolling historical volatility"
            predictions = pd.concat([garch, historical], ignore_index=True)
    if "target_date" in predictions:
        predictions["target_date"] = pd.to_datetime(predictions["target_date"], errors="coerce")
    for col in ("horizon", "forecast_variance", "forecast_volatility", "realized_variance",
                "realized_volatility", "historical_forecast_volatility"):
        if col in predictions:
            predictions[col] = pd.to_numeric(predictions[col], errors="coerce")
    return normalized_metrics, predictions


def evaluation_file_inventory() -> pd.DataFrame:
    """List the exact saved outputs used by pages and their availability."""
    files = [
        ("ARIMA", "metrics.csv"), ("ARIMA", "predictions.csv"),
        ("XGBoost", "metrics.csv"), ("XGBoost", "all_test_predictions.csv"),
        ("XGBoost", "xgboost_1d.ubj"), ("XGBoost", "xgboost_5d.ubj"),
        ("XGBoost", "xgboost_21d.ubj"), ("XGBoost", "xgboost_30d.ubj"),
        ("XGBoost", "xgboost_45d.ubj"),
        ("LSTM", "metrics.csv"), ("LSTM", "predictions_1d.csv"),
        ("LSTM", "lstm_1d_metadata.json"),
        ("GARCH v4", "metrics.csv"), ("GARCH v4", "predictions.csv"),
    ]
    rows = []
    for model, name in files:
        directory = EVALUATION_DIRS[model] if model in EVALUATION_DIRS else RESULTS / "garch_v4"
        path = directory / name
        rows.append({"Model": model, "File": str(path.relative_to(PROJECT_ROOT)),
                     "Available": path.is_file(), "Size (KB)": round(path.stat().st_size / 1024, 1) if path.is_file() else None})
    return pd.DataFrame(rows)
