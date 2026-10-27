"""Callable, leakage-aware adapters around the project's existing model specifications."""
from __future__ import annotations

import ast
import importlib
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

try:
    from .data_loader import DEFAULT_DATA_PATH, load_daily_data
    from .results import PriceForecastResult, VolatilityForecastResult
except ImportError:
    from data_loader import DEFAULT_DATA_PATH, load_daily_data
    from results import PriceForecastResult, VolatilityForecastResult

logger = logging.getLogger(__name__)
TICKER = "TCS.NS"
HORIZONS = (1, 5, 21, 30, 45)
PRICE_MODELS = ("arima", "xgboost", "lstm")
ALL_MODELS = PRICE_MODELS + ("garch",)


def validate_horizon(horizon: int) -> int:
    horizon = int(horizon)
    if horizon not in HORIZONS:
        raise ValueError(f"horizon must be one of {HORIZONS}")
    return horizon


def _date_text(value) -> str:
    return pd.Timestamp(value).date().isoformat()


def _metadata_csv_order(horizon: int) -> tuple[int, int, int]:
    metrics_path = RESULTS_DIR / "arima_v2" / "metrics.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        if {"horizon", "selected_order"}.issubset(metrics.columns):
            rows = metrics.loc[pd.to_numeric(metrics["horizon"], errors="coerce") == horizon]
            if not rows.empty:
                value = ast.literal_eval(str(rows.iloc[0]["selected_order"]))
                if len(value) == 3:
                    return tuple(int(part) for part in value)
    return (1, 1, 1)


def forecast_arima(data: pd.DataFrame, horizon: int) -> PriceForecastResult:
    """Fit the active ARIMA model using the latest saved order, without order search."""
    from statsmodels.tsa.arima.model import ARIMA

    horizon = validate_horizon(horizon)
    order = _metadata_csv_order(horizon)
    close = data["Close"].astype(float)
    fitted = ARIMA(close.to_numpy(), order=order,
                   enforce_stationarity=False,
                   enforce_invertibility=False).fit(method_kwargs={"maxiter": 100})
    predicted = float(np.asarray(fitted.forecast(steps=horizon)).reshape(-1)[-1])
    if not np.isfinite(predicted):
        raise ValueError("ARIMA returned a non-finite forecast.")
    origin = float(close.iloc[-1])
    return PriceForecastResult(
        ticker=TICKER, model="ARIMA", forecast_origin_date=_date_text(close.index[-1]),
        target_date=None, horizon_trading_days=horizon, origin_close=origin,
        predicted_return_pct=(predicted / origin - 1.0) * 100.0,
        predicted_close=predicted, model_version="models/arima_model.py (ARIMA v2)",
        metadata={"selected_order": list(order), "selection_source": "saved ARIMA v2 metrics" if (RESULTS_DIR / "arima_v2" / "metrics.csv").exists() else "fallback ARIMA(1,1,1)"},
    )


def _load_xgb_artifact(horizon: int, latest_date: str):
    """Load only a saved artifact trained against the same cached-data endpoint."""
    artifact_dir = RESULTS_DIR / "xgboost_return_walkforward_v4"
    run_path = artifact_dir / "run_metadata.json"
    model_path = artifact_dir / f"xgboost_{horizon}d.ubj"
    model_meta_path = artifact_dir / f"xgboost_{horizon}d.json"
    if not all(path.is_file() for path in (run_path, model_path, model_meta_path)):
        return None
    run_meta = json.loads(run_path.read_text(encoding="utf-8"))
    model_meta = json.loads(model_meta_path.read_text(encoding="utf-8"))
    if run_meta.get("ticker") != TICKER or run_meta.get("last_data_date") != latest_date:
        return None
    from xgboost import XGBRegressor

    fitted = XGBRegressor()
    fitted.load_model(str(model_path))
    return fitted, model_meta


def forecast_xgboost(data: pd.DataFrame, horizon: int) -> PriceForecastResult:
    """Prefer the saved XGBoost model; otherwise use the existing fixed-fit helper."""
    horizon = validate_horizon(horizon)
    model_module = importlib.import_module("xgboost_model_v2")
    features = model_module.build_features(data)
    columns = list(features.columns)
    latest_date = _date_text(data.index[-1])
    artifact_warning = None
    try:
        loaded = _load_xgb_artifact(horizon, latest_date)
    except Exception as exc:
        artifact_warning = f"Saved model load failed: {type(exc).__name__}: {exc}"
        logger.warning(artifact_warning)
        loaded = None
    if loaded:
        fitted, metadata = loaded
        model_columns = metadata.get("feature_columns", columns)
        if model_columns != columns:
            artifact_warning = "Saved model feature columns do not match the current feature builder."
            logger.warning(artifact_warning)
            loaded = None
        else:
            centre = float(metadata.get("centre_median_return", 0.0))
            beta = float(metadata.get("shrinkage_beta", 0.0))
            configuration = "saved walk-forward v4 artifact"
    if not loaded:
        table, valid = model_module.make_table(data, features, horizon)
        train = table.loc[valid & (table["target_date"] <= data.index[-1])].copy()
        params_path = RESULTS_DIR / "xgboost_return_walkforward_v4" / f"selected_params_{horizon}d.json"
        if params_path.is_file():
            params = json.loads(params_path.read_text(encoding="utf-8")).get("selected_params", {})
        else:
            params = dict(model_module.PARAM_CANDIDATES[0])
        if not params:
            params = dict(model_module.PARAM_CANDIDATES[0])
        fitted, centre, beta = model_module.fit_with_shrinkage(train, columns, horizon, params)
        configuration = "existing fixed-parameter fit; no parameter search"
    latest_features = features.iloc[[-1]][columns]
    if latest_features.isna().any(axis=None):
        raise ValueError("Latest XGBoost feature row contains missing values.")
    deviation = float(fitted.predict(latest_features)[0])
    predicted_return = float(beta * deviation)
    origin = float(data["Close"].iloc[-1])
    predicted = origin * (1.0 + predicted_return)
    if not np.isfinite(predicted):
        raise ValueError("XGBoost returned a non-finite forecast.")
    return PriceForecastResult(
        ticker=TICKER, model="XGBoost", forecast_origin_date=latest_date,
        target_date=None, horizon_trading_days=horizon, origin_close=origin,
        predicted_return_pct=predicted_return * 100.0, predicted_close=predicted,
        model_version="models/xgboost_model_v2.py (walk-forward v4 protocol)",
        metadata={"shrinkage_beta": float(beta), "centre_median_return": float(centre),
                  "artifact_or_fit": configuration, "artifact_warning": artifact_warning},
    )


def forecast_lstm(data: pd.DataFrame, horizon: int) -> PriceForecastResult:
    """Generate a current-origin LSTM forecast with the existing v2 training routine."""
    from sklearn.preprocessing import StandardScaler

    horizon = validate_horizon(horizon)
    model = importlib.import_module("lstm_model_v2")
    tf = model.load_tf()
    features = model.build_features(data)
    close = data["Close"].reindex(features.index)
    target = (close.shift(-horizon) / close - 1.0).to_numpy(dtype=float)
    positions = np.arange(model.LOOKBACK - 1, len(features))
    train_positions = positions[np.isfinite(target[positions])]
    if len(train_positions) < model.MIN_TRAIN:
        raise ValueError(f"LSTM needs at least {model.MIN_TRAIN} valid training rows; found {len(train_positions)}.")
    scaler = StandardScaler().fit(features.values)
    scaled = np.clip(scaler.transform(features.values), -model.CLIP, model.CLIP)
    X_train = model.windows(scaled, train_positions)
    y_train = target[train_positions]
    X_latest = model.windows(scaled, np.array([len(features) - 1]))
    raw, shrunk, beta, epochs = model.fit_predict(tf, X_train, y_train, X_latest, horizon, model.SEED)
    origin = float(close.iloc[-1])
    predicted_return = float(shrunk[0])
    predicted = origin * (1.0 + predicted_return)
    if not np.isfinite(predicted):
        raise ValueError("LSTM returned a non-finite forecast.")
    return PriceForecastResult(
        ticker=TICKER, model="LSTM", forecast_origin_date=_date_text(data.index[-1]),
        target_date=None, horizon_trading_days=horizon, origin_close=origin,
        predicted_return_pct=predicted_return * 100.0, predicted_close=predicted,
        model_version="models/lstm_model_v2.py (direct return v2)",
        metadata={"shrinkage_beta": float(beta), "seed_count": model.N_SEEDS,
                  "epochs_per_seed": [int(epoch) for epoch in epochs],
                  "raw_return_pct": float(raw[0]) * 100.0,
                  "training_rows": int(len(train_positions)),
                  "artifact_or_fit": "on-demand fit; model script does not save deployable weights"},
    )


def forecast_garch(data: pd.DataFrame, horizon: int) -> VolatilityForecastResult:
    """Forecast cumulative horizon volatility with the active GARCH v2 specification."""
    from arch import arch_model

    horizon = validate_horizon(horizon)
    close = data["Close"].astype(float)
    returns = (100.0 * np.log(close / close.shift(1))).dropna()
    fitted = arch_model(returns, mean="Constant", vol="GARCH", p=1, o=0, q=1,
                        dist="StudentsT", rescale=False).fit(disp="off", show_warning=False)
    variances = fitted.forecast(horizon=horizon, reindex=False).variance.iloc[-1].to_numpy(dtype=float)
    variance = float(np.sum(variances[:horizon]))
    volatility = float(np.sqrt(max(variance, 0.0)))
    if not np.isfinite(volatility):
        raise ValueError("GARCH returned a non-finite volatility forecast.")
    return VolatilityForecastResult(
        ticker=TICKER, model="GARCH(1,1)-Student-t", forecast_origin_date=_date_text(close.index[-1]),
        target_date=None, horizon_trading_days=horizon, forecast_variance=variance,
        forecast_volatility=volatility, units="cumulative return percentage points",
        model_version="models/garch_model_v2.py (active source)",
        metadata={"variance_units": "squared return percentage points", "distribution": "Student-t"},
    )


FORECASTERS: dict[str, Callable] = {
    "arima": forecast_arima, "xgboost": forecast_xgboost,
    "lstm": forecast_lstm, "garch": forecast_garch,
}


def run_forecasts(data: pd.DataFrame, horizon: int, models=PRICE_MODELS) -> dict[str, dict]:
    """Run selected models at the same latest data origin and isolate each failure."""
    horizon = validate_horizon(horizon)
    results = {}
    for name in models:
        key = str(name).lower()
        if key not in FORECASTERS:
            raise ValueError(f"Unknown model '{name}'. Choose from {tuple(FORECASTERS)}")
        started = time.perf_counter()
        try:
            result = FORECASTERS[key](data, horizon)
            result.duration_seconds = round(time.perf_counter() - started, 3)
            results[key] = result.to_dict()
        except Exception as exc:
            logger.exception("%s forecast failed at horizon %s", key, horizon)
            common = {"ticker": TICKER, "model": key.upper(),
                      "forecast_origin_date": _date_text(data.index[-1]),
                      "target_date": None, "horizon_trading_days": horizon,
                      "status": "error", "duration_seconds": round(time.perf_counter() - started, 3),
                      "error": f"{type(exc).__name__}: {exc}"}
            if key == "garch":
                results[key] = VolatilityForecastResult(
                    **common, forecast_variance=None, forecast_volatility=None,
                    units="cumulative return percentage points",
                    model_version="models/garch_model_v2.py (active source)").to_dict()
            else:
                versions = {
                    "arima": "models/arima_model.py (ARIMA v2)",
                    "xgboost": "models/xgboost_model_v2.py (walk-forward v4 protocol)",
                    "lstm": "models/lstm_model_v2.py (direct return v2)",
                }
                results[key] = PriceForecastResult(
                    **common, origin_close=float(data["Close"].iloc[-1]),
                    predicted_return_pct=None, predicted_close=None,
                    model_version=versions[key]).to_dict()
    return results


def forecast(model_name: str, horizon: int, data: pd.DataFrame | None = None,
             data_path: str | Path = DEFAULT_DATA_PATH) -> dict:
    key = model_name.strip().lower()
    if key not in FORECASTERS:
        raise ValueError(f"model must be one of {tuple(FORECASTERS)}")
    frame = data if data is not None else load_daily_data(data_path)
    return run_forecasts(frame, horizon, (key,))[key]


def forecast_all(horizon: int, data: pd.DataFrame | None = None,
                 data_path: str | Path = DEFAULT_DATA_PATH) -> dict[str, dict]:
    frame = data if data is not None else load_daily_data(data_path)
    return run_forecasts(frame, horizon, ALL_MODELS)
