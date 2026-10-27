"""Adapters that expose the existing model implementations behind one API.

The adapters use each model's existing architecture and fixed settings. The
XGBoost adapter uses its first declared parameter set and does not run its
historical parameter search. The ARIMA adapter uses the project's saved order
when available, otherwise ARIMA(1,1,1).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

try:  # support both ``src.forecasting`` and the project entry point import style
    from .data_loader import load_daily_data
    from .results import ForecastResult
except ImportError:
    from data_loader import load_daily_data
    from results import ForecastResult

HORIZONS = (1, 5, 21, 30, 45)


def _validate_horizon(horizon: int) -> int:
    horizon = int(horizon)
    if horizon not in HORIZONS:
        raise ValueError(f"horizon must be one of {HORIZONS}")
    return horizon


def _price_result(name: str, date, current: float, horizon: int,
                  forecast_close: float, metadata=None) -> ForecastResult:
    ret = (forecast_close / current - 1.0) * 100.0
    return ForecastResult(name, "price", pd.Timestamp(date).date().isoformat(), horizon,
                          float(current), float(forecast_close), "INR", float(forecast_close),
                          float(ret), metadata or {})


def forecast_arima(data: pd.DataFrame, horizon: int) -> ForecastResult:
    """Forecast close with the existing statsmodels ARIMA implementation."""
    from statsmodels.tsa.arima.model import ARIMA

    horizon = _validate_horizon(horizon)
    order = (1, 1, 1)
    selected = PROJECT_ROOT / "results" / "arima_v2" / "selected_orders.csv"
    if selected.exists():
        orders = pd.read_csv(selected)
        row = orders.loc[orders["horizon"] == horizon]
        if not row.empty:
            import ast
            parsed = ast.literal_eval(str(row.iloc[0]["selected_order"]))
            if len(parsed) == 3:
                order = tuple(map(int, parsed))
    close = data["Close"].astype(float)
    fitted = ARIMA(close.to_numpy(), order=order,
                   enforce_stationarity=False,
                   enforce_invertibility=False).fit(method_kwargs={"maxiter": 100})
    value = float(np.asarray(fitted.forecast(steps=horizon)).reshape(-1)[-1])
    return _price_result("ARIMA", close.index[-1], close.iloc[-1], horizon, value,
                         {"order": list(order)})


def forecast_xgboost(data: pd.DataFrame, horizon: int) -> ForecastResult:
    """Fit the existing XGBoost model once using its established default settings."""
    import xgboost_model_v2 as model

    horizon = _validate_horizon(horizon)
    features = model.build_features(data)
    table, valid = model.make_table(data, features, horizon)
    train = table.loc[valid & (table["target_date"] <= data.index[-1])].copy()
    cols = list(features.columns)
    params = dict(model.PARAM_CANDIDATES[0])
    fitted, centre, beta = model.fit_with_shrinkage(train, cols, horizon, params)
    latest = features.iloc[[-1]][cols]
    deviation = float(fitted.predict(latest)[0])
    current = float(data["Close"].iloc[-1])
    return _price_result("XGBoost", data.index[-1], current, horizon,
                         current * (1.0 + beta * deviation),
                         {"shrinkage_beta": float(beta), "raw_centre_return": float(centre),
                          "configuration": "existing default candidate; no parameter search"})


def forecast_lstm(data: pd.DataFrame, horizon: int) -> ForecastResult:
    """Fit one latest-origin forecast through the existing LSTM helpers."""
    import lstm_model_v2 as model
    from sklearn.preprocessing import StandardScaler

    horizon = _validate_horizon(horizon)
    tf = model.load_tf()
    features = model.build_features(data)
    close = data["Close"].reindex(features.index)
    target = (close.shift(-horizon) / close - 1.0).to_numpy()
    positions = np.arange(model.LOOKBACK - 1, len(features))
    valid = np.isfinite(target[positions])
    train_positions = positions[valid]
    if len(train_positions) < model.MIN_TRAIN:
        raise ValueError("Not enough training rows for the existing LSTM configuration.")
    scaler = StandardScaler().fit(features.values)
    scaled = np.clip(scaler.transform(features.values), -model.CLIP, model.CLIP)
    X_train = model.windows(scaled, train_positions)
    y_train = target[train_positions]
    latest_pos = np.array([len(features) - 1])
    X_latest = model.windows(scaled, latest_pos)
    _, shrunk, beta, _ = model.fit_predict(tf, X_train, y_train, X_latest, horizon, model.SEED)
    current = float(close.iloc[-1])
    return _price_result("LSTM", data.index[-1], current, horizon,
                         current * (1.0 + float(shrunk[0])),
                         {"shrinkage_beta": float(beta), "seeds": model.N_SEEDS,
                          "configuration": "existing LSTM architecture and training settings"})


def forecast_garch(data: pd.DataFrame, horizon: int) -> ForecastResult:
    """Forecast cumulative volatility from the existing GARCH(1,1)-Student-t spec."""
    from arch import arch_model

    horizon = _validate_horizon(horizon)
    close = data["Close"].astype(float)
    returns = (100.0 * np.log(close / close.shift(1))).dropna()
    fit = arch_model(returns, mean="Constant", vol="GARCH", p=1, o=0, q=1,
                     dist="StudentsT", rescale=False).fit(disp="off", show_warning=False)
    variance = float(fit.forecast(horizon=horizon, reindex=False).variance.iloc[-1]
                     .to_numpy(dtype=float)[:horizon].sum())
    return ForecastResult("GARCH", "volatility", close.index[-1].date().isoformat(), horizon,
                          float(close.iloc[-1]), float(np.sqrt(max(variance, 0.0))),
                          "cumulative return percentage points", metadata={"variance": variance,
                          "specification": "GARCH(1,1)-Student-t"})


FORECASTERS = {"arima": forecast_arima, "xgboost": forecast_xgboost,
               "lstm": forecast_lstm, "garch": forecast_garch}


def forecast(model_name: str, horizon: int, data_path: str | Path | None = None) -> dict:
    key = model_name.strip().lower()
    if key not in FORECASTERS:
        raise ValueError(f"model must be one of {tuple(FORECASTERS)}")
    data = load_daily_data(data_path or PROJECT_ROOT / "data" / "TCS.NS_daily.csv")
    return FORECASTERS[key](data, horizon).to_dict()


def forecast_all(horizon: int, data_path: str | Path | None = None) -> dict[str, dict]:
    """Run all four independently and retain per-model errors in the shared response."""
    horizon = _validate_horizon(horizon)
    data = load_daily_data(data_path or PROJECT_ROOT / "data" / "TCS.NS_daily.csv")
    results = {}
    for name, run in FORECASTERS.items():
        try:
            results[name] = run(data, horizon).to_dict()
        except Exception as exc:
            results[name] = {"model": name, "error": f"{type(exc).__name__}: {exc}"}
    return {"ticker": "TCS.NS", "horizon_trading_days": int(horizon), "results": results}
