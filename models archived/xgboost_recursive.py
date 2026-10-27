"""Recursive one-step XGBoost forecasting for TCS.NS with naive and momentum benchmarks.

This is a separate experiment from the direct-horizon XGBoost model. It recursively
predicts daily returns up to each horizon using only features computable from past
and predicted closing prices. It intentionally excludes future volume, open, high,
and low because these are unknown during a real recursive forecast.
"""
from pathlib import Path
import json, sys, warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=FutureWarning)
TICKER = "TCS.NS"
HORIZONS = [1, 5, 21, 30, 45]
VALIDATION_DAYS = 252
MIN_TRAIN_ROWS = 250
RETRAIN_EVERY = 5
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "xgboost_recursive_walkforward_v1"
OUT.mkdir(parents=True, exist_ok=True)


def download_data():
    print(f"[1/5] Downloading five years of daily data for {TICKER}...", flush=True)
    raw = yf.download(TICKER, period="5y", interval="1d", auto_adjust=True,
                      progress=False, threads=False)
    if raw.empty:
        raise RuntimeError("No data downloaded. Check internet connection and ticker.")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    needed = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in needed if c not in raw.columns]
    if missing:
        raise RuntimeError(f"Missing columns: {missing}")
    df = raw[needed].copy()
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) < 700:
        raise RuntimeError(f"Only {len(df)} observations downloaded.")
    print(f"      {len(df)} rows: {df.index.min().date()} to {df.index.max().date()}", flush=True)
    return df


def close_features_from_history(history, date):
    """Features at date using close history ending at the previous observation.

    history includes the close at `date` (the forecast origin close for that step).
    All features are derived from observed or recursively predicted closes only.
    """
    s = pd.Series(np.asarray(history, dtype=float))
    if len(s) < 51:
        raise ValueError("Need at least 51 closes to construct recursive features.")
    ret = s.pct_change()
    f = {}
    for lag in [1, 2, 3, 5, 10, 15, 21, 30]:
        f[f"return_lag_{lag}"] = float(ret.iloc[-lag])
    for w in [2, 5, 10, 21, 30]:
        f[f"return_{w}d"] = float(s.iloc[-1] / s.iloc[-1-w] - 1)
    for w in [5, 10, 21, 30, 50]:
        f[f"close_to_sma_{w}"] = float(s.iloc[-1] / s.iloc[-w:].mean() - 1)
    for w in [5, 10, 21, 30]:
        f[f"volatility_{w}d"] = float(ret.iloc[-w:].std(ddof=1))
    f["day_of_week"] = date.dayofweek
    f["month"] = date.month
    return f


def build_training_features(close):
    ret = close.pct_change()
    x = pd.DataFrame(index=close.index)
    for lag in [1, 2, 3, 5, 10, 15, 21, 30]:
        x[f"return_lag_{lag}"] = ret.shift(lag - 1)
    for w in [2, 5, 10, 21, 30]:
        x[f"return_{w}d"] = close.pct_change(w)
    for w in [5, 10, 21, 30, 50]:
        x[f"close_to_sma_{w}"] = close / close.rolling(w).mean() - 1
    for w in [5, 10, 21, 30]:
        x[f"volatility_{w}d"] = ret.rolling(w).std()
    x["day_of_week"] = close.index.dayofweek
    x["month"] = close.index.month
    return x.replace([np.inf, -np.inf], np.nan)


def make_model():
    return XGBRegressor(objective="reg:squarederror", n_estimators=250,
        max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=2.0, tree_method="hist", n_jobs=2, random_state=42, verbosity=0)


def recursive_forecast(model, close_history, future_dates, feature_cols):
    """Forecast each future close recursively without accessing future OHLCV."""
    hist = list(map(float, close_history))
    out = []
    for future_date in future_dates:
        xrow = pd.DataFrame([close_features_from_history(hist, future_date)], columns=feature_cols)
        pred_return = float(model.predict(xrow)[0])
        # Guard against numerically absurd one-day predictions destabilizing the path.
        # Clip is intentionally conservative and is documented in output metadata.
        pred_return = float(np.clip(pred_return, -0.20, 0.20))
        next_close = hist[-1] * (1.0 + pred_return)
        out.append((pred_return, next_close))
        hist.append(next_close)
    return out


def main():
    df = download_data()
    close = df["Close"].astype(float)
    features = build_training_features(close)
    feature_cols = list(features.columns)
    last_date = df.index[-1]
    validation_start = df.index[-VALIDATION_DAYS]
    target_next_date = pd.Series(df.index, index=df.index).shift(-1)
    one_day_target = (close.shift(-1) / close - 1).rename("target_return")
    training_table = features.copy()
    training_table["target_return"] = one_day_target
    training_table["target_date"] = target_next_date
    training_table = training_table.replace([np.inf, -np.inf], np.nan)
    eligible_origins = [d for d in df.index if d >= validation_start and d < last_date]
    eligible_origins = [d for d in eligible_origins if features.loc[[d], feature_cols].notna().all(axis=1).iloc[0]]
    print("[2/5] Recursive walk-forward setup...", flush=True)
    print(f"      Validation origins: {eligible_origins[0].date()} to {eligible_origins[-1].date()}", flush=True)
    print(f"      Refit every {RETRAIN_EVERY} origins; future features use predicted closes only.", flush=True)

    all_metrics = []
    all_summaries = []
    for h in HORIZONS:
        print(f"\n[3/5] Recursive horizon: {h} trading day(s)", flush=True)
        records = []
        model, last_fit_i, last_fit_origin, latest_train_target = None, None, None, None
        origins = []
        for origin in eligible_origins:
            pos = df.index.get_loc(origin)
            if pos + h >= len(df.index):
                continue
            target_date = df.index[pos + h]
            if target_date < validation_start or target_date > last_date:
                continue
            origins.append(origin)

        for i, origin in enumerate(origins):
            refit = model is None or (i - last_fit_i) >= RETRAIN_EVERY
            if refit:
                train_mask = (training_table["target_date"].notna()
                    & (training_table["target_date"] < origin)
                    & training_table["target_return"].notna()
                    & training_table[feature_cols].notna().all(axis=1))
                train = training_table.loc[train_mask]
                if len(train) < MIN_TRAIN_ROWS:
                    continue
                model = make_model()
                model.fit(train[feature_cols], train["target_return"], verbose=False)
                last_fit_i, last_fit_origin = i, origin
                latest_train_target = pd.Timestamp(train["target_date"].max())
                if not latest_train_target < origin:
                    raise RuntimeError(f"Leakage guard failed: training target {latest_train_target} >= origin {origin}")

            pos = df.index.get_loc(origin)
            future_dates = list(df.index[pos + 1:pos + h + 1])
            forecast_path = recursive_forecast(model, close.iloc[:pos + 1].tolist(), future_dates, feature_cols)
            _, pred_close = forecast_path[-1]
            origin_close = float(close.loc[origin])
            pred_return = pred_close / origin_close - 1
            actual_close = float(close.iloc[pos + h])
            actual_return = actual_close / origin_close - 1
            trailing_return = origin_close / float(close.iloc[pos - h]) - 1 if pos >= h else np.nan
            momentum_close = origin_close * (1 + trailing_return)
            naive_close = origin_close
            records.append({
                "forecast_origin_date": origin.date().isoformat(),
                "target_date": future_dates[-1].date().isoformat(),
                "horizon_trading_days": h,
                "origin_close": origin_close,
                "actual_close": actual_close,
                "actual_return_pct": actual_return * 100,
                "recursive_xgb_predicted_return_pct": pred_return * 100,
                "recursive_xgb_predicted_close": pred_close,
                "naive_predicted_return_pct": 0.0,
                "naive_predicted_close": naive_close,
                "momentum_trailing_return_pct": trailing_return * 100,
                "momentum_predicted_return_pct": trailing_return * 100,
                "momentum_predicted_close": momentum_close,
                "recursive_xgb_absolute_error": abs(actual_close - pred_close),
                "naive_absolute_error": abs(actual_close - naive_close),
                "momentum_absolute_error": abs(actual_close - momentum_close),
                "model_last_fit_origin": last_fit_origin.date().isoformat(),
                "training_latest_target_date": latest_train_target.date().isoformat(),
                "recursive_daily_return_clip_pct": 20.0,
            })
            if (i + 1) % 50 == 0:
                print(f"      Processed {i+1}/{len(origins)} origins...", flush=True)

        pred = pd.DataFrame(records)
        if pred.empty:
            raise RuntimeError(f"No recursive predictions generated for {h}D.")
        pred.to_csv(OUT / f"predictions_recursive_{h}d.csv", index=False)
        actual_p = pred["actual_close"].to_numpy()
        xgb_p = pred["recursive_xgb_predicted_close"].to_numpy()
        naive_p = pred["naive_predicted_close"].to_numpy()
        mom_p = pred["momentum_predicted_close"].to_numpy()
        actual_r = pred["actual_return_pct"].to_numpy()
        xgb_r = pred["recursive_xgb_predicted_return_pct"].to_numpy()
        mom_r = pred["momentum_predicted_return_pct"].to_numpy()
        mae = lambda a, b: float(mean_absolute_error(a, b))
        rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, b)))
        direction = lambda a, b: float((np.sign(a) == np.sign(b)).mean() * 100)
        xmae, nmae, mmae = mae(actual_p, xgb_p), mae(actual_p, naive_p), mae(actual_p, mom_p)
        xrmae = mae(actual_r, xgb_r)
        nrmae = mae(actual_r, np.zeros_like(actual_r))
        mrmae = mae(actual_r, mom_r)
        all_metrics.append({
            "ticker": TICKER, "forecast_method": "recursive_one_day_xgb",
            "horizon_trading_days": h, "validation_predictions": len(pred),
            "validation_target_start": pred["target_date"].min(), "validation_target_end": pred["target_date"].max(),
            "recursive_xgb_price_rmse": rmse(actual_p, xgb_p),
            "naive_price_rmse": rmse(actual_p, naive_p), "momentum_price_rmse": rmse(actual_p, mom_p),
            "recursive_xgb_price_mae": xmae, "naive_price_mae": nmae, "momentum_price_mae": mmae,
            "recursive_xgb_return_mae_percentage_points": xrmae,
            "naive_return_mae_percentage_points": nrmae, "momentum_return_mae_percentage_points": mrmae,
            "recursive_xgb_directional_accuracy_pct": direction(xgb_r, actual_r),
            "momentum_directional_accuracy_pct": direction(mom_r, actual_r),
            "recursive_xgb_price_mae_improvement_vs_naive_pct": (nmae-xmae)/nmae*100 if nmae else np.nan,
            "recursive_xgb_price_mae_improvement_vs_momentum_pct": (mmae-xmae)/mmae*100 if mmae else np.nan,
            "recursive_xgb_return_mae_improvement_vs_naive_pct": (nrmae-xrmae)/nrmae*100 if nrmae else np.nan,
            "recursive_xgb_return_mae_improvement_vs_momentum_pct": (mrmae-xrmae)/mrmae*100 if mrmae else np.nan,
        })
        sorted_pred = pred.sort_values("target_date")
        plt.figure(figsize=(11, 5))
        plt.plot(pd.to_datetime(sorted_pred["target_date"]), sorted_pred["actual_close"], label="Actual close")
        plt.plot(pd.to_datetime(sorted_pred["target_date"]), sorted_pred["recursive_xgb_predicted_close"], label="Recursive XGBoost")
        plt.plot(pd.to_datetime(sorted_pred["target_date"]), sorted_pred["naive_predicted_close"], label="Naive baseline", alpha=.8)
        plt.plot(pd.to_datetime(sorted_pred["target_date"]), sorted_pred["momentum_predicted_close"], label="Momentum benchmark", alpha=.8)
        plt.title(f"{TICKER}: Recursive {h}-Day Forecast vs Benchmarks")
        plt.xlabel("Target date"); plt.ylabel("Adjusted closing price (INR)")
        plt.legend(); plt.grid(True, alpha=.25); plt.tight_layout()
        plt.savefig(OUT / f"recursive_forecast_{h}d.png", dpi=160); plt.close()
        print(f"      Price MAE: recursive XGB ₹{xmae:.2f} | naive ₹{nmae:.2f} | momentum ₹{mmae:.2f}", flush=True)
        print(f"      Return MAE (percentage points): recursive XGB {xrmae:.3f} | naive {nrmae:.3f} | momentum {mrmae:.3f}", flush=True)

        # Save a final one-day model for inference, using only labels known by the latest date.
        final_mask = (training_table["target_date"].notna()
            & (training_table["target_date"] <= last_date)
            & training_table["target_return"].notna()
            & training_table[feature_cols].notna().all(axis=1))
        final_train = training_table.loc[final_mask]
        final_model = make_model()
        final_model.fit(final_train[feature_cols], final_train["target_return"], verbose=False)
        final_model.save_model(str(OUT / "xgboost_recursive_1d.ubj"))
        with open(OUT / "model_metadata.json", "w", encoding="utf-8") as f:
            json.dump({"ticker": TICKER, "forecast_method": "recursive one-day return XGBoost",
                "horizons_trading_days": HORIZONS, "feature_columns": feature_cols,
                "parameters": final_model.get_params(), "retrain_every_origins": RETRAIN_EVERY,
                "training_rule": "one-day target date strictly before forecast origin during validation",
                "recursive_features": "close-derived returns, moving-average distances, rolling return volatility, weekday, month",
                "excluded_future_inputs": ["future open", "future high", "future low", "future volume"],
                "daily_return_clip": [-0.20, 0.20],
                "warning": "Research estimate only; not a trading recommendation."}, f, indent=2, default=str)
        current = float(close.iloc[-1])
        future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=max(HORIZONS))
        # For a current illustrative path, business days are only placeholders, not an exchange calendar.
        latest_path = recursive_forecast(final_model, close.tolist(), list(future_dates), feature_cols)
        for h2 in HORIZONS if h == HORIZONS[-1] else []:
            _, p_close = latest_path[h2 - 1]
            p_ret = p_close / current - 1
            trailing = current / float(close.iloc[-1-h2]) - 1
            all_summaries.append({"ticker": TICKER, "as_of_date": last_date.date().isoformat(),
                "current_adjusted_close": current, "horizon_trading_days": h2,
                "recursive_xgb_predicted_return_pct": p_ret * 100,
                "recursive_xgb_predicted_close": p_close,
                "naive_predicted_close": current,
                "momentum_trailing_return_pct": trailing * 100,
                "momentum_predicted_close": current * (1 + trailing),
                "note": "Future business dates are placeholders; use an exchange calendar for production inference.",
                "warning": "Research estimate only; not a trading recommendation."})

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(OUT / "metrics.csv", index=False)
    pd.DataFrame(all_summaries).to_csv(OUT / "forecast_summary.csv", index=False)
    print("\n[5/5] COMPLETE", flush=True)
    print(f"Outputs saved to {OUT}", flush=True)
    print(metrics_df[["horizon_trading_days", "recursive_xgb_price_mae", "naive_price_mae", "momentum_price_mae",
        "recursive_xgb_return_mae_percentage_points", "naive_return_mae_percentage_points",
        "momentum_return_mae_percentage_points", "recursive_xgb_directional_accuracy_pct"]].to_string(index=False), flush=True)
    print("Review metrics before drawing conclusions; this is a research experiment, not a trading signal.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr, flush=True)
        raise
