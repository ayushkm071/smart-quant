"""
Smart-Quant: XGBoost return forecasting + naive baseline + walk-forward evaluation.

Forecast horizons: 1, 5, 21, and 30 trading sessions.
Ticker: TCS.NS
Target: future close-to-close return from the forecast origin.
Baseline: predict zero return, equivalent to predicting the origin close.

Evaluation design:
- Each forecast origin is evaluated only if its target date is in the final 252 trading sessions.
- At each origin, fit only on examples whose target dates are strictly earlier than that origin.
- Training window is expanding; all features are computed using information available at each origin.
- For speed, retraining occurs at each origin. This is more realistic but can take longer than the fixed-fit prototype.
- Overlapping horizons mean individual forecast errors are not statistically independent.

Outputs in results/xgboost_return_walkforward/:
- metrics.csv
- predictions_{h}d.csv
- forecast_summary.csv (latest-origin estimates, if enough history)
- forecast_{h}d.png
- xgboost_{h}d.ubj (final model trained on all eligible history)
- feature_importance_{h}d.csv
"""

from pathlib import Path
import json
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=FutureWarning)

TICKER = "TCS.NS"
PERIOD = "5y"
HORIZONS = [1, 5, 21, 30]
VALIDATION_DAYS = 252
MIN_TRAIN_ROWS = 250
# Retrain every N forecast origins to reduce runtime. Set to 1 for every-origin retraining.
RETRAIN_EVERY = 5
RANDOM_STATE = 42

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "xgboost_return_walkforward"
OUT.mkdir(parents=True, exist_ok=True)


def download_data() -> pd.DataFrame:
    print(f"[1/5] Downloading {PERIOD} of daily data for {TICKER}...", flush=True)
    raw = yf.download(
        TICKER, period=PERIOD, interval="1d",
        auto_adjust=True, progress=False, threads=False,
    )
    if raw.empty:
        raise RuntimeError("No data downloaded. Check internet connection and ticker symbol.")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise RuntimeError(f"Downloaded data is missing columns: {missing}")

    df = raw[required].copy()
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) < 700:
        raise RuntimeError(f"Only {len(df)} rows downloaded; expected substantially more history.")
    print(f"      {len(df)} observations: {df.index.min().date()} to {df.index.max().date()}", flush=True)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features at t use only market data known by the close of t."""
    close, volume = df["Close"], df["Volume"]
    returns = close.pct_change()
    x = pd.DataFrame(index=df.index)

    for lag in [1, 2, 3, 5, 10, 15, 21, 30]:
        x[f"return_lag_{lag}"] = returns.shift(lag - 1) if lag > 1 else returns

    for window in [2, 5, 10, 21, 30]:
        x[f"return_{window}d"] = close.pct_change(window)

    for window in [5, 10, 21, 30, 50]:
        ma = close.rolling(window).mean()
        x[f"close_to_sma_{window}"] = close / ma - 1.0

    for window in [5, 10, 21, 30]:
        x[f"volatility_{window}d"] = returns.rolling(window).std()

    x["daily_range_pct"] = (df["High"] - df["Low"]) / close
    x["open_to_close_pct"] = (df["Close"] - df["Open"]) / df["Open"]
    x["volume_change_1d"] = volume.pct_change()
    x["volume_sma_ratio_5d"] = volume / volume.rolling(5).mean()
    x["volume_sma_ratio_21d"] = volume / volume.rolling(21).mean()
    x["day_of_week"] = df.index.dayofweek
    x["month"] = df.index.month

    return x.replace([np.inf, -np.inf], np.nan)


def make_model() -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=250,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        tree_method="hist",
        n_jobs=2,
        random_state=RANDOM_STATE,
        verbosity=0,
    )


def main():
    df = download_data()
    features = build_features(df)
    feature_cols = list(features.columns)
    close = df["Close"]
    last_date = df.index[-1]
    validation_start = df.index[-VALIDATION_DAYS]
    metrics_rows, summary_rows = [], []

    print("[2/5] Preparing direct return targets and walk-forward validation...", flush=True)
    print(f"      Validation target window: {validation_start.date()} to {last_date.date()}", flush=True)
    print(f"      Retraining every {RETRAIN_EVERY} forecast origins.", flush=True)

    for horizon in HORIZONS:
        print(f"\n[3/5] Evaluating {horizon}-trading-day horizon...", flush=True)
        target_return = (close.shift(-horizon) / close - 1.0).rename("target_return")
        target_close = close.shift(-horizon).rename("target_close")
        target_date = pd.Series(df.index, index=df.index).shift(-horizon).rename("target_date")

        table = features.copy()
        table["target_return"] = target_return
        table["target_close"] = target_close
        table["target_date"] = target_date
        table = table.replace([np.inf, -np.inf], np.nan)

        # Validation origins are dates whose future target falls within the last 252 sessions.
        origin_mask = (
            table["target_date"].notna()
            & (table["target_date"] >= validation_start)
            & (table["target_date"] <= last_date)
        )
        origins = table.index[origin_mask]
        origins = origins[table.loc[origins, feature_cols].notna().all(axis=1)]

        if len(origins) < 50:
            raise RuntimeError(f"Too few valid forecast origins for {horizon}D: {len(origins)}")

        records = []
        cached_model = None
        cached_train_end = None

        for i, origin in enumerate(origins):
            # At origin t, train only on examples with outcomes known before t.
            eligible = (
                table["target_date"].notna()
                & (table["target_date"] < origin)
                & table["target_return"].notna()
                & table[feature_cols].notna().all(axis=1)
            )
            train = table.loc[eligible]
            if len(train) < MIN_TRAIN_ROWS:
                continue

            should_retrain = cached_model is None or i % RETRAIN_EVERY == 0
            # To keep runtime manageable, refit every N forecast origins. Between refits,
            # use the last fitted model; this is a deliberate approximation to daily retraining.
            if should_retrain:
                cached_model = make_model()
                cached_model.fit(train[feature_cols], train["target_return"], verbose=False)
                cached_train_end = train.index[-1]

            x_now = table.loc[[origin], feature_cols]
            predicted_return = float(cached_model.predict(x_now)[0])
            current_price = float(close.loc[origin])
            actual_price = float(table.loc[origin, "target_close"])
            actual_return = float(table.loc[origin, "target_return"])
            predicted_price = current_price * (1.0 + predicted_return)
            baseline_price = current_price
            baseline_return = 0.0

            records.append({
                "forecast_origin_date": origin.date().isoformat(),
                "target_date": pd.Timestamp(table.loc[origin, "target_date"]).date().isoformat(),
                "horizon_trading_days": horizon,
                "origin_close": current_price,
                "actual_close": actual_price,
                "predicted_close": predicted_price,
                "baseline_predicted_close": baseline_price,
                "actual_return_pct": actual_return * 100.0,
                "predicted_return_pct": predicted_return * 100.0,
                "baseline_return_pct": baseline_return,
                "absolute_error": abs(actual_price - predicted_price),
                "baseline_absolute_error": abs(actual_price - baseline_price),
            })

            if (i + 1) % 50 == 0:
                print(f"      Processed {i + 1}/{len(origins)} origins...", flush=True)

        pred_df = pd.DataFrame(records)
        if pred_df.empty:
            raise RuntimeError(f"No valid predictions generated for {horizon}D.")

        pred_df.to_csv(OUT / f"predictions_{horizon}d.csv", index=False)

        actual = pred_df["actual_close"].to_numpy()
        predicted = pred_df["predicted_close"].to_numpy()
        baseline = pred_df["baseline_predicted_close"].to_numpy()
        actual_ret = pred_df["actual_return_pct"].to_numpy()
        predicted_ret = pred_df["predicted_return_pct"].to_numpy()

        rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
        mae = float(mean_absolute_error(actual, predicted))
        baseline_rmse = float(np.sqrt(mean_squared_error(actual, baseline)))
        baseline_mae = float(mean_absolute_error(actual, baseline))
        ret_mae = float(mean_absolute_error(actual_ret, predicted_ret))
        baseline_ret_mae = float(mean_absolute_error(actual_ret, np.zeros_like(actual_ret)))

        # Directional accuracy compares sign of the predicted return with actual return.
        direction = float((np.sign(predicted_ret) == np.sign(actual_ret)).mean() * 100.0)
        baseline_direction = float((np.sign(np.zeros_like(actual_ret)) == np.sign(actual_ret)).mean() * 100.0)

        metrics_rows.append({
            "ticker": TICKER,
            "horizon_trading_days": horizon,
            "validation_predictions": len(pred_df),
            "validation_target_start": pred_df["target_date"].min(),
            "validation_target_end": pred_df["target_date"].max(),
            "xgb_price_rmse": rmse,
            "xgb_price_mae": mae,
            "naive_price_rmse": baseline_rmse,
            "naive_price_mae": baseline_mae,
            "xgb_return_mae_percentage_points": ret_mae,
            "naive_return_mae_percentage_points": baseline_ret_mae,
            "xgb_directional_accuracy_pct": direction,
            "naive_directional_accuracy_pct_zero_return": baseline_direction,
            "price_mae_improvement_pct_vs_naive": (
                (baseline_mae - mae) / baseline_mae * 100.0 if baseline_mae else np.nan
            ),
            "return_mae_improvement_pct_vs_naive": (
                (baseline_ret_mae - ret_mae) / baseline_ret_mae * 100.0
                if baseline_ret_mae else np.nan
            ),
        })

        # Save a final model trained on all eligible labeled examples for future use.
        final_train = table.loc[
            table["target_return"].notna()
            & table["target_date"].notna()
            & (table["target_date"] <= last_date)
            & table[feature_cols].notna().all(axis=1)
        ]
        final_model = make_model()
        final_model.fit(final_train[feature_cols], final_train["target_return"], verbose=False)
        final_model.save_model(str(OUT / f"xgboost_{horizon}d.ubj"))
        with open(OUT / f"xgboost_{horizon}d.json", "w", encoding="utf-8") as f:
            json.dump({
                "ticker": TICKER,
                "horizon_trading_days": horizon,
                "target": "future close-to-close return",
                "feature_columns": feature_cols,
                "parameters": final_model.get_params(),
                "validation_protocol": "expanding walk-forward; train target date strictly before forecast origin",
                "retrain_every_origins": RETRAIN_EVERY,
            }, f, indent=2, default=str)

        importance = pd.DataFrame({
            "feature": feature_cols,
            "importance": final_model.feature_importances_,
        }).sort_values("importance", ascending=False)
        importance.to_csv(OUT / f"feature_importance_{horizon}d.csv", index=False)

        plot_df = pred_df.sort_values("target_date")
        plt.figure(figsize=(11, 5))
        plt.plot(pd.to_datetime(plot_df["target_date"]), plot_df["actual_close"], label="Actual close")
        plt.plot(pd.to_datetime(plot_df["target_date"]), plot_df["predicted_close"], label=f"XGBoost {horizon}D")
        plt.plot(pd.to_datetime(plot_df["target_date"]), plot_df["baseline_predicted_close"], label="Naive baseline", alpha=0.8)
        plt.title(f"{TICKER}: {horizon}-Trading-Day Return Forecast vs Baseline")
        plt.xlabel("Target date")
        plt.ylabel("Adjusted closing price (INR)")
        plt.legend()
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(OUT / f"forecast_{horizon}d.png", dpi=160)
        plt.close()

        # Latest-origin forecast from final model. Latest feature row uses current-known data.
        latest_x = features.iloc[[-1]][feature_cols]
        latest_return = float(final_model.predict(latest_x)[0])
        current = float(close.iloc[-1])
        summary_rows.append({
            "ticker": TICKER,
            "as_of_date": last_date.date().isoformat(),
            "current_adjusted_close": current,
            "horizon_trading_days": horizon,
            "predicted_return_pct": latest_return * 100.0,
            "predicted_close": current * (1.0 + latest_return),
            "naive_predicted_close": current,
            "warning": "Research estimate only; not a trading recommendation.",
        })

        print(
            f"      N={len(pred_df)} | XGB MAE ₹{mae:.2f} vs naive ₹{baseline_mae:.2f} | "
            f"XGB return MAE {ret_mae:.3f} pp vs naive {baseline_ret_mae:.3f} pp | "
            f"direction={direction:.1f}%",
            flush=True,
        )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(OUT / "metrics.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(OUT / "forecast_summary.csv", index=False)

    print("\n[5/5] COMPLETE", flush=True)
    print(f"Outputs: {OUT}", flush=True)
    print(metrics_df[[
        "horizon_trading_days", "validation_predictions",
        "xgb_price_mae", "naive_price_mae",
        "xgb_return_mae_percentage_points", "naive_return_mae_percentage_points",
        "xgb_directional_accuracy_pct"
    ]].to_string(index=False), flush=True)
    print("\nDo not use these forecasts as trading signals until the strategy is separately backtested.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr, flush=True)
        raise
