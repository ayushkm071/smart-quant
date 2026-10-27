"""XGBoost return forecasting with naive and momentum benchmarks for TCS.NS, including a 45-trading-day horizon."""
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
OUT = ROOT / "results" / "xgboost_return_walkforward_v3_45day"
OUT.mkdir(parents=True, exist_ok=True)

def download_data():
    print(f"[1/5] Downloading historic data for {TICKER}...", flush=True)
    raw = yf.download(TICKER,  period="5y", interval="1d", auto_adjust=True,
                      progress=False, threads=False)
    if raw.empty:
        raise RuntimeError("No data downloaded. Check internet connection and ticker.")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    cols = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in cols if c not in raw.columns]
    if missing:
        raise RuntimeError(f"Missing columns: {missing}")
    df = raw[cols].copy()
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) < 700:
        raise RuntimeError(f"Only {len(df)} observations downloaded.")
    print(f"      {len(df)} rows: {df.index.min().date()} to {df.index.max().date()}", flush=True)
    return df

def build_features(df):
    close, volume = df["Close"], df["Volume"]
    returns = close.pct_change()
    x = pd.DataFrame(index=df.index)
    for lag in [1, 2, 3, 5, 10, 15, 21, 30]:
        x[f"return_lag_{lag}"] = returns.shift(lag - 1)
    for w in [2, 5, 10, 21, 30]:
        x[f"return_{w}d"] = close.pct_change(w)
    for w in [5, 10, 21, 30, 50]:
        x[f"close_to_sma_{w}"] = close / close.rolling(w).mean() - 1
    for w in [5, 10, 21, 30]:
        x[f"volatility_{w}d"] = returns.rolling(w).std()
    x["daily_range_pct"] = (df["High"] - df["Low"]) / close
    x["open_to_close_pct"] = (df["Close"] - df["Open"]) / df["Open"]
    x["volume_change_1d"] = volume.pct_change()
    x["volume_sma_ratio_5d"] = volume / volume.rolling(5).mean()
    x["volume_sma_ratio_21d"] = volume / volume.rolling(21).mean()
    x["day_of_week"] = df.index.dayofweek
    x["month"] = df.index.month
    return x.replace([np.inf, -np.inf], np.nan)

def make_model():
    return XGBRegressor(objective="reg:squarederror", n_estimators=250,
        max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=2.0, tree_method="hist", n_jobs=2, random_state=42, verbosity=0)

def main():
    df = download_data()
    features = build_features(df)
    cols = list(features.columns)
    close = df["Close"]
    last_date = df.index[-1]
    validation_start = df.index[-VALIDATION_DAYS]
    metrics, summaries = [], []
    print("[2/5] Walk-forward validation setup...", flush=True)
    print(f"      Target-date window: {validation_start.date()} to {last_date.date()}", flush=True)
    print(f"      Retrain every {RETRAIN_EVERY} forecast origins.", flush=True)

    for h in HORIZONS:
        print(f"\\n[3/5] Horizon {h} trading day(s)", flush=True)
        target_ret = (close.shift(-h) / close - 1).rename("target_return")
        target_price = close.shift(-h).rename("target_close")
        target_date = pd.Series(df.index, index=df.index).shift(-h).rename("target_date")
        table = features.copy()
        table["target_return"], table["target_close"], table["target_date"] = target_ret, target_price, target_date
        table = table.replace([np.inf, -np.inf], np.nan)
        mask = table["target_date"].notna() & (table["target_date"] >= validation_start) & (table["target_date"] <= last_date)
        origins = list(table.index[mask])
        origins = [o for o in origins if table.loc[[o], cols].notna().all(axis=1).iloc[0]]
        if len(origins) < 50:
            raise RuntimeError(f"Too few validation origins for {h}D: {len(origins)}")

        records, model, last_fit_i, last_fit_origin, last_train_target_date = [], None, None, None, None
        for i, origin in enumerate(origins):
            refit = model is None or (i - last_fit_i) >= RETRAIN_EVERY
            if refit:
                train_mask = (table["target_date"].notna() & (table["target_date"] < origin)
                    & table["target_return"].notna() & table[cols].notna().all(axis=1))
                train = table.loc[train_mask]
                if len(train) < MIN_TRAIN_ROWS:
                    continue
                model = make_model()
                model.fit(train[cols], train["target_return"], verbose=False)
                last_fit_i, last_fit_origin = i, origin
                last_train_target_date = pd.Timestamp(train["target_date"].max())
                if not last_train_target_date < origin:
                    raise RuntimeError(f"Leakage guard failed: {last_train_target_date} >= {origin}")

            pred_ret = float(model.predict(table.loc[[origin], cols])[0])
            origin_price = float(close.loc[origin])
            actual_price = float(table.loc[origin, "target_close"])
            actual_ret = float(table.loc[origin, "target_return"])
            naive_price = origin_price
            trailing_ret = float(close.loc[origin] / close.shift(h).loc[origin] - 1)
            momentum_price = origin_price * (1 + trailing_ret)
            records.append({
                "forecast_origin_date": origin.date().isoformat(),
                "target_date": pd.Timestamp(table.loc[origin, "target_date"]).date().isoformat(),
                "horizon_trading_days": h, "origin_close": origin_price,
                "actual_close": actual_price, "actual_return_pct": actual_ret * 100,
                "xgb_predicted_return_pct": pred_ret * 100,
                "xgb_predicted_close": origin_price * (1 + pred_ret),
                "naive_predicted_return_pct": 0.0, "naive_predicted_close": naive_price,
                "momentum_trailing_return_pct": trailing_ret * 100,
                "momentum_predicted_return_pct": trailing_ret * 100,
                "momentum_predicted_close": momentum_price,
                "xgb_absolute_error": abs(actual_price - origin_price * (1 + pred_ret)),
                "naive_absolute_error": abs(actual_price - naive_price),
                "momentum_absolute_error": abs(actual_price - momentum_price),
                "model_last_fit_origin": last_fit_origin.date().isoformat(),
                "training_latest_target_date": last_train_target_date.date().isoformat()
            })
            if (i + 1) % 50 == 0:
                print(f"      Processed {i+1}/{len(origins)} origins...", flush=True)

        pred = pd.DataFrame(records)
        if pred.empty:
            raise RuntimeError(f"No predictions generated for {h}D.")
        pred.to_csv(OUT / f"predictions_{h}d.csv", index=False)
        actual_p = pred["actual_close"].to_numpy()
        xgb_p, naive_p, mom_p = [pred[c].to_numpy() for c in
            ["xgb_predicted_close", "naive_predicted_close", "momentum_predicted_close"]]
        actual_r = pred["actual_return_pct"].to_numpy()
        xgb_r, naive_r, mom_r = [pred[c].to_numpy() for c in
            ["xgb_predicted_return_pct", "naive_predicted_return_pct", "momentum_predicted_return_pct"]]
        rmse = lambda a,b: float(np.sqrt(mean_squared_error(a,b)))
        mae = lambda a,b: float(mean_absolute_error(a,b))
        direction = lambda a,b: float((np.sign(a) == np.sign(b)).mean() * 100)

        xgb_p_mae, naive_p_mae, mom_p_mae = mae(actual_p,xgb_p), mae(actual_p,naive_p), mae(actual_p,mom_p)
        xgb_r_mae, naive_r_mae, mom_r_mae = mae(actual_r,xgb_r), mae(actual_r,naive_r), mae(actual_r,mom_r)
        metrics.append({
            "ticker": TICKER, "horizon_trading_days": h, "validation_predictions": len(pred),
            "validation_target_start": pred["target_date"].min(), "validation_target_end": pred["target_date"].max(),
            "xgb_price_rmse": rmse(actual_p,xgb_p), "naive_price_rmse": rmse(actual_p,naive_p), "momentum_price_rmse": rmse(actual_p,mom_p),
            "xgb_price_mae": xgb_p_mae, "naive_price_mae": naive_p_mae, "momentum_price_mae": mom_p_mae,
            "xgb_return_mae_percentage_points": xgb_r_mae, "naive_return_mae_percentage_points": naive_r_mae,
            "momentum_return_mae_percentage_points": mom_r_mae,
            "xgb_directional_accuracy_pct": direction(xgb_r,actual_r),
            "momentum_directional_accuracy_pct": direction(mom_r,actual_r),
            "xgb_price_mae_improvement_vs_naive_pct": (naive_p_mae-xgb_p_mae)/naive_p_mae*100,
            "xgb_price_mae_improvement_vs_momentum_pct": (mom_p_mae-xgb_p_mae)/mom_p_mae*100,
            "xgb_return_mae_improvement_vs_naive_pct": (naive_r_mae-xgb_r_mae)/naive_r_mae*100 if naive_r_mae else np.nan,
            "xgb_return_mae_improvement_vs_momentum_pct": (mom_r_mae-xgb_r_mae)/mom_r_mae*100 if mom_r_mae else np.nan
        })

        # Final fit for future inference; its targets are fully observed by latest date.
        final_mask = table["target_date"].notna() & (table["target_date"] <= last_date) & table["target_return"].notna() & table[cols].notna().all(axis=1)
        final_train = table.loc[final_mask]
        final_model = make_model()
        final_model.fit(final_train[cols], final_train["target_return"], verbose=False)
        final_model.save_model(str(OUT / f"xgboost_{h}d.ubj"))
        with open(OUT / f"xgboost_{h}d.json", "w", encoding="utf-8") as f:
            json.dump({"ticker":TICKER, "horizon_trading_days":h, "target":"future close-to-close return",
                "feature_columns":cols, "parameters":final_model.get_params(),
                "retrain_every_origins":RETRAIN_EVERY,
                "validation_rule":"training target date strictly before forecast origin"}, f, indent=2, default=str)
        latest_x = features.iloc[[-1]][cols]
        latest_pred_ret = float(final_model.predict(latest_x)[0])
        current = float(close.iloc[-1])
        trailing = float(close.iloc[-1] / close.iloc[-1-h] - 1)
        summaries.append({"ticker":TICKER, "as_of_date":last_date.date().isoformat(),
            "current_adjusted_close":current, "horizon_trading_days":h,
            "xgb_predicted_return_pct":latest_pred_ret*100,
            "xgb_predicted_close":current*(1+latest_pred_ret),
            "naive_predicted_close":current, "momentum_trailing_return_pct":trailing*100,
            "momentum_predicted_close":current*(1+trailing),
            "warning":"Research estimate only; not a trading recommendation."})

        plot = pred.sort_values("target_date")
        plt.figure(figsize=(11,5))
        plt.plot(pd.to_datetime(plot["target_date"]),plot["actual_close"],label="Actual close")
        plt.plot(pd.to_datetime(plot["target_date"]),plot["xgb_predicted_close"],label="XGBoost")
        plt.plot(pd.to_datetime(plot["target_date"]),plot["naive_predicted_close"],label="Naive baseline",alpha=.8)
        plt.plot(pd.to_datetime(plot["target_date"]),plot["momentum_predicted_close"],label="Momentum benchmark",alpha=.8)
        plt.title(f"{TICKER}: {h}-Trading-Day Forecast vs Benchmarks")
        plt.xlabel("Target date"); plt.ylabel("Adjusted closing price (INR)")
        plt.legend(); plt.grid(True,alpha=.25); plt.tight_layout()
        plt.savefig(OUT / f"forecast_{h}d.png",dpi=160); plt.close()
        print(f"      Price MAE: XGB ₹{xgb_p_mae:.2f} | naive ₹{naive_p_mae:.2f} | momentum ₹{mom_p_mae:.2f}",flush=True)
        print(f"      Return MAE (pp): XGB {xgb_r_mae:.3f} | naive {naive_r_mae:.3f} | momentum {mom_r_mae:.3f}",flush=True)

    md = pd.DataFrame(metrics)
    md.to_csv(OUT / "metrics.csv",index=False)
    pd.DataFrame(summaries).to_csv(OUT / "forecast_summary.csv",index=False)
    print("\\n[5/5] COMPLETE",flush=True)
    print(f"Outputs saved to {OUT}",flush=True)
    print(md[["horizon_trading_days","xgb_price_mae","naive_price_mae","momentum_price_mae",
        "xgb_return_mae_percentage_points","naive_return_mae_percentage_points",
        "momentum_return_mae_percentage_points","xgb_directional_accuracy_pct"]].to_string(index=False),flush=True)
    print("Review metrics before drawing conclusions; no live trading signals are generated.",flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\\nERROR: {exc}",file=sys.stderr,flush=True)
        raise
