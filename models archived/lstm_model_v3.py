"""Direct LSTM return forecasting for TCS.NS across 1/5/21/30/45 trading days (v2).

What changed vs v1 (and why):
  1. More history (2010+) instead of a moving 5y window -> ~4x more training rows, several regimes.
  2. Stationary features (log returns / log ratios), clipped after scaling, calendar ordinals dropped.
  3. Target is scaled before training and a Huber loss is used (raw 1-day returns are ~0.01, so
     MSE was ~1e-4 and gradients were tiny; outliers also dominated).
  4. Smaller model (single LSTM(16) + dropout + L2) -> far fewer parameters than rows.
  5. Chronological fit / early-stop / calibration blocks with an embargo of `h` rows between them
     (overlapping h-day targets would otherwise leak across the block boundaries).
  6. Training rows are purged: only rows whose target was already realised at the first test origin.
  7. Seed ensemble (mean of N_SEEDS models) -> removes most of the prediction noise of a single LSTM.
  8. Target is centred on the training median (no learned drift) and predictions are multiplied by
     beta in [0, 1] fitted on the calibration block, but only if the calibration correlation passes a
     significance gate (overlapping targets => few independent observations).
     If the model has no skill, beta -> 0 and the forecast falls back to the naive (0% return) one.
  9. Walk-forward evaluation (N_FOLDS expanding-window refits over the last VALIDATION_DAYS days)
     instead of one fixed holdout, which is a much less noisy estimate and is closer to XGBoost WF.
 10. Data is cached to CSV so every run sees identical data.

Research experiment; not a trading recommendation.
"""
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["ABSL_MIN_LOG_LEVEL"] = "2"

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

# ----------------------------------------------------------------------------- config
TICKER = "TCS.NS"
START = "2010-01-01"
HORIZONS = [1, 5, 21, 30, 45]
LOOKBACK = 30
VALIDATION_DAYS = 252      # last ~1 trading year of *target dates* is the evaluation window
N_FOLDS = 4                # walk-forward blocks inside the evaluation window (~63 days each)
N_SEEDS = 3                # models averaged per fold/horizon
EPOCHS = 60
BATCH_SIZE = 64
CLIP = 5.0                 # clip standardised features to +-CLIP sigma
SKILL_Z = 1.65             # model is used only if calibration corr > SKILL_Z / sqrt(n_cal / h)
MIN_TRAIN = 500
SEED = 42

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "lstm_direct_walkforward_v2"
DATA_CACHE = ROOT / "data" / f"{TICKER}_daily.csv"


# ----------------------------------------------------------------------------- setup
def load_tf():
    try:
        import tensorflow as tf

        tf.get_logger().setLevel("ERROR")
        try:
            import absl.logging

            absl.logging.set_verbosity(absl.logging.ERROR)
        except Exception:
            pass
        return tf
    except Exception as exc:
        raise RuntimeError(
            "TensorFlow could not be imported. Use a Python version that has a TensorFlow wheel "
            f"(3.12/3.13) and install tensorflow. Original error: {exc}"
        ) from exc


def load_data(csv_path=None, refresh=False):
    """Load OHLCV from a CSV cache; download with yfinance only if the cache is missing."""
    path = Path(csv_path) if csv_path else DATA_CACHE
    if path.exists() and not refresh:
        print(f"[1/4] Loading cached data from {path}", flush=True)
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        import yfinance as yf  # imported lazily so cached runs work offline

        print(f"[1/4] Downloading daily data for {TICKER} from {START}...", flush=True)
        raw = yf.download(TICKER, start=START, interval="1d", auto_adjust=True, progress=False, threads=False)
        if raw.empty:
            raise RuntimeError("No data downloaded. Check internet connection and ticker.")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path)
        print(f"      cached to {path}", flush=True)

    needed = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns: {missing}")
    df = df[needed].copy()
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df[(df[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]
    if len(df) < 1500:
        raise RuntimeError(f"Only {len(df)} rows; need more history for this experiment.")
    print(f"      {len(df)} rows: {df.index.min().date()} to {df.index.max().date()}", flush=True)
    return df


# ----------------------------------------------------------------------------- features
def build_features(df):
    """Stationary, scale-free features. Everything at row t uses data up to and including day t."""
    close, volume = df["Close"], df["Volume"]
    logc = np.log(close)
    r1 = logc.diff()
    lv = np.log1p(volume)
    x = pd.DataFrame(index=df.index)
    for lag in [1, 2, 3, 5, 10]:
        x[f"ret_lag_{lag}"] = r1.shift(lag - 1)
    for w in [5, 10, 21, 42, 63]:
        x[f"ret_{w}d"] = logc.diff(w)
    for w in [10, 21, 50, 100]:
        x[f"close_to_sma_{w}"] = np.log(close / close.rolling(w).mean())
    for w in [5, 10, 21, 63]:
        x[f"vol_{w}d"] = r1.rolling(w).std()
    x["vol_ratio_5_21"] = x["vol_5d"] / x["vol_21d"]
    x["range_pct"] = np.log(df["High"] / df["Low"])
    x["open_to_close"] = np.log(close / df["Open"])
    x["dist_252d_high"] = np.log(close / close.rolling(252).max())
    x["log_volume_chg"] = lv.diff()
    x["log_volume_vs_21d"] = lv - lv.rolling(21).mean()
    return x.replace([np.inf, -np.inf], np.nan).dropna()


# ----------------------------------------------------------------------------- model
def build_model(tf, n_features, seed):
    tf.keras.utils.set_random_seed(seed)
    reg = tf.keras.regularizers.l2(1e-4)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(LOOKBACK, n_features)),
            tf.keras.layers.LSTM(16, dropout=0.2, kernel_regularizer=reg),
            tf.keras.layers.Dense(8, activation="relu", kernel_regularizer=reg),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
        loss=tf.keras.losses.Huber(delta=1.0),
    )
    return model


def windows(arr, end_positions):
    return np.stack([arr[p - LOOKBACK + 1 : p + 1] for p in end_positions]).astype(np.float32)


def fit_predict(tf, X_tr, y_tr, X_te, h, fold_seed):
    """Train a seed ensemble on chronological fit/early-stop blocks, then shrink with beta.

    The target is centred on the training median, so the network only learns the deviation from a
    "typical" return and the shrunk forecast is drift-free (it falls back to the naive 0% forecast).
    Returns (raw prediction incl. drift, shrunk prediction, beta, epochs list) in return units.
    """
    n = len(X_tr)
    n_fit, n_es = int(n * 0.75), int(n * 0.125)
    fit = slice(0, n_fit)
    es = slice(n_fit + h, n_fit + h + n_es)            # embargo of h rows between blocks
    cal = slice(n_fit + h + n_es + h, n)
    centre = float(np.median(y_tr[fit]))
    y_scale = float(np.std(y_tr[fit])) or 1.0
    ys = (y_tr - centre) / y_scale

    cal_preds, te_preds, epochs = [], [], []
    for s in range(N_SEEDS):
        model = build_model(tf, X_tr.shape[2], fold_seed + s)
        stopper = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
        hist = model.fit(
            X_tr[fit], ys[fit],
            validation_data=(X_tr[es], ys[es]),
            epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True, verbose=0, callbacks=[stopper],
        )
        epochs.append(len(hist.history["loss"]))
        cal_preds.append(model(X_tr[cal], training=False).numpy().reshape(-1))
        te_preds.append(model(X_te, training=False).numpy().reshape(-1))
        tf.keras.backend.clear_session()

    p_cal = np.mean(cal_preds, axis=0) * y_scale
    p_te = np.mean(te_preds, axis=0) * y_scale
    denom = float(np.sum(p_cal ** 2))
    beta = 0.0
    if denom > 0 and np.std(p_cal) > 0:
        # significance gate: h-day targets overlap, so the calibration block holds only ~n/h independent
        # observations; a correlation below ~1.65/sqrt(n/h) is indistinguishable from noise -> abstain.
        rho = float(np.corrcoef(p_cal, y_tr[cal])[0, 1])
        if rho > SKILL_Z / np.sqrt(max(len(p_cal) // h, 1)):
            beta = float(np.clip(np.sum(p_cal * (y_tr[cal] - centre)) / denom, 0.0, 1.0))
    return centre + p_te, beta * p_te, beta, epochs


# ----------------------------------------------------------------------------- main
def main(args):
    global OUT
    if args.out:
        OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)

    tf = load_tf()
    df = load_data(args.data_csv, args.refresh)
    feat = build_features(df)
    close_all = df["Close"]
    close = close_all.reindex(feat.index)
    last_date = df.index[-1]
    val_dates = df.index[-VALIDATION_DAYS:]
    val_start = val_dates[0]
    fold_chunks = [c for c in np.array_split(np.asarray(val_dates), N_FOLDS) if len(c)]
    dates_series = pd.Series(df.index, index=df.index)
    metric_rows = []

    print(f"[2/4] Walk-forward evaluation window: {val_start.date()} -> {last_date.date()} "
          f"({len(fold_chunks)} folds x {N_SEEDS} seeds per horizon)", flush=True)

    for h in HORIZONS:
        print(f"\n[3/4] Horizon {h}d", flush=True)
        t_ret = (close_all.shift(-h) / close_all - 1).reindex(feat.index).values
        t_price = close_all.shift(-h).reindex(feat.index).values
        t_date = dates_series.shift(-h).reindex(feat.index)
        t_date_arr = pd.to_datetime(t_date).values
        ok = np.isfinite(t_ret) & ~pd.isna(t_date_arr)

        pos_all = np.arange(LOOKBACK - 1, len(feat))
        pos_all = pos_all[ok[pos_all]]
        origins_all = feat.index[pos_all]
        tdates_all = pd.DatetimeIndex(t_date_arr[pos_all])

        parts = []
        for k, chunk in enumerate(fold_chunks):
            lo, hi = pd.Timestamp(chunk[0]), pd.Timestamp(chunk[-1])
            test_m = (tdates_all >= lo) & (tdates_all <= hi)
            if test_m.sum() < 10:
                continue
            first_test_origin = origins_all[test_m].min()
            # purge: only rows whose target was already realised before the first test origin
            train_m = tdates_all < first_test_origin
            if train_m.sum() < MIN_TRAIN:
                raise RuntimeError(f"{h}D fold {k}: only {train_m.sum()} training rows (need {MIN_TRAIN}).")

            last_train_origin = origins_all[train_m].max()
            scaler = StandardScaler().fit(feat.loc[:last_train_origin].values)
            arr = np.clip(scaler.transform(feat.values), -CLIP, CLIP)

            tr_pos, te_pos = pos_all[train_m], pos_all[test_m]
            X_tr, y_tr = windows(arr, tr_pos), t_ret[tr_pos]
            X_te = windows(arr, te_pos)

            raw, shrunk, beta, eps = fit_predict(tf, X_tr, y_tr, X_te, h, SEED + 1000 * k)
            origin_px = close.values[te_pos]
            parts.append(pd.DataFrame({
                "fold": k,
                "forecast_origin_date": origins_all[test_m].date,
                "target_date": tdates_all[test_m].date,
                "origin_close": origin_px,
                "actual_close": t_price[te_pos],
                "actual_return_pct": t_ret[te_pos] * 100,
                "lstm_raw_return_pct": raw * 100,
                "lstm_return_pct": shrunk * 100,
                "momentum_return_pct": (close_all.pct_change(h).reindex(feat.index).values[te_pos]) * 100,
                "beta": beta,
            }))
            print(f"      fold {k}: train={len(X_tr)} test={len(X_te)} beta={beta:.2f} epochs={eps}", flush=True)

        pdf = pd.concat(parts, ignore_index=True).sort_values("target_date").reset_index(drop=True)
        pdf["horizon_trading_days"] = h
        for col, src in [("lstm_predicted_close", "lstm_return_pct"),
                         ("lstm_raw_predicted_close", "lstm_raw_return_pct"),
                         ("momentum_predicted_close", "momentum_return_pct")]:
            pdf[col] = pdf["origin_close"] * (1 + pdf[src] / 100)
        pdf["naive_predicted_close"] = pdf["origin_close"]
        pdf.to_csv(OUT / f"predictions_{h}d.csv", index=False)

        a_px, a_pct = pdf["actual_close"].values, pdf["actual_return_pct"].values
        mae = lambda a, b: float(mean_absolute_error(a, b))
        rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, b)))
        direction = lambda a, b: float((np.sign(a) == np.sign(b)).mean() * 100)
        x_mae = mae(a_px, pdf["lstm_predicted_close"])
        n_mae = mae(a_px, pdf["naive_predicted_close"])
        m_mae = mae(a_px, pdf["momentum_predicted_close"])
        raw_mae = mae(a_px, pdf["lstm_raw_predicted_close"])
        corr = float(np.corrcoef(pdf["lstm_raw_return_pct"], a_pct)[0, 1]) if pdf["lstm_raw_return_pct"].std() > 0 else 0.0

        metric_rows.append({
            "ticker": TICKER, "forecast_method": "direct_lstm_v2", "horizon_trading_days": h,
            "validation_predictions": len(pdf),
            "effective_independent_obs_approx": len(pdf) // h,
            "validation_target_start": str(pdf.target_date.min()), "validation_target_end": str(pdf.target_date.max()),
            "mean_beta": float(pdf.groupby("fold")["beta"].first().mean()),
            "lstm_price_mae": x_mae, "lstm_raw_price_mae": raw_mae,
            "naive_price_mae": n_mae, "momentum_price_mae": m_mae,
            "lstm_price_rmse": rmse(a_px, pdf["lstm_predicted_close"]),
            "naive_price_rmse": rmse(a_px, pdf["naive_predicted_close"]),
            "momentum_price_rmse": rmse(a_px, pdf["momentum_predicted_close"]),
            "lstm_return_mae_percentage_points": mae(a_pct, pdf["lstm_return_pct"]),
            "naive_return_mae_percentage_points": mae(a_pct, np.zeros_like(a_pct)),
            "momentum_return_mae_percentage_points": mae(a_pct, pdf["momentum_return_pct"]),
            "lstm_directional_accuracy_pct": direction(pdf["lstm_raw_return_pct"], a_pct),
            "momentum_directional_accuracy_pct": direction(pdf["momentum_return_pct"], a_pct),
            "lstm_raw_return_corr": corr,
            "pred_std_pct": float(pdf["lstm_raw_return_pct"].std()), "actual_std_pct": float(np.std(a_pct)),
            "pred_mean_pct": float(pdf["lstm_raw_return_pct"].mean()), "actual_mean_pct": float(np.mean(a_pct)),
            "lstm_price_mae_improvement_vs_naive_pct": (n_mae - x_mae) / n_mae * 100,
            "lstm_price_mae_improvement_vs_momentum_pct": (m_mae - x_mae) / m_mae * 100,
        })
        with open(OUT / f"lstm_{h}d_metadata.json", "w", encoding="utf-8") as f:
            json.dump({
                "ticker": TICKER, "horizon_trading_days": h, "lookback_days": LOOKBACK,
                "feature_columns": list(feat.columns), "n_folds": len(fold_chunks), "n_seeds": N_SEEDS,
                "protocol": "expanding-window walk-forward; train rows purged to targets realised before first "
                            "test origin; scaler fit on training rows per fold; fit/early-stop/calibration blocks "
                            "separated by an h-row embargo; seed ensemble shrunk by beta in [0,1]",
                "warning": "Research estimate only; not a trading recommendation.",
            }, f, indent=2)

        plt.figure(figsize=(11, 5))
        x = pd.to_datetime(pdf.target_date)
        plt.plot(x, pdf.actual_close, label="Actual close")
        plt.plot(x, pdf.lstm_predicted_close, label="LSTM (ensemble, shrunk)")
        plt.plot(x, pdf.naive_predicted_close, label="Naive baseline", alpha=.8)
        plt.plot(x, pdf.momentum_predicted_close, label="Momentum benchmark", alpha=.5)
        plt.title(f"{TICKER}: {h}-trading-day direct LSTM v2 vs benchmarks")
        plt.xlabel("Target date"); plt.ylabel("Adjusted close (INR)")
        plt.legend(); plt.grid(True, alpha=.25); plt.tight_layout()
        plt.savefig(OUT / f"forecast_{h}d.png", dpi=150); plt.close()

        print(f"      Price MAE: LSTM {x_mae:.2f} | LSTM raw {raw_mae:.2f} | naive {n_mae:.2f} | momentum {m_mae:.2f}", flush=True)

    mdf = pd.DataFrame(metric_rows)
    mdf.to_csv(OUT / "metrics.csv", index=False)
    print("\n[4/4] COMPLETE", flush=True)
    print(f"Outputs saved to: {OUT}", flush=True)
    cols = ["horizon_trading_days", "mean_beta", "lstm_price_mae", "lstm_raw_price_mae", "naive_price_mae",
            "lstm_return_mae_percentage_points", "naive_return_mae_percentage_points",
            "lstm_raw_return_corr", "pred_std_pct", "actual_std_pct"]
    print(mdf[cols].round(3).to_string(index=False), flush=True)
    print("Read this as: LSTM MAE <= naive MAE means it adds value. mean_beta near 0 means the model found no "
          "usable signal and defaulted to naive. Overlapping targets => effective sample size is ~n/h, so "
          "differences of a few percent are noise.", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-csv", help="OHLCV CSV to use/cache (default: ../data/TCS.NS_daily.csv)")
    ap.add_argument("--refresh", action="store_true", help="re-download data even if cache exists")
    ap.add_argument("--folds", type=int, help="override N_FOLDS")
    ap.add_argument("--seeds", type=int, help="override N_SEEDS")
    ap.add_argument("--epochs", type=int, help="override EPOCHS")
    ap.add_argument("--out", help="output directory")
    a = ap.parse_args()
    if a.folds:
        N_FOLDS = a.folds
    if a.seeds:
        N_SEEDS = a.seeds
    if a.epochs:
        EPOCHS = a.epochs
    try:
        main(a)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr, flush=True)
        raise