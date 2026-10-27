"""XGBoost direct return forecasting for TCS.NS at 1/5/21/30/45 trading days (v3).

Your v1 protocol (walk-forward, retrain every 5 origins, training targets strictly before the
origin) was already sound and is kept. What changed, and why:
  1. Objective is reg:absoluteerror. v1 trained on squared error but was judged on MAE; with fat-tailed
     returns the squared-error model chases outliers and predicts large noisy values.
  2. The target is centred on the training median, so the model predicts only the deviation from
     "typical return". v1 learned the 2010-2024 uptrend as a constant positive drift, which is exactly
     what hurts when the evaluation year trends differently (directional accuracy of 40% at 5d).
  3. Shrinkage: every refit first trains on an early block, measures on a later embargoed block, and
     fits beta in [0, 1] so that forecast = beta * deviation. beta is set to 0 unless the calibration
     correlation is statistically distinguishable from noise (overlapping targets => ~n/h independent
     observations), so no skill -> naive forecast.
  4. Much smaller model (depth 2, min_child_weight 40, more regularisation, 200 trees). Overlapping
     h-day targets mean the effective sample size is ~n/h, so deep trees just memorise noise.
  5. Same stationary features as the LSTM v2 (log returns/ratios, no day_of_week/month), so the two
     models can be compared fairly. Data is cached to the same CSV.
  6. Extra benchmark + diagnostics (drift benchmark, up-rate, prediction std, correlation) so you can
     see *why* a model wins or loses, not just that it does.

Research experiment; not a trading recommendation.
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=FutureWarning)

# ----------------------------------------------------------------------------- config
TICKER = "TCS.NS"
START = "2010-01-01"
HORIZONS = [1, 5, 21, 30, 45]
VALIDATION_DAYS = 252
MIN_TRAIN_ROWS = 500
RETRAIN_EVERY = 5
SKILL_Z = 1.65                  # model is used only if calibration corr > SKILL_Z / sqrt(n_cal / h)
CAL_FRAC = 0.20                 # last 20% of the training rows (after an h-row embargo) calibrates beta
OBJECTIVE = "reg:absoluteerror"  # matches the MAE metric; use "reg:squarederror" to reproduce v1 behaviour
SEED = 42

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "xgboost_v3"
DATA_CACHE = ROOT / "data" / f"{TICKER}_daily.csv"


# ----------------------------------------------------------------------------- data
def load_data(csv_path=None, refresh=False):
    """Load OHLCV from the CSV cache (shared with the LSTM script); download only if missing."""
    path = Path(csv_path) if csv_path else DATA_CACHE
    if path.exists() and not refresh:
        print(f"[1/4] Loading cached data from {path}", flush=True)
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        import yfinance as yf  # lazy import so cached runs work offline

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


def build_features(df):
    """Stationary, scale-free features (identical to the LSTM v2 feature set)."""
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
    return x.replace([np.inf, -np.inf], np.nan)


"""XGBoost direct return forecasting for TCS.NS at 1/5/21/30/45 trading days (v2).

Your v1 protocol (walk-forward, retrain every 5 origins, training targets strictly before the
origin) was already sound and is kept. What changed, and why:
  1. Objective is reg:absoluteerror. v1 trained on squared error but was judged on MAE; with fat-tailed
     returns the squared-error model chases outliers and predicts large noisy values.
  2. The target is centred on the training median, so the model predicts only the deviation from
     "typical return". v1 learned the 2010-2024 uptrend as a constant positive drift, which is exactly
     what hurts when the evaluation year trends differently (directional accuracy of 40% at 5d).
  3. Shrinkage: every refit first trains on an early block, measures on a later embargoed block, and
     fits beta in [0, 1] so that forecast = beta * deviation. beta is set to 0 unless the calibration
     correlation is statistically distinguishable from noise (overlapping targets => ~n/h independent
     observations), so no skill -> naive forecast.
  4. Much smaller model (depth 2, min_child_weight 40, more regularisation, 200 trees). Overlapping
     h-day targets mean the effective sample size is ~n/h, so deep trees just memorise noise.
  5. Same stationary features as the LSTM v2 (log returns/ratios, no day_of_week/month), so the two
     models can be compared fairly. Data is cached to the same CSV.
  6. Extra benchmark + diagnostics (drift benchmark, up-rate, prediction std, correlation) so you can
     see *why* a model wins or loses, not just that it does.

Research experiment; not a trading recommendation.
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=FutureWarning)

# ----------------------------------------------------------------------------- config
TICKER = "TCS.NS"
START = "2010-01-01"
HORIZONS = [1, 5, 21, 30, 45]
VALIDATION_DAYS = 252
MIN_TRAIN_ROWS = 500
RETRAIN_EVERY = 5
SKILL_Z = 1.65                  # model is used only if calibration corr > SKILL_Z / sqrt(n_cal / h)
CAL_FRAC = 0.20                 # last 20% of the training rows (after an h-row embargo) calibrates beta
OBJECTIVE = "reg:absoluteerror"  # matches the MAE metric; use "reg:squarederror" to reproduce v1 behaviour
SEED = 42

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "xgboost_return_walkforward_v4"
DATA_CACHE = ROOT / "data" / f"{TICKER}_daily.csv"


# ----------------------------------------------------------------------------- data
def load_data(csv_path=None, refresh=False):
    """Load OHLCV from the CSV cache (shared with the LSTM script); download only if missing."""
    path = Path(csv_path) if csv_path else DATA_CACHE
    if path.exists() and not refresh:
        print(f"[1/4] Loading cached data from {path}", flush=True)
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        import yfinance as yf  # lazy import so cached runs work offline

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


def build_features(df):
    """Stationary, scale-free features (identical to the LSTM v2 feature set)."""
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
    return x.replace([np.inf, -np.inf], np.nan)


# ----------------------------------------------------------------------------- model and temporal validation
from itertools import product
from scipy.stats import pearsonr

# Small, predeclared candidate set; avoids a huge Cartesian grid.
PARAM_CANDIDATES = [
    {"n_estimators": 200, "max_depth": 2, "learning_rate": 0.03, "min_child_weight": 40, "subsample": 0.70, "colsample_bytree": 0.70, "reg_lambda": 5.0},
    {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.03, "min_child_weight": 40, "subsample": 0.70, "colsample_bytree": 0.70, "reg_lambda": 10.0},
    {"n_estimators": 300, "max_depth": 2, "learning_rate": 0.01, "min_child_weight": 20, "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 10.0},
    {"n_estimators": 100, "max_depth": 2, "learning_rate": 0.05, "min_child_weight": 60, "subsample": 0.85, "colsample_bytree": 0.70, "reg_lambda": 20.0},
    {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.03, "min_child_weight": 60, "subsample": 0.70, "colsample_bytree": 0.85, "reg_lambda": 20.0},
]


def make_model(params=None):
    p = dict(PARAM_CANDIDATES[0] if params is None else params)
    return XGBRegressor(objective=OBJECTIVE, tree_method="hist", n_jobs=2,
                        random_state=SEED, verbosity=0, **p)


def make_table(df, features, h):
    close = df["Close"]
    t = features.copy()
    t["target_return"] = close.shift(-h) / close - 1.0
    t["target_close"] = close.shift(-h)
    t["target_date"] = pd.Series(df.index, index=df.index).shift(-h)
    t["momentum_return"] = close.pct_change(h)
    t["target_date"] = pd.to_datetime(t["target_date"])
    t = t.replace([np.inf, -np.inf], np.nan)
    valid = features.notna().all(axis=1) & t["target_return"].notna() & t["target_date"].notna()
    return t, valid


def tune_params(table, valid, cols, h, test_start):
    """Select parameters using only pre-test data and purged chronological folds."""
    pool = table.loc[valid & (table["target_date"] < test_start)].copy()
    if len(pool) < MIN_TRAIN_ROWS + 4 * h + 100:
        return dict(PARAM_CANDIDATES[0]), pd.DataFrame()
    # Three expanding-window validation folds; each validation target is after the training targets,
    # and the h-row embargo prevents overlapping target windows from crossing the boundary.
    fold_size = max(60, min(100, len(pool) // 8))
    starts = [len(pool) - 3 * fold_size, len(pool) - 2 * fold_size, len(pool) - fold_size]
    results = []
    for pi, params in enumerate(PARAM_CANDIDATES):
        errs = []
        for start in starts:
            if start - h < MIN_TRAIN_ROWS or start + fold_size > len(pool):
                continue
            tr = pool.iloc[:start-h]
            va = pool.iloc[start:start+fold_size]
            centre = float(np.median(tr["target_return"]))
            model = make_model(params)
            model.fit(tr[cols], tr["target_return"].to_numpy() - centre, verbose=False)
            pred = centre + model.predict(va[cols])
            errs.extend(np.abs(va["target_return"].to_numpy() - pred).tolist())
        score = float(np.mean(errs)) if errs else np.inf
        results.append({"candidate": pi, **params, "validation_return_mae": score, "validation_predictions": len(errs)})
    result = pd.DataFrame(results).sort_values("validation_return_mae")
    if result.empty or not np.isfinite(result.iloc[0]["validation_return_mae"]):
        return dict(PARAM_CANDIDATES[0]), result
    chosen_idx = int(result.iloc[0]["candidate"])
    return dict(PARAM_CANDIDATES[chosen_idx]), result


def fit_with_shrinkage(train, cols, h, params):
    """Fit a centred model and calibrate shrinkage on a later, embargoed block.

    Calibration uses a chronological holdout. The correlation gate is retained only as a
    conservative heuristic; it is not a formal significance test for overlapping returns.
    """
    n = len(train)
    n_cal = max(int(n * CAL_FRAC), 100)
    n_fit = n - n_cal - h
    if n_fit < 250:
        raise RuntimeError(f"Not enough rows for calibration split (n={n}, h={h}).")
    X, y = train[cols], train["target_return"].to_numpy()
    centre_fit = float(np.median(y[:n_fit]))
    cal_model = make_model(params)
    cal_model.fit(X.iloc[:n_fit], y[:n_fit] - centre_fit, verbose=False)
    p_cal = cal_model.predict(X.iloc[n-n_cal:])
    dev_cal = y[n-n_cal:] - centre_fit
    beta = 0.0
    if len(p_cal) > 2 and np.std(p_cal) > 0 and np.std(dev_cal) > 0:
        rho = float(np.corrcoef(p_cal, dev_cal)[0, 1])
        # Conservative rule of thumb only; overlapping observations mean this is approximate.
        threshold = SKILL_Z / np.sqrt(max(n_cal // h, 1))
        if np.isfinite(rho) and rho > threshold:
            denom = float(np.dot(p_cal, p_cal))
            if denom > 0:
                beta = float(np.clip(np.dot(p_cal, dev_cal) / denom, 0.0, 1.0))
    centre = float(np.median(y))
    model = make_model(params)
    model.fit(X, y - centre, verbose=False)
    return model, centre, beta


def metrics_for(pred):
    ap = pred["actual_close"].to_numpy()
    ar = pred["actual_return_pct"].to_numpy()
    names = ["xgb", "xgb_raw", "naive", "drift", "momentum"]
    P = {k: pred[f"{k}_predicted_close"].to_numpy() for k in names}
    R = {k: pred[f"{k}_predicted_return_pct"].to_numpy() for k in names}
    mae = lambda a, b: float(mean_absolute_error(a, b))
    rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, b)))
    direction = lambda a, b: float((np.sign(a) == np.sign(b)).mean() * 100)
    pm = {k: mae(ap, v) for k, v in P.items()}
    rm = {k: mae(ar, v) for k, v in R.items()}
    dev = pred["xgb_deviation_pct"].to_numpy()
    corr = float(np.corrcoef(dev, ar)[0, 1]) if dev.std() > 0 and ar.std() > 0 else np.nan
    return {
        "test_predictions": len(pred),
        "test_target_start": pred["target_date"].min(), "test_target_end": pred["target_date"].max(),
        "effective_independent_obs_approx": len(pred) // int(pred["horizon_trading_days"].iloc[0]),
        "mean_beta": float(pred["shrinkage_beta"].mean()),
        "xgb_price_mae": pm["xgb"], "xgb_raw_price_mae": pm["xgb_raw"], "naive_price_mae": pm["naive"],
        "drift_price_mae": pm["drift"], "momentum_price_mae": pm["momentum"],
        "xgb_price_rmse": rmse(ap, P["xgb"]), "naive_price_rmse": rmse(ap, P["naive"]),
        "momentum_price_rmse": rmse(ap, P["momentum"]),
        "xgb_return_mae_percentage_points": rm["xgb"], "xgb_raw_return_mae_percentage_points": rm["xgb_raw"],
        "naive_return_mae_percentage_points": rm["naive"], "drift_return_mae_percentage_points": rm["drift"],
        "momentum_return_mae_percentage_points": rm["momentum"],
        "xgb_directional_accuracy_pct": direction(R["xgb_raw"], ar),
        "momentum_directional_accuracy_pct": direction(R["momentum"], ar),
        "actual_up_pct": float((ar > 0).mean() * 100),
        "xgb_deviation_corr": corr, "pred_dev_std_pct": float(dev.std()), "actual_std_pct": float(ar.std()),
        "actual_mean_pct": float(ar.mean()), "xgb_beats_naive_pct_of_origins": float((np.abs(ap-P["xgb"]) < np.abs(ap-P["naive"])).mean()*100),
        "xgb_price_mae_improvement_vs_naive_pct": (pm["naive"]-pm["xgb"])/pm["naive"]*100 if pm["naive"] else np.nan,
        "xgb_price_mae_improvement_vs_momentum_pct": (pm["momentum"]-pm["xgb"])/pm["momentum"]*100 if pm["momentum"] else np.nan,
    }


def main(args):
    global OUT
    if args.out:
        OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_data(args.data_csv, args.refresh)
    features = build_features(df)
    cols = list(features.columns)
    close = df["Close"]
    last_date = df.index[-1]
    feat_ok = features.notna().all(axis=1)
    metrics, summaries, all_predictions = [], [], []

    print(f"[2/4] Final holdout: last {VALIDATION_DAYS} eligible target dates; retrain every {RETRAIN_EVERY} origins", flush=True)
    for h in HORIZONS:
        print(f"\n[3/4] Horizon {h} trading day(s)", flush=True)
        table, valid = make_table(df, features, h)
        eligible = table.loc[valid].sort_values("target_date")
        test_targets = eligible["target_date"].drop_duplicates().sort_values().tail(VALIDATION_DAYS)
        if len(test_targets) < 100:
            raise RuntimeError(f"Too few eligible target dates for {h}D: {len(test_targets)}")
        test_start = pd.Timestamp(test_targets.iloc[0])
        chosen_params, search_results = tune_params(table, valid, cols, h, test_start)
        search_results.to_csv(OUT / f"hyperparameter_search_{h}d.csv", index=False)
        with open(OUT / f"selected_params_{h}d.json", "w", encoding="utf-8") as f:
            json.dump({"horizon": h, "test_start": test_start.date().isoformat(), "selected_params": chosen_params,
                       "candidate_results_file": f"hyperparameter_search_{h}d.csv"}, f, indent=2)
        print(f"      selected parameters: {chosen_params}; test starts {test_start.date()}", flush=True)

        # Final test forecasts: every target date in the test set is evaluated once; training targets
        # must be fully realised before each forecast origin. The test period is never used for tuning.
        origins = list(table.index[valid & table["target_date"].isin(set(test_targets))])
        records, model, centre, beta = [], None, 0.0, 0.0
        last_fit_i, last_fit_origin, last_train_td = None, None, None
        for i, origin in enumerate(origins):
            if model is None or i - last_fit_i >= RETRAIN_EVERY:
                train = table.loc[valid & (table["target_date"] < origin)]
                # Parameter selection is pre-test; model refits use expanding training data only.
                if len(train) < MIN_TRAIN_ROWS:
                    continue
                model, centre, beta = fit_with_shrinkage(train, cols, h, chosen_params)
                last_fit_i, last_fit_origin = i, origin
                last_train_td = pd.Timestamp(train["target_date"].max())
                if not last_train_td < origin:
                    raise RuntimeError(f"Leakage guard failed: {last_train_td} >= {origin}")
            dev = float(model.predict(table.loc[[origin], cols])[0])
            pred_ret, raw_ret = beta * dev, centre + dev
            o_px = float(close.loc[origin]); a_px = float(table.loc[origin, "target_close"])
            a_ret = float(table.loc[origin, "target_return"]); momentum = float(table.loc[origin, "momentum_return"])
            records.append({
                "forecast_origin_date": origin.date().isoformat(), "target_date": pd.Timestamp(table.loc[origin,"target_date"]).date().isoformat(),
                "horizon_trading_days": h, "origin_close": o_px, "actual_close": a_px, "actual_return_pct": a_ret*100,
                "xgb_predicted_return_pct": pred_ret*100, "xgb_predicted_close": o_px*(1+pred_ret),
                "xgb_raw_predicted_return_pct": raw_ret*100, "xgb_raw_predicted_close": o_px*(1+raw_ret),
                "xgb_deviation_pct": dev*100, "naive_predicted_return_pct": 0.0, "naive_predicted_close": o_px,
                "drift_predicted_return_pct": centre*100, "drift_predicted_close": o_px*(1+centre),
                "momentum_predicted_return_pct": momentum*100, "momentum_predicted_close": o_px*(1+momentum),
                "shrinkage_beta": beta, "model_last_fit_origin": last_fit_origin.date().isoformat(),
                "training_latest_target_date": last_train_td.date().isoformat(),
            })
            if (i+1) % 50 == 0: print(f"      Processed {i+1}/{len(origins)} test origins", flush=True)
        pred = pd.DataFrame(records)
        if pred.empty: raise RuntimeError(f"No test predictions generated for {h}D")
        pred.to_csv(OUT / f"predictions_{h}d.csv", index=False)
        all_predictions.append(pred)
        m = metrics_for(pred); m.update({"ticker": TICKER, "horizon_trading_days": h,
                                         "selected_params": json.dumps(chosen_params, sort_keys=True)})
        metrics.append(m)

        # Fit deployable model on all currently realised targets, separately from the test evaluation.
        final_train = table.loc[valid & (table["target_date"] <= last_date)]
        f_model, f_centre, f_beta = fit_with_shrinkage(final_train, cols, h, chosen_params)
        f_model.save_model(str(OUT / f"xgboost_{h}d.ubj"))
        with open(OUT / f"xgboost_{h}d.json", "w", encoding="utf-8") as f:
            json.dump({"ticker": TICKER, "horizon_trading_days": h, "target": "future close-to-close return",
                       "objective": OBJECTIVE, "feature_columns": cols, "parameters": f_model.get_params(),
                       "selected_parameters": chosen_params, "centre_median_return": f_centre, "shrinkage_beta": f_beta,
                       "retrain_every_origins": RETRAIN_EVERY,
                       "validation_rule": "chronological pre-test tuning; test targets are last eligible target dates; training target date strictly before forecast origin",
                       "calibration_note": "Shrinkage correlation threshold is a heuristic, not a formal significance test."}, f, indent=2, default=str)
        latest_dev = float(f_model.predict(features.iloc[[-1]][cols])[0]); current = float(close.iloc[-1])
        summaries.append({"ticker": TICKER, "as_of_date": last_date.date().isoformat(), "current_adjusted_close": current,
                          "horizon_trading_days": h, "shrinkage_beta": f_beta,
                          "xgb_predicted_return_pct": f_beta*latest_dev*100,
                          "xgb_predicted_close": current*(1+f_beta*latest_dev),
                          "xgb_raw_predicted_return_pct": (f_centre+latest_dev)*100,
                          "naive_predicted_close": current,
                          "momentum_trailing_return_pct": float(close.iloc[-1]/close.iloc[-1-h]-1)*100,
                          "warning": "Research estimate only; not a trading recommendation."})

        plot = pred.sort_values("target_date"); x = pd.to_datetime(plot["target_date"])
        plt.figure(figsize=(11,5)); plt.plot(x,plot["actual_close"],label="Actual close")
        plt.plot(x,plot["xgb_predicted_close"],label="XGBoost (shrunk)")
        plt.plot(x,plot["naive_predicted_close"],label="Naive baseline",alpha=.8)
        plt.plot(x,plot["momentum_predicted_close"],label="Momentum benchmark",alpha=.5)
        plt.title(f"{TICKER}: {h}-trading-day test forecasts vs benchmarks (XGBoost v3)")
        plt.xlabel("Target date"); plt.ylabel("Adjusted closing price (INR)"); plt.legend(); plt.grid(True,alpha=.25)
        plt.tight_layout(); plt.savefig(OUT/f"forecast_{h}d.png",dpi=160); plt.close()
        print(f"      test N={len(pred)} | XGB MAE ₹{m['xgb_price_mae']:.2f} | naive ₹{m['naive_price_mae']:.2f} | momentum ₹{m['momentum_price_mae']:.2f}", flush=True)

    md = pd.DataFrame(metrics); md.to_csv(OUT/"metrics.csv",index=False)
    pd.DataFrame(summaries).to_csv(OUT/"forecast_summary.csv",index=False)
    # Combined predictions make cross-horizon review easier.
    pd.concat(all_predictions,ignore_index=True).to_csv(OUT/"all_test_predictions.csv",index=False)
    with open(OUT/"run_metadata.json","w",encoding="utf-8") as f:
        json.dump({"ticker":TICKER,"start":START,"last_data_date":last_date.date().isoformat(),
                   "horizons":HORIZONS,"validation_days":VALIDATION_DAYS,"retrain_every":RETRAIN_EVERY,
                   "objective":OBJECTIVE,"feature_columns":cols,"seed":SEED,
                   "python":sys.version,"pandas":pd.__version__,"numpy":np.__version__},f,indent=2)
    print("\n[4/4] COMPLETE",flush=True); print(f"Outputs saved to {OUT}",flush=True)
    print(md[["horizon_trading_days","test_predictions","mean_beta","xgb_price_mae","naive_price_mae","momentum_price_mae","xgb_directional_accuracy_pct","actual_up_pct"]].round(3).to_string(index=False),flush=True)
    print("Interpretation: lower XGB MAE than the baselines is evidence to investigate, not proof of a profitable strategy. Overlapping multi-day targets are dependent.",flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-csv", help="OHLCV CSV to use/cache (default: ../data/TCS.NS_daily.csv)")
    ap.add_argument("--refresh", action="store_true", help="re-download data even if cache exists")
    ap.add_argument("--objective", help="override XGBoost objective, e.g. reg:squarederror")
    ap.add_argument("--out", help="output directory")
    a = ap.parse_args()
    if a.objective: OBJECTIVE = a.objective
    try: main(a)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr, flush=True)
        raise
