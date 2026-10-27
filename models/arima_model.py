"""ARIMA v2: horizon-specific rolling-origin selection and walk-forward evaluation.

Forecasts adjusted close for TCS.NS at 1, 5, 21, 30 and 45 trading days.
The final 252 target dates are held out from order selection. Each horizon gets
its own ARIMA(p,d,q), selected only on earlier rolling-origin validation data.
"""
from __future__ import annotations

import itertools
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ------------------------------- Configuration ------------------------------
TICKER = "TCS.NS"
START_DATE = "2010-01-01"
HORIZONS = [1, 5, 21, 30, 45]
VALIDATION_SIZE = 252
TEST_TARGETS = 252
REFIT_EVERY = 5
P_VALUES = range(0, 4)
D_VALUES = range(0, 2)
Q_VALUES = range(0, 4)
VALIDATION_ORIGIN_COUNT = 12  # spaced chronologically across validation window
MAXITER = 100

# Keeps output at C:\Quant project folder\results\arima_v2 when this script is
# saved in C:\Quant project folder\models\ (the expected project layout).
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name.lower() in {"models", "scripts"} else SCRIPT_DIR
RESULTS_DIR = PROJECT_DIR / "results" / "arima_v2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def rmse(actual, predicted):
    return float(np.sqrt(mean_squared_error(actual, predicted)))


def fit_arima(values, order):
    """Fit without globally suppressing warnings; caller records warnings."""
    model = ARIMA(np.asarray(values, dtype=float), order=order,
                  enforce_stationarity=False, enforce_invertibility=False)
    return model.fit(method_kwargs={"maxiter": MAXITER})


def holm_adjust(p_values):
    vals = np.asarray(p_values, dtype=float)
    out = np.full(vals.shape, np.nan, dtype=float)
    valid = np.isfinite(vals)
    if valid.any():
        out[valid] = multipletests(vals[valid], method="holm")[1]
    return out


print(f"Downloading {TICKER} from {START_DATE} onward...", flush=True)
data = yf.download(TICKER, start=START_DATE, interval="1d", auto_adjust=True, progress=False)
if data.empty:
    raise RuntimeError("Yahoo Finance returned no data. Check ticker and internet connection.")
if isinstance(data.columns, pd.MultiIndex):
    candidates = [c for c in data.columns if "Close" in c]
    if not candidates:
        raise KeyError("Could not locate Close in yfinance MultiIndex columns.")
    close = data[candidates[0]]
else:
    close = data["Close"]
close = pd.to_numeric(close, errors="coerce").dropna()
close = close[~close.index.duplicated(keep="last")].sort_index().astype(float)
close.name = "adjusted_close"

min_required = VALIDATION_SIZE + TEST_TARGETS + max(HORIZONS) + 500
if len(close) < min_required:
    raise RuntimeError(f"Not enough observations ({len(close)}); need approximately {min_required}.")
close.to_frame().to_csv(RESULTS_DIR / "TCS_adjusted_close_2010_onward.csv")
print(f"Observations: {len(close)} | {close.index.min().date()} to {close.index.max().date()}")

# Hold out final TEST_TARGETS target dates. Keep all selection origins and
# their forecast endpoints strictly before the earliest test origin.
test_start = len(close) - TEST_TARGETS
selection_end = test_start - max(HORIZONS)
selection_start = selection_end - VALIDATION_SIZE
if selection_start <= 500:
    raise RuntimeError("Insufficient pre-validation training history.")

orders = list(itertools.product(P_VALUES, D_VALUES, Q_VALUES))
selected_orders = {}
grid_rows = []
validation_rows = []

# -------------------------- 1. Horizon-wise selection ------------------------
for h in HORIZONS:
    print(f"\nSelecting order for horizon h={h}...", flush=True)
    # Origins distributed through the validation window; each forecast is made
    # using data available at that origin and evaluated at exactly h steps.
    lo = selection_start
    hi = selection_end - h
    origins = np.unique(np.linspace(lo, hi, VALIDATION_ORIGIN_COUNT, dtype=int))
    candidate_scores = []
    for order in orders:
        errors, abs_errors, aics = [], [], []
        failed = 0
        for origin in origins:
            train = close.iloc[:origin + 1].to_numpy()
            actual = float(close.iloc[origin + h])
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    fitted = fit_arima(train, order)
                pred = float(np.asarray(fitted.forecast(steps=h)).reshape(-1)[h - 1])
                if not np.isfinite(pred):
                    raise ValueError("Non-finite forecast")
                errors.append(pred - actual)
                abs_errors.append(abs(pred - actual))
                aics.append(float(fitted.aic))
                validation_rows.append({"horizon": h, "origin_date": close.index[origin],
                                        "target_date": close.index[origin+h], "p": order[0],
                                        "d": order[1], "q": order[2], "forecast": pred,
                                        "actual": actual, "warning_count": len(caught)})
            except Exception as exc:
                failed += 1
                print(f"  ARIMA{order} validation failure at {close.index[origin].date()}: {exc}", flush=True)
        score = rmse(np.zeros(len(errors)), errors) if errors else np.inf
        row = {"horizon": h, "p": order[0], "d": order[1], "q": order[2],
               "validation_origins_successful": len(errors), "validation_failures": failed,
               "validation_RMSE": score,
               "validation_MAE": float(np.mean(abs_errors)) if abs_errors else np.inf,
               "mean_AIC": float(np.mean(aics)) if aics else np.nan}
        grid_rows.append(row)
        if len(errors) >= max(5, int(0.7 * len(origins))):
            candidate_scores.append(row)
    if not candidate_scores:
        raise RuntimeError(f"All ARIMA candidates failed for horizon {h}.")
    # RMSE is the primary selection metric; tie-break by MAE, then simpler order.
    winner = min(candidate_scores, key=lambda r: (r["validation_RMSE"], r["validation_MAE"],
                                                   r["p"] + r["q"], r["d"]))
    selected_orders[h] = (int(winner["p"]), int(winner["d"]), int(winner["q"]))
    print(f"  Selected ARIMA{selected_orders[h]} | validation RMSE={winner['validation_RMSE']:.3f}", flush=True)

pd.DataFrame(grid_rows).sort_values(["horizon", "validation_RMSE"]).to_csv(
    RESULTS_DIR / "order_selection_grid_by_horizon.csv", index=False)
pd.DataFrame(validation_rows).to_csv(RESULTS_DIR / "validation_predictions.csv", index=False)
pd.DataFrame([{"horizon": h, "selected_order": str(selected_orders[h])} for h in HORIZONS]).to_csv(
    RESULTS_DIR / "selected_orders.csv", index=False)

# -------------------------- 2. Walk-forward test -----------------------------
metric_rows, prediction_rows, diagnostic_rows = [], [], []
for h in HORIZONS:
    order = selected_orders[h]
    print(f"\nTesting horizon h={h}, ARIMA{order}...", flush=True)
    target_indices = np.arange(test_start, len(close))
    origins = target_indices - h
    fitted, last_fit_origin = None, None
    last_fit_resid = None
    for j, origin in enumerate(origins):
        history = close.iloc[:origin + 1]
        current_price = float(close.iloc[origin])
        target_idx = int(origin + h)
        actual = float(close.iloc[target_idx])
        refit = fitted is None or (j % REFIT_EVERY == 0) or last_fit_origin is None
        warning_messages = []
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                if refit:
                    fitted = fit_arima(history.to_numpy(), order)
                    last_fit_origin = int(origin)
                    last_fit_resid = np.asarray(fitted.resid, dtype=float)
                else:
                    # Previous model contains data through previous origin;
                    # append only the newly observed price at this origin.
                    fitted = fitted.append(np.asarray([current_price]), refit=False)
                warning_messages = [str(w.message) for w in caught]
            path = np.asarray(fitted.forecast(steps=h), dtype=float).reshape(-1)
            arima_price = float(path[h - 1])
            if not np.isfinite(arima_price):
                raise ValueError("Non-finite ARIMA forecast")
            prior = float(close.iloc[origin - h]) if origin >= h else current_price
            momentum_price = current_price * (current_price / prior) if prior else current_price
            prediction_rows.append({"horizon": h, "origin_date": close.index[origin],
                                    "target_date": close.index[target_idx], "origin_price": current_price,
                                    "actual_price": actual, "arima_price": arima_price,
                                    "naive_price": current_price, "momentum_price": momentum_price,
                                    "order": str(order), "refit": refit,
                                    "warning_count": len(warning_messages),
                                    "warnings": " | ".join(warning_messages)})
        except Exception as exc:
            print(f"  TEST forecast failed at origin {close.index[origin].date()}: {exc}", flush=True)
            prediction_rows.append({"horizon": h, "origin_date": close.index[origin],
                                    "target_date": close.index[target_idx], "origin_price": current_price,
                                    "actual_price": actual, "arima_price": np.nan,
                                    "naive_price": current_price, "momentum_price": np.nan,
                                    "order": str(order), "refit": refit, "warning_count": len(warning_messages),
                                    "warnings": f"FAILED: {exc}"})

pred_df = pd.DataFrame(prediction_rows)
# Metrics are computed on identical valid target dates across all three forecasts.
for h in HORIZONS:
    part = pred_df[pred_df.horizon == h].copy()
    valid = part.dropna(subset=["arima_price", "naive_price", "momentum_price", "actual_price"]).copy()
    if valid.empty:
        print(f"No valid predictions for horizon {h}; skipping summary.")
        continue
    actual = valid.actual_price.to_numpy()
    row = {"horizon": h, "selected_order": str(selected_orders[h]),
           "n_test_targets_requested": len(part), "n_test_targets_valid": len(valid),
           "n_failed_arima_forecasts": int(part.arima_price.isna().sum()),
           "ARIMA_MAE": mean_absolute_error(actual, valid.arima_price),
           "Naive_MAE": mean_absolute_error(actual, valid.naive_price),
           "Momentum_MAE": mean_absolute_error(actual, valid.momentum_price),
           "ARIMA_RMSE": rmse(actual, valid.arima_price),
           "Naive_RMSE": rmse(actual, valid.naive_price),
           "Momentum_RMSE": rmse(actual, valid.momentum_price),
           "ARIMA_MAE_vs_Naive_pct": (mean_absolute_error(actual, valid.arima_price) /
                                      mean_absolute_error(actual, valid.naive_price) - 1) * 100,
           "ARIMA_directional_accuracy_pct": float(np.mean(np.sign(valid.arima_price-valid.origin_price) ==
                                                            np.sign(valid.actual_price-valid.origin_price)) * 100)}
    metric_rows.append(row)

    # Residual diagnostics on first fitted model's in-sample residuals for this horizon.
    # These are diagnostics, not proof that forecasts are unbiased.
    try:
        residuals = np.asarray(last_fit_resid, dtype=float)
        residuals = residuals[np.isfinite(residuals)]
        lb_lag = min(20, max(1, len(residuals) // 5))
        lb = acorr_ljungbox(residuals, lags=[lb_lag], return_df=True)
        acf_values = acf(residuals, nlags=min(40, max(1, len(residuals)//4)), fft=True)
        diagnostic_rows.append({"horizon": h, "selected_order": str(order),
                                "residual_count": len(residuals), "ljung_box_lag": lb_lag,
                                "ljung_box_stat": float(lb["lb_stat"].iloc[0]),
                                "ljung_box_pvalue": float(lb["lb_pvalue"].iloc[0]),
                                "residual_acf": ";".join(f"lag{i}:{v:.6g}" for i, v in enumerate(acf_values))})
    except Exception as exc:
        diagnostic_rows.append({"horizon": h, "selected_order": str(order), "diagnostic_error": str(exc)})

# ----------------------- 3. HAC paired comparisons ---------------------------
# For each horizon, test mean loss differential using Newey-West/HAC covariance.
# Overlapping h-step forecasts induce serial dependence; use maxlags=h-1.
comparison_rows = []
for h in HORIZONS:
    part = pred_df[pred_df.horizon == h].dropna(subset=["arima_price", "naive_price", "momentum_price", "actual_price"])
    if len(part) < 10:
        continue
    y = part.actual_price.to_numpy()
    comparisons = [("ARIMA_vs_Naive", part.arima_price.to_numpy(), part.naive_price.to_numpy()),
                   ("ARIMA_vs_Momentum", part.arima_price.to_numpy(), part.momentum_price.to_numpy())]
    for name, p1, p2 in comparisons:
        # Positive mean differential means ARIMA has larger (worse) absolute error.
        diff = np.abs(y - p1) - np.abs(y - p2)
        try:
            fit = sm.OLS(diff, np.ones((len(diff), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": min(h-1, len(diff)-1)})
            comparison_rows.append({"horizon": h, "comparison": name, "n": len(diff),
                                    "mean_abs_error_difference_ARIMA_minus_baseline": float(np.mean(diff)),
                                    "hac_t_stat": float(fit.tvalues[0]), "p_value": float(fit.pvalues[0])})
        except Exception as exc:
            comparison_rows.append({"horizon": h, "comparison": name, "n": len(diff), "error": str(exc)})
if comparison_rows:
    comp_df = pd.DataFrame(comparison_rows)
    if "p_value" in comp_df:
        comp_df["holm_adjusted_p_value"] = np.nan
        valid = comp_df["p_value"].notna()
        if valid.any():
            comp_df.loc[valid, "holm_adjusted_p_value"] = holm_adjust(comp_df.loc[valid, "p_value"].to_numpy())
    comp_df.to_csv(RESULTS_DIR / "hac_paired_error_tests.csv", index=False)

metrics_df = pd.DataFrame(metric_rows)
metrics_df.to_csv(RESULTS_DIR / "metrics.csv", index=False)
pred_df.to_csv(RESULTS_DIR / "predictions.csv", index=False)
pd.DataFrame(diagnostic_rows).to_csv(RESULTS_DIR / "residual_diagnostics.csv", index=False)

# ------------------------------ 4. Plots -------------------------------------
for h in HORIZONS:
    part = pred_df[pred_df.horizon == h].copy()
    if part.empty:
        continue
    plt.figure(figsize=(13, 5))
    plt.plot(part.target_date, part.actual_price, label="Actual")
    plt.plot(part.target_date, part.arima_price, label=f"ARIMA{selected_orders[h]}")
    plt.plot(part.target_date, part.naive_price, label="Naive baseline", alpha=0.8)
    plt.title(f"{TICKER}: {h}-trading-day walk-forward forecast")
    plt.xlabel("Target date")
    plt.ylabel("Adjusted close (₹)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"forecast_h{h}.png", dpi=150)
    plt.close()

print("\n===== FINAL TEST METRICS =====")
print(metrics_df.to_string(index=False) if not metrics_df.empty else "No metrics available")
print(f"\nResults directory: {RESULTS_DIR}")
print("Files: metrics.csv, predictions.csv, selected_orders.csv, order_selection_grid_by_horizon.csv,")
print("       validation_predictions.csv, residual_diagnostics.csv, hac_paired_error_tests.csv, forecast_h*.png")
print("Interpret results cautiously: selection is validation-only; test results do not guarantee future performance.")
