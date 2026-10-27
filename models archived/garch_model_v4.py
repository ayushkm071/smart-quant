"""
Smart-Quant — GARCH v4: dependence-aware forecast comparison + parameter significance

This script:
1. Downloads TCS.NS adjusted prices from 2010 onward.
2. Fits GARCH(1,1)-Student-t and GJR-GARCH(1,1)-Student-t on the same walk-forward
   origins as the prior v3 experiment.
3. Compares both against a rolling 252-return historical-volatility benchmark.
4. Reports MAE, RMSE and QLIKE.
5. Performs Diebold-Mariano-style tests on paired loss differentials, using a
   Newey-West/HAC long-run variance estimator. The HAC lag is at least horizon-1
   to account for overlapping h-day realized-volatility targets.
6. Saves fitted parameter standard errors and p-values at selected diagnostic
   origins, including the GJR asymmetry parameter gamma.

Important:
- Tests are run separately by horizon and loss function.
- This is exploratory multiple testing across horizons/model pairs/metrics.
  Interpret p-values cautiously; adjust for multiplicity before strong claims.
- The DM-style test is an asymptotic comparison of paired losses. With only about
  252 overlapping targets per horizon, inference can be noisy.
- QLIKE uses the conventional variance loss log(forecast variance) +
  realized variance / forecast variance; the realized-only term is omitted
  because it cancels in paired comparisons.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.sandwich_covariance import cov_hac

warnings.filterwarnings("ignore", category=FutureWarning)

TICKER = "TCS.NS"
START_DATE = "2010-01-01"
HORIZONS = [1, 5, 21, 30, 45]
TEST_TARGETS = 252
ROLLING_WINDOW = 252
REFIT_EVERY = 5
MIN_TRAIN_RETURNS = 750

PROJECT_ROOT = Path(r"C:\Quant project folder")
OUTPUT_DIR = PROJECT_ROOT / "results" / "garch_v4"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def download_prices() -> pd.Series:
    data = yf.download(
        TICKER, start=START_DATE, auto_adjust=True,
        progress=False, threads=False
    )
    if data.empty:
        raise RuntimeError("No price data returned. Check internet access/ticker.")
    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close[TICKER] if TICKER in close.columns else close.iloc[:, 0]
    close = pd.to_numeric(close, errors="coerce").dropna()
    close = close[~close.index.duplicated(keep="last")].sort_index()
    if len(close) < MIN_TRAIN_RETURNS + TEST_TARGETS + max(HORIZONS):
        raise RuntimeError(f"Insufficient data: {len(close)} price rows.")
    return close.rename("Close")


def fit_model(train: pd.Series, spec: str):
    if spec == "GARCH(1,1)-Student-t":
        model = arch_model(
            train, mean="Constant", vol="GARCH",
            p=1, o=0, q=1, dist="StudentsT", rescale=False
        )
    elif spec == "GJR-GARCH(1,1)-Student-t":
        model = arch_model(
            train, mean="Constant", vol="GARCH",
            p=1, o=1, q=1, dist="StudentsT", rescale=False
        )
    else:
        raise ValueError(f"Unknown model spec: {spec}")
    return model.fit(disp="off", show_warning=False)


def qlike(actual_var: np.ndarray, forecast_var: np.ndarray) -> np.ndarray:
    eps = 1e-12
    actual = np.maximum(np.asarray(actual_var, dtype=float), eps)
    forecast = np.maximum(np.asarray(forecast_var, dtype=float), eps)
    return np.log(forecast) + actual / forecast


def newey_west_mean_test(loss_difference: np.ndarray, max_lag: int):
    """
    HAC test of mean loss differential = 0.
    Uses Bartlett/Newey-West weights and finite-sample correction.
    Positive mean means first model has higher loss (worse) than second.
    """
    d = np.asarray(loss_difference, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 20:
        return {
            "n": n, "mean_loss_difference": np.nan, "hac_se": np.nan,
            "t_stat": np.nan, "p_value_two_sided": np.nan, "hac_lag": max_lag
        }

    d_mean = float(np.mean(d))
    centered = d - d_mean
    lag = min(max_lag, n - 1)
    # Long-run variance of the sample mean: gamma0 + 2 sum w_k gamma_k, divided by n.
    gamma0 = float(np.dot(centered, centered) / n)
    long_run_var = gamma0
    for k in range(1, lag + 1):
        gamma_k = float(np.dot(centered[k:], centered[:-k]) / n)
        weight = 1.0 - k / (lag + 1.0)
        long_run_var += 2.0 * weight * gamma_k

    variance_of_mean = max(long_run_var, 0.0) / n
    hac_se = float(np.sqrt(variance_of_mean))
    if hac_se <= 0 or not np.isfinite(hac_se):
        t_stat = np.nan
        p_value = np.nan
    else:
        t_stat = d_mean / hac_se
        # Normal approximation for two-sided p-value; stated as asymptotic.
        from scipy.stats import norm
        p_value = float(2.0 * norm.sf(abs(t_stat)))

    return {
        "n": n,
        "mean_loss_difference": d_mean,
        "hac_se": hac_se,
        "t_stat": t_stat,
        "p_value_two_sided": p_value,
        "hac_lag": lag,
    }


def main() -> None:
    print("=" * 78)
    print("SMART-QUANT | GARCH v4: FORMAL FORECAST COMPARISON")
    print("=" * 78)

    close = download_prices()
    returns = (100.0 * np.log(close / close.shift(1))).dropna()
    returns.name = "return_pct"
    close.to_frame().to_csv(OUTPUT_DIR / "adjusted_close.csv")
    returns.to_frame().to_csv(OUTPUT_DIR / "daily_log_returns_pct.csv")

    first_target = len(returns) - TEST_TARGETS
    target_indices = np.arange(first_target, len(returns))
    specs = ["GARCH(1,1)-Student-t", "GJR-GARCH(1,1)-Student-t"]
    forecast_rows = []
    param_rows = []
    residual_rows = []

    print(f"Ticker: {TICKER}")
    print(f"Returns: {len(returns)}")
    print(f"Test target dates: {returns.index[first_target].date()} to {returns.index[-1].date()}")
    print(f"Horizons: {HORIZONS}; refit every {REFIT_EVERY} origins")

    for horizon in HORIZONS:
        print(f"\nHorizon {horizon}")
        origins = target_indices - horizon
        states = {
            spec: {"result": None, "last_fit_origin": None}
            for spec in specs
        }

        for count, origin in enumerate(origins):
            for spec in specs:
                state = states[spec]
                should_refit = (
                    state["result"] is None
                    or state["last_fit_origin"] is None
                    or origin - state["last_fit_origin"] >= REFIT_EVERY
                )
                if should_refit:
                    train = returns.iloc[:origin + 1]
                    try:
                        result = fit_model(train, spec)
                        state["result"] = result
                        state["last_fit_origin"] = int(origin)

                        # Store parameter inference and residual diagnostics on
                        # the first and final refit for each horizon/model.
                        if count == 0 or count >= len(origins) - REFIT_EVERY:
                            params = result.params
                            ses = result.std_err
                            pvals = result.pvalues
                            row = {
                                "horizon": horizon,
                                "model": spec,
                                "origin_date": returns.index[origin],
                                "log_likelihood": float(result.loglikelihood),
                                "AIC": float(result.aic),
                                "BIC": float(result.bic),
                            }
                            for name, value in params.items():
                                row[f"estimate_{name}"] = float(value)
                                row[f"std_error_{name}"] = float(ses.get(name, np.nan))
                                row[f"p_value_{name}"] = float(pvals.get(name, np.nan))
                            alpha = float(params.get("alpha[1]", np.nan))
                            beta = float(params.get("beta[1]", np.nan))
                            gamma = float(params.get("gamma[1]", 0.0))
                            row["approx_persistence"] = alpha + beta + 0.5 * gamma
                            param_rows.append(row)

                            resid = pd.Series(result.std_resid).dropna()
                            lb_r = acorr_ljungbox(resid, lags=[10], return_df=True)
                            lb_sq = acorr_ljungbox(resid ** 2, lags=[10], return_df=True)
                            try:
                                arch_stat, arch_p, _, _ = het_arch(resid, nlags=10)
                            except Exception:
                                arch_stat, arch_p = np.nan, np.nan
                            residual_rows.append({
                                "horizon": horizon,
                                "model": spec,
                                "origin_date": returns.index[origin],
                                "Ljung_Box_resid_lag10_p": float(lb_r.loc[10, "lb_pvalue"]),
                                "Ljung_Box_squared_resid_lag10_p": float(lb_sq.loc[10, "lb_pvalue"]),
                                "ARCH_LM_lag10_p": float(arch_p),
                            })
                    except Exception as exc:
                        print(f"  Fit failed ({spec}) at {returns.index[origin].date()}: {exc}")
                        state["result"] = None
                        continue

                result = state["result"]
                if result is None:
                    continue
                try:
                    # Keep the latest fitted parameters, but update the
                    # conditional volatility state using every observation
                    # available through this forecast origin. This avoids
                    # forecasting from a stale refit-origin state.
                    current_history = returns.iloc[:origin + 1]
                    if spec == "GARCH(1,1)-Student-t":
                        current_model = arch_model(
                            current_history, mean="Constant", vol="GARCH",
                            p=1, o=0, q=1, dist="StudentsT", rescale=False
                        )
                    else:
                        current_model = arch_model(
                            current_history, mean="Constant", vol="GARCH",
                            p=1, o=1, q=1, dist="StudentsT", rescale=False
                        )
                    fc = current_model.forecast(
                        params=result.params, horizon=horizon, reindex=False
                    )
                    daily_vars = fc.variance.iloc[-1].to_numpy(dtype=float)
                    fvar = float(np.sum(daily_vars[:horizon]))
                    if not np.isfinite(fvar) or fvar <= 0:
                        continue
                except Exception as exc:
                    print(f"  Forecast failed ({spec}) at {returns.index[origin].date()}: {exc}")
                    continue

                future = returns.iloc[origin + 1:origin + horizon + 1]
                if len(future) != horizon:
                    continue
                avar = float(np.sum(np.square(future.to_numpy(dtype=float))))
                forecast_rows.append({
                    "model": spec,
                    "horizon": horizon,
                    "origin_date": returns.index[origin],
                    "target_date": returns.index[origin + horizon],
                    "forecast_variance": fvar,
                    "forecast_volatility": float(np.sqrt(fvar)),
                    "realized_variance": avar,
                    "realized_volatility": float(np.sqrt(max(avar, 0.0))),
                })

            # Add the benchmark for the same origin.
            hist = returns.iloc[max(0, origin - ROLLING_WINDOW + 1):origin + 1].to_numpy(dtype=float)
            hist_var = float(np.mean(hist ** 2) * horizon)
            future = returns.iloc[origin + 1:origin + horizon + 1]
            if len(future) == horizon:
                avar = float(np.sum(np.square(future.to_numpy(dtype=float))))
                forecast_rows.append({
                    "model": "Rolling historical volatility",
                    "horizon": horizon,
                    "origin_date": returns.index[origin],
                    "target_date": returns.index[origin + horizon],
                    "forecast_variance": hist_var,
                    "forecast_volatility": float(np.sqrt(max(hist_var, 0.0))),
                    "realized_variance": avar,
                    "realized_volatility": float(np.sqrt(max(avar, 0.0))),
                })

            if (count + 1) % 50 == 0:
                print(f"  Processed {count + 1}/{len(origins)} origins")

    predictions = pd.DataFrame(forecast_rows)
    if predictions.empty:
        raise RuntimeError("No forecast rows generated.")
    predictions.to_csv(OUTPUT_DIR / "predictions.csv", index=False)
    pd.DataFrame(param_rows).to_csv(OUTPUT_DIR / "parameter_inference.csv", index=False)
    pd.DataFrame(residual_rows).to_csv(OUTPUT_DIR / "residual_diagnostics.csv", index=False)

    # Per-row loss values.
    predictions["absolute_error"] = np.abs(
        predictions["forecast_volatility"] - predictions["realized_volatility"]
    )
    predictions["squared_error"] = (
        predictions["forecast_volatility"] - predictions["realized_volatility"]
    ) ** 2
    predictions["QLIKE"] = qlike(
        predictions["realized_variance"].to_numpy(),
        predictions["forecast_variance"].to_numpy(),
    )
    predictions.to_csv(OUTPUT_DIR / "predictions.csv", index=False)

    metrics_rows = []
    for (h, model_name), g in predictions.groupby(["horizon", "model"]):
        metrics_rows.append({
            "horizon": int(h),
            "model": model_name,
            "n_predictions": int(len(g)),
            "MAE": float(g["absolute_error"].mean()),
            "RMSE": float(np.sqrt(g["squared_error"].mean())),
            "mean_QLIKE": float(g["QLIKE"].mean()),
        })
    metrics = pd.DataFrame(metrics_rows).sort_values(["horizon", "model"])
    metrics.to_csv(OUTPUT_DIR / "metrics.csv", index=False)

    # Pairwise tests use only common origin dates and same horizon.
    model_pairs = [
        ("GARCH(1,1)-Student-t", "Rolling historical volatility"),
        ("GJR-GARCH(1,1)-Student-t", "Rolling historical volatility"),
        ("GJR-GARCH(1,1)-Student-t", "GARCH(1,1)-Student-t"),
    ]
    dm_rows = []
    for horizon in HORIZONS:
        subset = predictions[predictions["horizon"] == horizon]
        for model_a, model_b in model_pairs:
            a = subset[subset["model"] == model_a].copy()
            b = subset[subset["model"] == model_b].copy()
            merged = a.merge(
                b, on=["horizon", "origin_date", "target_date"],
                suffixes=("_a", "_b"), how="inner"
            )
            if merged.empty:
                continue

            loss_specs = [
                ("absolute_error", "MAE loss"),
                ("squared_error", "Squared error loss"),
                ("QLIKE", "QLIKE loss"),
            ]
            for col, loss_label in loss_specs:
                diff = (
                    merged[f"{col}_a"].to_numpy(dtype=float)
                    - merged[f"{col}_b"].to_numpy(dtype=float)
                )
                test = newey_west_mean_test(diff, max_lag=horizon - 1)
                dm_rows.append({
                    "horizon": horizon,
                    "model_a": model_a,
                    "model_b": model_b,
                    "loss": loss_label,
                    "interpretation": "positive mean difference means model_a has higher loss",
                    **test,
                })

    dm = pd.DataFrame(dm_rows)
    dm.to_csv(OUTPUT_DIR / "forecast_comparison_tests.csv", index=False)
    print("\nMETRICS")
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nPAIRED HAC FORECAST COMPARISON TESTS")
    print(dm.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    # Holm adjustment within each loss family across all pair/horizon tests.
    if not dm.empty:
        from statsmodels.stats.multitest import multipletests
        dm["holm_adjusted_p_value"] = np.nan
        for loss_label, idx in dm.groupby("loss").groups.items():
            p = dm.loc[idx, "p_value_two_sided"].to_numpy(dtype=float)
            valid = np.isfinite(p)
            if valid.any():
                adjusted = np.full(len(p), np.nan)
                adjusted[valid] = multipletests(p[valid], method="holm")[1]
                dm.loc[idx, "holm_adjusted_p_value"] = adjusted
        dm.to_csv(OUTPUT_DIR / "forecast_comparison_tests.csv", index=False)

    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("Read p-values cautiously: only ~252 overlapping targets per horizon.")
    print("Inspect Holm-adjusted p-values and effect sizes, not raw p-values alone.")


if __name__ == "__main__":
    main()
