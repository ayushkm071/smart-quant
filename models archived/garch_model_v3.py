"""
Smart-Quant — GARCH v3: parameter diagnostics + GJR-GARCH comparison

Compares three volatility specifications using walk-forward forecasts:
  1. GARCH(1,1), Student-t innovations
  2. GJR-GARCH(1,1), Student-t innovations (asymmetric response)
  3. Rolling historical variance benchmark

Ticker: TCS.NS
Training data: from 2010 onward
Test targets: latest 252 eligible target dates
Horizons: 1, 5, 21, 30, 45 trading days
Refit frequency: every 5 forecast origins

Outputs:
  results/garch_v3/metrics.csv
  results/garch_v3/predictions.csv
  results/garch_v3/benchmark_improvement.csv
  results/garch_v3/model_diagnostics.csv
  results/garch_v3/average_forecasts_by_horizon.csv
  diagnostic and comparison plots

Notes:
- Realized volatility proxy = sqrt(sum of future squared daily log returns),
  where returns are in percentage points.
- QLIKE is reported as a raw mean. Lower is better.
- Forecast horizons overlap, so errors are dependent. Results are descriptive;
  use a dependence-aware test before claiming statistical superiority.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

warnings.filterwarnings("ignore", category=FutureWarning)

TICKER = "TCS.NS"
START_DATE = "2010-01-01"
HORIZONS = [1, 5, 21, 30, 45]
TEST_TARGETS = 252
ROLLING_WINDOW = 252
REFIT_EVERY = 5
MIN_TRAIN_RETURNS = 750

PROJECT_ROOT = Path(r"C:\Quant project folder")
OUTPUT_DIR = PROJECT_ROOT / "results" / "garch_v3"
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


def qlike(actual_variance: float, forecast_variance: float) -> float:
    # QLIKE form omitting the actual-only term; lower is better.
    eps = 1e-12
    a = max(float(actual_variance), eps)
    f = max(float(forecast_variance), eps)
    return float(np.log(f) + a / f)


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


def diagnostics_for_fit(result, spec: str, origin_date) -> dict:
    params = result.params
    resid = pd.Series(result.std_resid).dropna()
    lb_resid = acorr_ljungbox(resid, lags=[10], return_df=True)
    lb_squared = acorr_ljungbox(resid ** 2, lags=[10], return_df=True)
    try:
        arch_stat, arch_p, _, _ = het_arch(resid, nlags=10)
    except Exception:
        arch_stat, arch_p = np.nan, np.nan

    row = {
        "model": spec,
        "diagnostic_origin_date": origin_date,
        "log_likelihood": float(result.loglikelihood),
        "AIC": float(result.aic),
        "BIC": float(result.bic),
        "Ljung_Box_resid_lag10_p": float(lb_resid.loc[10, "lb_pvalue"]),
        "Ljung_Box_squared_resid_lag10_p": float(lb_squared.loc[10, "lb_pvalue"]),
        "ARCH_LM_lag10_p": float(arch_p),
    }
    for name, value in params.items():
        row[f"param_{name}"] = float(value)

    # For GARCH(1,1), persistence is alpha+beta. For GJR-GARCH it is
    # alpha + beta + 0.5*gamma as a common approximation under symmetric shocks.
    alpha = float(params.get("alpha[1]", np.nan))
    beta = float(params.get("beta[1]", np.nan))
    gamma = float(params.get("gamma[1]", 0.0))
    row["approx_persistence"] = alpha + beta + 0.5 * gamma
    return row


def main() -> None:
    print("=" * 78)
    print("SMART-QUANT | GARCH v3: GARCH vs GJR-GARCH + DIAGNOSTICS")
    print("=" * 78)

    close = download_prices()
    returns = (100.0 * np.log(close / close.shift(1))).dropna()
    returns.name = "return_pct"
    returns.to_frame().to_csv(OUTPUT_DIR / "daily_log_returns_pct.csv")
    close.to_frame().to_csv(OUTPUT_DIR / "adjusted_close.csv")

    first_target = len(returns) - TEST_TARGETS
    target_indices = np.arange(first_target, len(returns))
    print(f"Ticker: {TICKER}")
    print(f"Return observations: {len(returns):,}")
    print(f"Test target dates: {returns.index[first_target].date()} to {returns.index[-1].date()}")
    print(f"Refit frequency: every {REFIT_EVERY} origins")

    specs = ["GARCH(1,1)-Student-t", "GJR-GARCH(1,1)-Student-t"]
    all_rows = []
    diagnostic_rows = []

    for horizon in HORIZONS:
        print(f"\nHorizon {horizon} day(s)")
        origins = target_indices - horizon
        if origins.min() < MIN_TRAIN_RETURNS - 1:
            raise RuntimeError("Not enough training history.")

        # Maintain separate fitted state per model specification.
        states = {
            spec: {"result": None, "last_fit_origin": None}
            for spec in specs
        }
        failures = {spec: 0 for spec in specs}

        for count, origin in enumerate(origins):
            for spec in specs:
                state = states[spec]
                should_refit = (
                    state["result"] is None
                    or state["last_fit_origin"] is None
                    or origin - state["last_fit_origin"] >= REFIT_EVERY
                )
                if should_refit:
                    try:
                        train = returns.iloc[:origin + 1]
                        result = fit_model(train, spec)
                        state["result"] = result
                        state["last_fit_origin"] = int(origin)
                        # Store diagnostics at the first and final refit per horizon.
                        if count == 0 or count >= len(origins) - REFIT_EVERY:
                            diagnostic_rows.append(
                                diagnostics_for_fit(result, spec, returns.index[origin])
                            )
                    except Exception as exc:
                        failures[spec] += 1
                        state["result"] = None
                        print(
                            f"  {spec} fit failed at {returns.index[origin].date()}: "
                            f"{type(exc).__name__}: {exc}"
                        )
                        continue

                result = state["result"]
                if result is None:
                    continue

                try:
                    forecast = result.forecast(horizon=horizon, reindex=False)
                    daily_vars = forecast.variance.iloc[-1].to_numpy(dtype=float)
                    forecast_variance = float(np.sum(daily_vars[:horizon]))
                    if not np.isfinite(forecast_variance) or forecast_variance <= 0:
                        failures[spec] += 1
                        continue
                except Exception as exc:
                    failures[spec] += 1
                    print(
                        f"  {spec} forecast failed at {returns.index[origin].date()}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    continue

                future_returns = returns.iloc[origin + 1:origin + horizon + 1]
                if len(future_returns) != horizon:
                    continue
                realized_variance = float(np.sum(np.square(future_returns.to_numpy(dtype=float))))
                realized_volatility = float(np.sqrt(max(realized_variance, 0.0)))

                all_rows.append({
                    "model": spec,
                    "horizon": int(horizon),
                    "origin_date": returns.index[origin],
                    "target_date": returns.index[origin + horizon],
                    "forecast_variance": forecast_variance,
                    "forecast_volatility": float(np.sqrt(forecast_variance)),
                    "realized_variance": realized_variance,
                    "realized_volatility": realized_volatility,
                    "QLIKE": qlike(realized_variance, forecast_variance),
                })

            # Benchmark is independent of fitted model and uses only returns
            # through the current forecast origin.
            hist_window = returns.iloc[
                max(0, origin - ROLLING_WINDOW + 1):origin + 1
            ].to_numpy(dtype=float)
            hist_daily_variance = float(np.mean(np.square(hist_window)))
            hist_forecast_variance = hist_daily_variance * horizon
            future_returns = returns.iloc[origin + 1:origin + horizon + 1]
            if len(future_returns) == horizon:
                realized_variance = float(np.sum(np.square(future_returns.to_numpy(dtype=float))))
                realized_volatility = float(np.sqrt(max(realized_variance, 0.0)))
                all_rows.append({
                    "model": "Rolling historical volatility",
                    "horizon": int(horizon),
                    "origin_date": returns.index[origin],
                    "target_date": returns.index[origin + horizon],
                    "forecast_variance": hist_forecast_variance,
                    "forecast_volatility": float(np.sqrt(max(hist_forecast_variance, 0.0))),
                    "realized_variance": realized_variance,
                    "realized_volatility": realized_volatility,
                    "QLIKE": qlike(realized_variance, hist_forecast_variance),
                })

            if (count + 1) % 50 == 0:
                print(f"  Processed {count + 1}/{len(origins)} origins")

        print("  Fit/forecast issues:", failures)

    predictions = pd.DataFrame(all_rows)
    if predictions.empty:
        raise RuntimeError("No forecasts produced.")
    predictions.to_csv(OUTPUT_DIR / "predictions.csv", index=False)

    metric_rows = []
    for (horizon, model_name), group in predictions.groupby(["horizon", "model"]):
        errors = group["forecast_volatility"].to_numpy() - group["realized_volatility"].to_numpy()
        metric_rows.append({
            "horizon": int(horizon),
            "model": model_name,
            "n_predictions": int(len(group)),
            "volatility_MAE_pct_points": float(np.mean(np.abs(errors))),
            "volatility_RMSE_pct_points": float(np.sqrt(np.mean(errors ** 2))),
            "mean_QLIKE": float(group["QLIKE"].mean()),
        })
    metrics = pd.DataFrame(metric_rows).sort_values(["horizon", "model"])
    metrics.to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    print("\nWALK-FORWARD METRICS (lower is better)")
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    # Relative improvement against historical benchmark; positive means GARCH/GJR
    # lower error. QLIKE percentage improvement is omitted because raw QLIKE can
    # be negative and ratios are not generally interpretable.
    improvement_rows = []
    for horizon in HORIZONS:
        hist = metrics[
            (metrics["horizon"] == horizon)
            & (metrics["model"] == "Rolling historical volatility")
        ]
        if hist.empty:
            continue
        for spec in specs:
            model_row = metrics[
                (metrics["horizon"] == horizon)
                & (metrics["model"] == spec)
            ]
            if model_row.empty:
                continue
            improvement_rows.append({
                "horizon": horizon,
                "model": spec,
                "MAE_improvement_vs_historical_pct": 100 * (
                    1 - model_row["volatility_MAE_pct_points"].iloc[0]
                    / hist["volatility_MAE_pct_points"].iloc[0]
                ),
                "RMSE_improvement_vs_historical_pct": 100 * (
                    1 - model_row["volatility_RMSE_pct_points"].iloc[0]
                    / hist["volatility_RMSE_pct_points"].iloc[0]
                ),
                "GARCH_minus_historical_mean_QLIKE": (
                    model_row["mean_QLIKE"].iloc[0] - hist["mean_QLIKE"].iloc[0]
                ),
            })
    improvements = pd.DataFrame(improvement_rows)
    improvements.to_csv(OUTPUT_DIR / "benchmark_improvement.csv", index=False)
    print("\nIMPROVEMENT VS HISTORICAL BENCHMARK (positive MAE/RMSE = better)")
    print(improvements.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    diagnostics = pd.DataFrame(diagnostic_rows)
    diagnostics.to_csv(OUTPUT_DIR / "model_diagnostics.csv", index=False)

    # Average forecasts and realized volatility by horizon/model.
    avg_rows = []
    for (horizon, model_name), group in predictions.groupby(["horizon", "model"]):
        avg_rows.append({
            "horizon": int(horizon),
            "model": model_name,
            "mean_forecast_volatility": float(group["forecast_volatility"].mean()),
            "mean_realized_volatility": float(group["realized_volatility"].mean()),
        })
    avg = pd.DataFrame(avg_rows)
    avg.to_csv(OUTPUT_DIR / "average_forecasts_by_horizon.csv", index=False)

    for metric, ylabel, filename in [
        ("volatility_MAE_pct_points", "Volatility MAE (percentage points)", "horizon_mae.png"),
        ("volatility_RMSE_pct_points", "Volatility RMSE (percentage points)", "horizon_rmse.png"),
        ("mean_QLIKE", "Mean QLIKE (lower is better)", "horizon_qlike.png"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for model_name, group in metrics.groupby("model"):
            group = group.sort_values("horizon")
            ax.plot(group["horizon"], group[metric], marker="o", label=model_name)
        ax.set_title(f"TCS: {ylabel} by Forecast Horizon")
        ax.set_xlabel("Forecast horizon (trading days)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / filename, dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for model_name, group in avg.groupby("model"):
        group = group.sort_values("horizon")
        ax.plot(group["horizon"], group["mean_forecast_volatility"],
                marker="o", label=f"{model_name} forecast")
    realized = avg.groupby("horizon", as_index=False)["mean_realized_volatility"].mean()
    ax.plot(realized["horizon"], realized["mean_realized_volatility"],
            marker="o", linestyle="--", label="Mean realized volatility")
    ax.set_title("TCS: Forecast vs Realized Cumulative Volatility")
    ax.set_xlabel("Forecast horizon (trading days)")
    ax.set_ylabel("Cumulative volatility (percentage points)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "forecast_comparison.png", dpi=150)
    plt.close(fig)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print("Review model_diagnostics.csv before deciding whether GJR-GARCH is justified.")
    print("Overlapping forecast errors are dependent; results are descriptive.")


if __name__ == "__main__":
    main()
