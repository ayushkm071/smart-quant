"""
Smart-Quant — GARCH v2 walk-forward volatility evaluation
Ticker: TCS.NS

Compares GARCH(1,1) with Student-t innovations against a rolling 252-return
historical-volatility benchmark over 1, 5, 21, 30, and 45 trading days.
Evaluation uses the latest 252 eligible target dates. Metrics: MAE, RMSE, QLIKE.

Realized-volatility proxy:
    sqrt(sum(future daily log return percentage points ** 2))

Forecasts use only data available at each forecast origin. Horizons overlap,
so forecast errors are dependent; results are descriptive until a suitable
dependence-aware statistical test is performed. This is not a profitability test.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from arch import arch_model

warnings.filterwarnings("ignore", category=FutureWarning)

TICKER = "TCS.NS"
START_DATE = "2010-01-01"
HORIZONS = [1, 5, 21, 30, 45]
TEST_TARGETS = 252
ROLLING_WINDOW = 252
REFIT_EVERY = 5
MIN_TRAIN_RETURNS = 750

PROJECT_ROOT = Path(r"C:\Quant project folder")
OUTPUT_DIR = PROJECT_ROOT / "results" / "garch_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def download_prices() -> pd.Series:
    data = yf.download(
        TICKER, start=START_DATE, auto_adjust=True,
        progress=False, threads=False
    )
    if data.empty:
        raise RuntimeError(
            f"No data returned for {TICKER}. Check internet access and ticker."
        )

    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close[TICKER] if TICKER in close.columns else close.iloc[:, 0]

    close = pd.to_numeric(close, errors="coerce").dropna()
    close = close[~close.index.duplicated(keep="last")].sort_index()
    if len(close) < MIN_TRAIN_RETURNS + TEST_TARGETS + max(HORIZONS):
        raise RuntimeError(f"Not enough price observations: {len(close)}")
    return close.rename("Close")


def qlike(actual_variance: float, forecast_variance: float) -> float:
    # QLIKE loss, omitting the actual-only term that does not affect rankings.
    eps = 1e-12
    actual = max(float(actual_variance), eps)
    forecast = max(float(forecast_variance), eps)
    return float(np.log(forecast) + actual / forecast)


def main() -> None:
    print("=" * 76)
    print("SMART-QUANT | GARCH v2 WALK-FORWARD EVALUATION")
    print("=" * 76)

    close = download_prices()
    returns = (100.0 * np.log(close / close.shift(1))).dropna()
    returns.name = "return_pct"
    close.to_frame().to_csv(OUTPUT_DIR / "adjusted_close.csv")
    returns.to_frame().to_csv(OUTPUT_DIR / "daily_log_returns_pct.csv")

    first_target = len(returns) - TEST_TARGETS
    target_indices = np.arange(first_target, len(returns))

    print(f"Ticker: {TICKER}")
    print(f"Return observations: {len(returns):,}")
    print(f"Date range: {returns.index.min().date()} to {returns.index.max().date()}")
    print(
        f"Target dates: {returns.index[first_target].date()} to "
        f"{returns.index[-1].date()}"
    )
    print(f"Refit frequency: every {REFIT_EVERY} forecast origins")
    print(f"Benchmark: rolling {ROLLING_WINDOW}-return historical volatility")

    prediction_rows = []

    for horizon in HORIZONS:
        print(f"\nEvaluating horizon: {horizon} trading day(s)")
        # Each target date t has origin t-h; the h future returns ending at t
        # are used only to calculate realized volatility, never for fitting.
        origins = target_indices - horizon
        if origins.min() < MIN_TRAIN_RETURNS - 1:
            raise RuntimeError(
                f"Insufficient training history for horizon {horizon}."
            )

        fitted_result = None
        last_fit_origin = None
        fit_or_forecast_failures = 0

        for count, origin in enumerate(origins):
            if (
                fitted_result is None
                or last_fit_origin is None
                or origin - last_fit_origin >= REFIT_EVERY
            ):
                train = returns.iloc[:origin + 1]
                try:
                    model = arch_model(
                        train,
                        mean="Constant",
                        vol="GARCH",
                        p=1,
                        o=0,
                        q=1,
                        dist="StudentsT",
                        rescale=False,
                    )
                    fitted_result = model.fit(disp="off", show_warning=False)
                    last_fit_origin = int(origin)
                except Exception as exc:
                    fit_or_forecast_failures += 1
                    fitted_result = None
                    print(
                        f"  Fit failed at {returns.index[origin].date()}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    continue

            try:
                forecast = fitted_result.forecast(
                    horizon=horizon, reindex=False
                )
                daily_variances = forecast.variance.iloc[-1].to_numpy(dtype=float)
                garch_variance = float(np.sum(daily_variances[:horizon]))
                if not np.isfinite(garch_variance) or garch_variance <= 0:
                    fit_or_forecast_failures += 1
                    continue
            except Exception as exc:
                fit_or_forecast_failures += 1
                print(
                    f"  Forecast failed at {returns.index[origin].date()}: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue

            future_returns = returns.iloc[origin + 1:origin + horizon + 1]
            if len(future_returns) != horizon:
                continue
            realized_variance = float(
                np.sum(np.square(future_returns.to_numpy(dtype=float)))
            )
            realized_volatility = float(np.sqrt(max(realized_variance, 0.0)))

            hist_window = returns.iloc[
                max(0, origin - ROLLING_WINDOW + 1):origin + 1
            ].to_numpy(dtype=float)
            historical_daily_variance = float(np.mean(np.square(hist_window)))
            historical_variance = historical_daily_variance * horizon
            historical_volatility = float(np.sqrt(max(historical_variance, 0.0)))

            prediction_rows.append({
                "horizon": int(horizon),
                "origin_date": returns.index[origin],
                "target_date": returns.index[origin + horizon],
                "garch_forecast_variance": garch_variance,
                "garch_forecast_volatility": float(np.sqrt(garch_variance)),
                "historical_forecast_variance": historical_variance,
                "historical_forecast_volatility": historical_volatility,
                "realized_variance": realized_variance,
                "realized_volatility": realized_volatility,
                "garch_qlike": qlike(realized_variance, garch_variance),
                "historical_qlike": qlike(realized_variance, historical_variance),
            })

            if (count + 1) % 50 == 0:
                print(f"  Processed {count + 1}/{len(origins)} origins")

        print(f"  Fit/forecast failures: {fit_or_forecast_failures}")

    predictions = pd.DataFrame(prediction_rows)
    if predictions.empty:
        raise RuntimeError("No forecasts were generated. Check the error messages.")
    predictions.to_csv(OUTPUT_DIR / "predictions.csv", index=False)

    metric_rows = []
    for horizon, group in predictions.groupby("horizon"):
        specs = [
            ("GARCH(1,1)-Student-t", "garch_forecast_volatility", "garch_qlike"),
            ("Rolling historical volatility",
             "historical_forecast_volatility", "historical_qlike"),
        ]
        for model_name, forecast_col, qlike_col in specs:
            errors = (
                group[forecast_col].to_numpy(dtype=float)
                - group["realized_volatility"].to_numpy(dtype=float)
            )
            metric_rows.append({
                "horizon": int(horizon),
                "model": model_name,
                "n_predictions": int(len(group)),
                "volatility_MAE_pct_points": float(np.mean(np.abs(errors))),
                "volatility_RMSE_pct_points": float(np.sqrt(np.mean(errors ** 2))),
                "mean_QLIKE": float(group[qlike_col].mean()),
            })

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    print("\nWALK-FORWARD METRICS (lower is better)")
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    improvements = []
    for horizon in HORIZONS:
        garch_row = metrics[
            (metrics["horizon"] == horizon)
            & (metrics["model"] == "GARCH(1,1)-Student-t")
        ]
        hist_row = metrics[
            (metrics["horizon"] == horizon)
            & (metrics["model"] == "Rolling historical volatility")
        ]
        if garch_row.empty or hist_row.empty:
            continue
        improvements.append({
            "horizon": horizon,
            "GARCH_MAE_improvement_vs_historical_pct": 100 * (
                1 - garch_row["volatility_MAE_pct_points"].iloc[0]
                / hist_row["volatility_MAE_pct_points"].iloc[0]
            ),
            "GARCH_RMSE_improvement_vs_historical_pct": 100 * (
                1 - garch_row["volatility_RMSE_pct_points"].iloc[0]
                / hist_row["volatility_RMSE_pct_points"].iloc[0]
            ),
            "GARCH_QLIKE_improvement_vs_historical_pct": 100 * (
                1 - garch_row["mean_QLIKE"].iloc[0]
                / hist_row["mean_QLIKE"].iloc[0]
            ),
        })

    improvement_df = pd.DataFrame(improvements)
    improvement_df.to_csv(OUTPUT_DIR / "benchmark_improvement.csv", index=False)
    print("\nGARCH IMPROVEMENT VS HISTORICAL BENCHMARK (%)")
    print(improvement_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    summary = predictions.groupby("horizon", as_index=False).agg(
        realized_volatility=("realized_volatility", "mean"),
        garch_forecast_volatility=("garch_forecast_volatility", "mean"),
        historical_forecast_volatility=("historical_forecast_volatility", "mean"),
    )
    summary.to_csv(OUTPUT_DIR / "average_forecasts_by_horizon.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(summary["horizon"], summary["realized_volatility"],
            marker="o", label="Mean realized volatility")
    ax.plot(summary["horizon"], summary["garch_forecast_volatility"],
            marker="o", label="Mean GARCH forecast")
    ax.plot(summary["horizon"], summary["historical_forecast_volatility"],
            marker="o", label="Mean historical-volatility forecast")
    ax.set_title("TCS Volatility Forecasts vs Realized Volatility")
    ax.set_xlabel("Forecast horizon (trading days)")
    ax.set_ylabel("Cumulative volatility (percentage points)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "forecast_comparison.png", dpi=150)
    plt.close(fig)

    for metric, label, filename in [
        ("volatility_MAE_pct_points", "Volatility MAE (percentage points)", "horizon_mae.png"),
        ("volatility_RMSE_pct_points", "Volatility RMSE (percentage points)", "horizon_rmse.png"),
        ("mean_QLIKE", "Mean QLIKE (lower is better)", "horizon_qlike.png"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for model_name, group in metrics.groupby("model"):
            group = group.sort_values("horizon")
            ax.plot(group["horizon"], group[metric],
                    marker="o", label=model_name)
        ax.set_title(f"TCS: {label} by Forecast Horizon")
        ax.set_xlabel("Forecast horizon (trading days)")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / filename, dpi=150)
        plt.close(fig)

    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("Overlapping forecast horizons create dependent errors.")
    print("Treat results as descriptive until dependence-aware inference is performed.")


if __name__ == "__main__":
    main()
