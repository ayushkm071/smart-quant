import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==========================================
# SETUP & DIRECTORIES
# ==========================================
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 1. DOWNLOAD DATA
# ==========================================
ticker = "TCS.NS"

df = yf.download(
    ticker,
    period="5y",
    interval="1d",
    auto_adjust=True
)

# Keep only closing price and drop missing values
df = df[["Close"]].dropna()

print("Data:")
print(df.head())
print("\nNumber of observations:", len(df))

# ==========================================
# 2. VISUALIZE TCS PRICE
# ==========================================
plt.figure(figsize=(12, 5))
plt.plot(df.index, df["Close"])
plt.title("TCS - Last 5 Years")
plt.xlabel("Date")
plt.ylabel("Price (₹)")
plt.grid(True)
plt.show()

# ==========================================
# 3. TRAIN / VALIDATION SPLIT
# ==========================================
# Keep the final 252 trading days for validation (~1 trading year)
validation_size = 252

train = df["Close"].iloc[:-validation_size]
validation = df["Close"].iloc[-validation_size:]

print("\nTraining observations:", len(train))
print("Validation observations:", len(validation))

# ==========================================
# 4. ARIMA GRID SEARCH
# ==========================================
print("\n===== ARIMA GRID SEARCH STARTING =====", flush=True)

best_rmse = float("inf")
best_order = None
results = []

# ==========================================
# 4. ARIMA GRID SEARCH (108 COMBINATIONS)
# ==========================================
print("\n===== ARIMA GRID SEARCH STARTING =====", flush=True)

best_rmse = float("inf")
best_order = None
results = []

# Expanded grid: 6 (p) * 3 (d) * 6 (q) = 108 combinations
for p in range(0, 6):
    for d in range(0, 3):
        for q in range(0, 6):
            
            order = (p, d, q)
            print(f"\nSTARTING ARIMA{order}", flush=True)

            try:
                model = ARIMA(
                    train.to_numpy().ravel(),
                    order=order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )

                print(f"Fitting ARIMA{order}...", flush=True)
                fitted_model = model.fit(method_kwargs={"maxiter": 30})
                print(f"FIT COMPLETE: ARIMA{order}", flush=True)

                prediction = fitted_model.forecast(steps=len(validation))

                rmse = np.sqrt(mean_squared_error(validation.to_numpy().ravel(), prediction))
                mae = mean_absolute_error(validation.to_numpy().ravel(), prediction)
                aic = fitted_model.aic

                results.append({
                    "p": p,
                    "d": d,  # <-- Make sure this captures the variable d, not a hardcoded 1
                    "q": q,
                    "RMSE": rmse,
                    "MAE": mae,
                    "AIC": aic
                })

                print(f"ARIMA{order} | RMSE={rmse:.2f} | MAE={mae:.2f} | AIC={aic:.2f}", flush=True)

                if rmse < best_rmse:
                    best_rmse = rmse
                    best_order = order

            except Exception as e:
                print(f"ARIMA{order} FAILED: {e}", flush=True)
                continue

# ==========================================
# 5. CREATE & SAVE RESULTS TABLE
# ==========================================
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="RMSE").reset_index(drop=True)

results_file = RESULTS_DIR / "TCS_ARIMA_grid_results.csv"
results_df.to_csv(results_file, index=False)

print("\n================================")
print("GRID SEARCH COMPLETE")
print("================================")
print("Best parameters:", best_order)
print(f"Best RMSE: {best_rmse:.2f}")
print(f"\nResults saved to: {results_file}")

print("\nTop models:")
print(results_df.head(10).to_string(index=False))

# ==========================================
# 7. FIT FINAL MODEL ON ALL DATA (THE FIX)
# ==========================================
print(f"\nFitting final model on FULL dataset using best order: {best_order}")

# Use the entire dataset for out-of-sample forecasting
full_series = df["Close"].to_numpy().ravel()

final_model = ARIMA(
    full_series,
    order=best_order,
    enforce_stationarity=False,
    enforce_invertibility=False
)

final_model_fit = final_model.fit(method_kwargs={"maxiter": 50})

# ==========================================
# 8. MONTE CARLO SIMULATION
# ==========================================
forecast_steps = 252
num_paths = 50

print(f"\nRunning Monte Carlo simulation with {num_paths} paths...")

# simulate() uses the fitted parameters and error variance to generate random paths
# anchor="end" ensures it starts exactly where your historical data leaves off
simulated_paths = final_model_fit.simulate(
    nsimulations=forecast_steps,
    anchor="end",
    repetitions=num_paths
)

# Also grab the standard flat forecast to show the "expected mean"
mean_forecast = final_model_fit.forecast(steps=forecast_steps)


# ==========================================
# 9. CREATE FUTURE BUSINESS-DAY INDEX
# ==========================================
last_date = df.index[-1]
future_dates = pd.bdate_range(
    start=last_date + pd.Timedelta(days=1),
    periods=forecast_steps
)

# Wrap the simulations in a DataFrame for easy plotting
# .squeeze() flattens (252, 1, 50) into a clean 2D grid of (252, 50)
sim_df = pd.DataFrame(simulated_paths.squeeze(), index=future_dates)
mean_df = pd.DataFrame({"Mean Forecast": mean_forecast}, index=future_dates)


# ==========================================
# 10. PLOT MONTE CARLO SIMULATION
# ==========================================
plt.figure(figsize=(14, 7))

# 1. Plot historical actuals (last 1 trading year)
plt.plot(
    df.index[-252:], 
    df["Close"].iloc[-252:], 
    label="Actual (Last Year)", 
    color="blue",
    linewidth=2
)

# 2. Plot all 50 simulated random paths
plt.plot(
    sim_df.index, 
    sim_df.values, 
    color="gray", 
    alpha=0.15, # High transparency so it looks like a cloud
    linewidth=1
)

# 3. Plot the standard flat ARIMA mean forecast on top
plt.plot(
    mean_df.index, 
    mean_df["Mean Forecast"], 
    label="Expected Mean (Flat)", 
    color="orange",
    linestyle="--",
    linewidth=2
)

plt.title(f"TCS Monte Carlo Simulation - {num_paths} Paths (Order: {best_order})")
plt.xlabel("Date")
plt.ylabel("Price (₹)")

# Custom legend to avoid showing 50 gray path labels
handles, labels = plt.gca().get_legend_handles_labels()
from matplotlib.lines import Line2D
sim_patch = Line2D([0], [0], color='gray', alpha=0.5, lw=2, label='Simulated Paths')
handles.append(sim_patch)

plt.legend(handles=handles)
plt.grid(True)
plt.show()