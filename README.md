# Smart-Quant: TCS Multi-Model Forecasting Dashboard

Smart-Quant analyzes financial market time series. This project combines existing ARIMA, XGBoost, and LSTM price forecasts with a separate GARCH volatility forecast for TCS.NS.

## Project Goal

Explore and evaluate several approaches to price forecasting and volatility estimation using historical daily market data.

## Current Features

- Shared OHLCV preprocessing for `data/TCS.NS_daily.csv`
- ARIMA, XGBoost, and LSTM price forecast interfaces
- GARCH(1,1)-Student-t cumulative volatility forecast
- Unified JSON response and local web dashboard
- Walk-forward model evaluation outputs produced by the existing model scripts

## Requirements

- Python 3.12
- The packages listed in `requirements.txt`
- The included OHLCV CSV at `data/TCS.NS_daily.csv`

Create an environment and install the dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run locally

From the project root:

```powershell
python main.py
```

Open <http://127.0.0.1:8000>. Choose a model and horizon (1, 5, 21, 30, or 45 trading days), then select **Generate forecast**. The API is also available at `/api/models` and `/api/forecast?model=all&horizon=5`.

Forecasts are computed when requested. LSTM and XGBoost model fitting can take a while. GARCH returns a volatility forecast in cumulative return percentage points; the other models return a projected close in INR and the associated return percentage.

## Integration behavior

- `src/data_loader.py` is the shared, validated OHLCV reader.
- `src/forecasting.py` adapts the existing model code to a common result format.
- ARIMA uses an order saved in `results/arima_v2/selected_orders.csv` if present; otherwise it uses ARIMA(1,1,1).
- XGBoost uses the model's first declared parameter candidate. The integration does not run the historical candidate search.
- LSTM uses the existing model architecture, lookback, seed ensemble, and calibration routine.
- GARCH uses GARCH(1,1) with Student-t errors, matching the existing model specification.

The integration layer does not change the existing model training or optimization configuration. Model outputs are estimates, not investment advice.

## Tests

Run the lightweight integration checks with:

```powershell
python -m unittest discover -s tests -v
```

The test suite checks the CSV contract, result serialization, horizon/model validation, and the dashboard/API shell. A full model smoke test requires all model dependencies from `requirements.txt` and should be run in the same environment used for deployment.

## GitHub and hosting

The source repository is <https://github.com/ayushkm071/smart-quant>. GitHub stores the source; it does not run this Python API. For a live dashboard, use a Python web-service host that supports the `requirements.txt` dependencies, configure the start command as `python main.py` with `HOST=0.0.0.0`, and provide the CSV at `data/TCS.NS_daily.csv`. TensorFlow and model fitting require more memory and startup time than a static site, so confirm the host's resource limits before enabling public traffic.

## Technologies

Python, ARIMA, XGBoost, LSTM, GARCH, NumPy, pandas, scikit-learn, statsmodels, TensorFlow, and Matplotlib.

## Author

Ayush Mishra

