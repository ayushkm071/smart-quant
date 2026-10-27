# Smart-Quant — Multi-Model Forecasting & Risk Analytics

Smart-Quant is a local Streamlit research dashboard for TCS.NS. It reads the shared auto-adjusted OHLCV cache, serves current-origin price forecasts from ARIMA, XGBoost, and LSTM, and keeps GARCH volatility analysis separate from price direction.

## Run locally on Windows

Use Python 3.12 (recommended for TensorFlow compatibility). For everyday use, double-click `run_dashboard.bat` in the project folder. On first run it creates `.venv` and installs `requirements.txt`; later runs reuse that environment and open the local Streamlit page. Keep the terminal window open while using the dashboard. Close it (or press Ctrl+C) to stop the local server.

If the first dependency download times out, press a key to close the error window and run `run_dashboard.bat` again. The launcher retries package downloads with a longer timeout; packages that already installed are reused.

You can also start it manually from PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run main.py
```

The dashboard opens in the browser. The five pages are Overview, Forecasts, Model Performance, Volatility & Risk, and Diagnostics. Use the top-bar **Refresh data** control to update `data/TCS.NS_daily.csv`; the cache is replaced atomically only after downloaded data passes validation.

## Model and result sources

| Component | Active callable source | Saved evaluation/artifacts |
|---|---|---|
| ARIMA | `models/arima_model.py` specification, called through `src/forecasting.py` | `results/arima_v2/` |
| XGBoost | `models/xgboost_model_v2.py` feature/training helpers | `results/xgboost_return_walkforward_v4/`; loads matching-date `.ubj` models, otherwise fits the selected fixed settings without rerunning the candidate search |
| LSTM | `models/lstm_model_v2.py` existing training routine | `results/lstm_direct_walkforward_v2/`; evaluation weights are not saved, so latest-origin fit runs only when explicitly requested |
| GARCH forecast | Active `models/garch_model_v2.py` specification | Latest saved risk evaluation is `results/garch_v4/`; its source script is in `models archived/` and is labelled as historical |

The ARIMA adapter reads the selected order from `results/arima_v2/metrics.csv`. XGBoost uses saved artifacts only when their run metadata matches the current cache date; otherwise the adapter fits with saved per-horizon parameters or the existing first candidate. LSTM and GARCH run only after an explicit forecast action. No historical backtest is rerun when navigating between pages.

Future target dates are left unset because the project does not include an NSE holiday calendar. Forecasts use one shared data origin and horizon; the chart marks endpoints at a session offset and does not draw future paths. Consensus is an experimental equal-weight average of successful price-model returns. It is not presented as a superior estimator.

## Dashboard pages

- **Overview:** adjusted close, daily/5/21-session returns, historical price, volume, rolling volatility, and drawdown.
- **Forecasts:** ARIMA/XGBoost/LSTM comparison for 1, 5, 21, 30, or 45 sessions; status/errors, endpoint chart, disagreement, and experimental consensus.
- **Model Performance:** saved walk-forward MAE/RMSE, directional accuracy, correlation where available, benchmark errors, actual/predicted values, and residual views.
- **Volatility & Risk:** GARCH forecast and saved GARCH v4 benchmark evaluation; volatility remains in cumulative horizon return percentage points. Historical daily-return volatility is not annualized.
- **Diagnostics:** data validation, code/package availability, saved files, model metadata, orders, and execution status.

Portfolio optimization, VaR/CVaR, broker execution, and annualized volatility are not implemented.

## Tests

```powershell
python -m unittest discover -s tests -v
```

The lightweight suite checks the CSV contract, result schemas, forecast failure handling, saved-result readers, and dashboard source structure. Full model smoke tests require all packages in `requirements.txt`; LSTM and model-library behavior should be checked in the same Python environment used to run Streamlit.

## Data and research cautions

`Close` is treated as adjusted close, matching the project cache produced with Yahoo Finance `auto_adjust=True`. The UI shows the latest observation date and its calendar age; it does not describe the cached value as live. Forecasts and saved backtest metrics are research estimates, not investment advice. Overlapping multi-day evaluation targets are dependent.

## Repository and deployment

Source repository: <https://github.com/ayushkm071/smart-quant>. GitHub hosts the source and curated evaluation artifacts, not the running Streamlit application. To make a live dashboard, deploy to a Python service that supports Streamlit, TensorFlow, XGBoost, statsmodels, and `arch`, and provide the CSV at the expected `data/TCS.NS_daily.csv` path.

## Project author

Ayush Mishra
