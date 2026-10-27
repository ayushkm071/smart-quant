# Smart-Quant Dashboard Test Report

Date: 27 October 2026

## Automated checks

Command: `python -m unittest discover -s tests -v`

Result: **10 checks passed** in the bundled Python 3.12 runtime.

| Area | Check performed | Result |
|---|---|---|
| Market data | Load cached OHLCV, ordering, positive close, required columns | Pass |
| Data quality | Count valid rows, duplicate dates, latest date | Pass |
| Result contracts | Price forecast fields and separate volatility units | Pass |
| Forecast controls | Supported horizon validation | Pass |
| Failure handling | Model exception is returned with explicit error status | Pass |
| Historical performance | Load saved ARIMA, XGBoost, and LSTM metrics/predictions | Pass |
| Historical risk | Load saved GARCH evaluation and unit-labelled benchmark | Pass |
| Dashboard source | Python syntax, Streamlit entry point, five page functions | Pass |

## Page and model coverage

| Dashboard page | What was checked | Runtime status |
|---|---|---|
| Market Overview | Source implementation uses validated cached OHLCV and interactive price/volume/volatility/drawdown charts | Not launched in this environment |
| Forecasting Engine | Source implementation uses same origin/horizon, endpoint-only markers, explicit per-model status, experimental consensus | Not launched; live model runs unavailable |
| Model Performance & Backtesting | Saved evaluation readers tested for price models, metrics, and predictions | Reader tests pass; charts not visually checked |
| Volatility & Risk | Saved GARCH v4 metrics/predictions reader tested; separate volatility schema tested | Reader tests pass; live GARCH run unavailable |
| Diagnostics & System Monitor | Source implementation reports data quality, model/artifact availability, and result paths | Not launched in this environment |

| Model | Integration interface | Data/model test result |
|---|---|---|
| ARIMA | Uses saved horizon order when available; otherwise fixed ARIMA(1,1,1), with no order search | Interface source reviewed; live fit not run |
| XGBoost | Loads only a matching-date saved artifact; otherwise uses existing fixed parameters and fit helper, with no parameter search | Evaluation reader tests pass; artifact prediction not run |
| LSTM | Uses existing feature/training helpers on explicit request; no deployable weights exist in saved results | Evaluation reader tests pass; TensorFlow fit not run |
| GARCH | Separate active v2 conditional volatility forecast; v4 saved output is historical evaluation | Schema/evaluation reader tests pass; live fit not run |

## Limits

The available bundled runtime has pandas and NumPy but lacks Streamlit, Plotly, statsmodels, XGBoost, scikit-learn, TensorFlow, `arch`, and yfinance. Therefore the Streamlit server, interactive browser rendering, and dependency-backed live forecasts could not be exercised here. Install `requirements.txt` in the project's Python 3.12 environment, run the unit test command above, then launch with `streamlit run main.py` for the runtime smoke test. No future market values or model metrics were fabricated for these checks.
