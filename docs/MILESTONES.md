# Milestones 3–5 — Smart-Quant

Project: Design and Implementation of a Multi-Model Quantitative Forecasting System  
Date: 27 October 2026

## Milestone 3 — Integration & Intermediate Evaluation

**Implemented:** a shared, validated OHLCV loader; separate price and volatility result schemas; explicit ARIMA, XGBoost, LSTM, and GARCH forecast interfaces; and a five-page Streamlit dashboard in `front end/dashboard_app.py`, launched through `main.py`.

**Pages:** Overview, Forecasts, Model Performance, Volatility & Risk, and Diagnostics. Saved backtest metrics/predictions are read from `results/`. New model fitting occurs only after a user requests a forecast; navigation does not retrain.

**Version selection:** active source files are ARIMA `models/arima_model.py`, XGBoost `models/xgboost_model_v2.py`, LSTM `models/lstm_model_v2.py`, and GARCH `models/garch_model_v2.py`. The latest saved price evaluations are ARIMA v2, XGBoost walk-forward v4, and LSTM v2. The newest GARCH evaluation files are `results/garch_v4/`, while their source script is in `models archived/`; the dashboard labels these as historical GARCH v4 results and uses active GARCH v2 for new forecasts.

**Forecast integrity:** all models use the same last available data date and chosen horizon. Future target dates remain unset because no NSE holiday calendar is installed; the forecast chart shows endpoint markers at a session offset and no future price path. Equal-weight consensus is marked experimental and uses only successful ARIMA/XGBoost/LSTM forecasts. GARCH results have separate units and schema.

**Saved evaluation snapshot:** the reports cover 252 target dates ending 26 October 2026. Price MAE is in INR; GARCH MAE is in cumulative horizon return percentage points and must not be ranked against price MAE.

| Horizon | ARIMA MAE | XGBoost MAE | LSTM MAE | Naive MAE | GARCH v4 MAE |
|---:|---:|---:|---:|---:|---:|
| 1 day | 33.29 | 32.87 | 32.87 | 32.87 | 1.059 |
| 5 days | 77.71 | 77.26 | 77.26 | 77.26 | 1.491 |
| 21 days | 179.91 | 177.61 | 177.61 | 177.61 | 1.993 |
| 30 days | 210.82 | 209.62 | 209.62 | 209.62 | 2.262 |
| 45 days | 255.85 | 255.23 | 255.23 | 255.23 | 2.638 |

The saved XGBoost and LSTM evaluation reports show zero mean shrinkage beta at these horizons. Their recorded shrunk forecasts therefore reduce to the naive current-price forecast in those backtests; this is not evidence of model skill.

## Milestone 4 — Comprehensive Testing & Optimization

**Automated report:** 10 unit/integration checks pass using the bundled Python 3.12 runtime. They cover CSV schema/date validation, data-quality counts, price/volatility result schemas, horizon validation, model-error reporting, saved price and volatility evaluation readers, and syntax/page presence for the Streamlit entry and all five pages.

Run locally with `python -m unittest discover -s tests -v` after installing pandas and NumPy from `requirements.txt`.

**Model execution coverage limitation:** this workspace runtime lacks Streamlit, Plotly, statsmodels, XGBoost, scikit-learn, TensorFlow, `arch`, and yfinance. The checks therefore do not launch the Streamlit server or run live model fits. The saved XGBoost artifacts and their metadata are present, but their load/predict path still needs a smoke test in the project's fully installed Python 3.12 environment. The ARIMA/LSTM/GARCH live forecast paths likewise need dependency-enabled smoke tests.

**Optimization scope:** existing architectures, evaluation methods, and model hyperparameters were preserved. XGBoost uses a matching-date saved artifact, or saved per-horizon parameters without repeating its parameter search. LSTM training is explicit and on-demand because its script saves no deployable weights. No new optimizer or portfolio engine was added.

**Data handling:** refresh downloads auto-adjusted OHLCV into a temporary sibling file, validates it, then atomically replaces the cache. A failed refresh leaves the prior CSV intact. Existing model results and archived files were not overwritten or deleted.

## Milestone 5 — Deployment, Documentation & Viva Preparation

**Local dashboard:** install `requirements.txt`, then run `streamlit run main.py` from the project root. See `README.md` for Windows/VS Code setup and page behavior.

**GitHub:** the project is tracked in `https://github.com/ayushkm071/smart-quant`. Curated evaluation tables, prediction rows, model metadata, and XGBoost artifacts required by the dashboard are included; generated plots, unrelated historic results, coursework, and reference PDFs are excluded.

**Hosting:** the Streamlit app is not deployed as a live web service by this milestone. GitHub stores the source; a separate Python host with sufficient memory and compatible ML libraries is needed for a live demo. There is no portfolio optimizer, VaR/CVaR page, broker integration, or trading automation.

**Viva outline:**

1. Research objective and TCS.NS data source.
2. Adjusted-price convention, validation, cache freshness, and data refresh.
3. Existing ARIMA, XGBoost, LSTM, and GARCH roles and saved artifacts.
4. Shared interfaces, model status/error handling, and five dashboard pages.
5. Walk-forward results by horizon; separate price and volatility units; explain zero-shrinkage backtest findings.
6. Integrity limits: no exact future calendar date without an NSE calendar, overlapping forecast targets, unavailable full model smoke test in this workspace, and no trading/portfolio features.
7. Demo: run the Streamlit dashboard, inspect Overview, request forecasts, filter historical performance, inspect risk metrics, and review Diagnostics.

## Evidence checklist for SDMS

- **Milestone 3:** app screenshots, sample forecast table, and evaluation source/version note.
- **Milestone 4:** `unittest` output, dependency-enabled model smoke-test output when run locally, and evaluation metric table.
- **Milestone 5:** GitHub URL, README/setup notes, live hosting URL if separately deployed, and viva slides or outline.
