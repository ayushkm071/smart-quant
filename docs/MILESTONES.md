# Project Milestones 3–5

Project: TCS.NS Multi-Level Quant Forecasting  
Date: 27 October 2026

## Milestone 3 — Integration & Intermediate Evaluation

**Implemented:** the shared loader reads and validates `data/TCS.NS_daily.csv`; ARIMA, XGBoost, LSTM, and GARCH have callable interfaces; price and volatility forecasts share a documented response structure; `main.py` serves a small dashboard and JSON API.

**Intermediate evaluation approach:** price models report projected close and percentage return; GARCH reports cumulative volatility in return percentage points. These output types are intentionally distinguished in the common result so the volatility estimate is not presented as a price target.

**Model preservation:** the existing model files and architectures remain unchanged. The XGBoost integration uses its first declared parameter candidate and skips historical parameter search. ARIMA reads a saved order if one exists and otherwise uses ARIMA(1,1,1). No new model optimization was added.

## Milestone 4 — Comprehensive Testing & Optimization

**Automated checks:** all 8 checks pass in the bundled Python environment. `tests/` validates CSV loading and schema errors, common-result serialization, unsupported model/horizon handling, and the dashboard/API routes without binding a network port. Run them using `python -m unittest discover -s tests -v`.

**Engineering improvements:** the shared loader normalizes dates and numeric OHLCV values; model errors from an all-model request are isolated per model; generated research outputs, Python caches, local secrets, and unrelated coursework/reference PDFs are excluded from source publication by `.gitignore`.

**Optimization scope:** no model architecture, fitting protocol, hyperparameter, or training optimization has been changed. This follows the project requirement to preserve the models. The integration uses existing settings and omits XGBoost's historical search for an on-demand dashboard forecast.

**Evaluation limitation:** this environment does not have the forecasting libraries installed, so full model-inference and deployed-resource performance checks remain to be run after installing `requirements.txt`. The included automated suite is an integration/unit suite, not evidence of comparative forecast performance.

**Existing evaluation snapshot:** the saved walk-forward reports cover 252 target dates through 26 October 2026. Price MAE below is in INR; GARCH MAE is in cumulative return percentage points. These are separate target types and should not be ranked against one another.

| Horizon | ARIMA price MAE | XGBoost price MAE | LSTM price MAE | Naive price MAE | GARCH volatility MAE |
|---:|---:|---:|---:|---:|---:|
| 1 day | 33.29 | 32.87 | 32.87 | 32.87 | 1.05 |
| 5 days | 77.71 | 77.26 | 77.26 | 77.26 | 1.47 |
| 21 days | 179.91 | 177.61 | 177.61 | 177.61 | 2.04 |
| 30 days | 210.82 | 209.62 | 209.62 | 209.62 | 2.27 |
| 45 days | 255.85 | 255.23 | 255.23 | 255.23 | 2.63 |

The stored XGBoost and LSTM evaluation reports have a zero mean shrinkage beta at each listed horizon, so their price forecasts reduce to the naive current-price forecast in those reports. The integration must not present those results as independent evidence of model skill.

## Milestone 5 — Deployment, Final Documentation & Viva Preparation

**Prepared:** `README.md` documents setup, run instructions, API endpoints, test command, model behavior, and hosting considerations. `.gitignore` avoids committing generated outputs and unrelated local material. `requirements.txt` declares the runtime dependencies. The app can be hosted as a Python web service with `python main.py` as its start command and the CSV made available at its expected path.

**GitHub publication:** pending an authenticated GitHub connection and a destination repository. Choose the repository name and visibility, then review the publish set before pushing. GitHub repository hosting publishes source code; a separate Python web-service host is needed to run the dashboard/API.

**Viva outline:**

1. Problem statement: compare price-forecasting approaches and model volatility separately for TCS.NS.
2. Data: daily OHLCV CSV, date cleanup, missing/non-finite-value handling, and adjusted close context.
3. Models: ARIMA for price levels; XGBoost and LSTM for direct return-based price forecasts; GARCH for conditional volatility.
4. Integration: shared loader, adapter functions, common response fields, API, and dashboard.
5. Evaluation: explain the existing models' walk-forward design and report their saved metrics with horizon and unit labels; avoid comparing GARCH volatility directly to price error.
6. Limitations: historical estimates are not guaranteed future performance; overlapping multi-day targets are dependent; live forecasts need installed dependencies and sufficient host memory/time.
7. Demo: launch `python main.py`, open the dashboard, request a horizon from each model, and show the corresponding JSON API response.

## Completion evidence to attach in SDMS

- Milestone 3: integration architecture, dashboard screenshot, and example all-model API response.
- Milestone 4: test command/output, existing model evaluation metric tables, and a note on the unchanged optimization scope.
- Milestone 5: GitHub repository URL, README, deployment URL if hosted, and viva slides/outline.
