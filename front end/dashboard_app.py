"""Five-page Streamlit dashboard for Smart-Quant."""
from __future__ import annotations

import importlib.util
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.dashboard_data import EVALUATION_DIRS, evaluation_file_inventory, load_price_evaluations, load_volatility_evaluations
from src.data_loader import DEFAULT_DATA_PATH, PROJECT_ROOT, inspect_data_file, load_daily_data, refresh_daily_data
from src.forecasting import ALL_MODELS, HORIZONS, PRICE_MODELS, run_forecasts, validate_horizon

TICKER = "TCS.NS"
MODEL_COLORS = {"ARIMA": "#387ED1", "XGBoost": "#8155C7", "LSTM": "#26A69A", "GARCH(1,1)-Student-t": "#D8913D", "GJR-GARCH(1,1)-Student-t": "#8C6D3F", "Rolling historical volatility": "#80868B"}

st.set_page_config(page_title="Smart-Quant · Market Analytics", page_icon="▦", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
[data-testid="stAppViewContainer"], [data-testid="stMain"] { background:#fff; }
[data-testid="stHeader"] { background:#fff; border-bottom:1px solid #e6e8eb; }
[data-testid="stMainBlockContainer"] { max-width:1540px; padding:1.2rem 2rem 3rem; }
html,body,[class*="css"] { font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:#333; }
h1 { font-size:1.55rem!important; font-weight:650!important; letter-spacing:-.025em; }
h2 { font-size:1.15rem!important; font-weight:620!important; }
[data-testid="stMetric"] { padding:.35rem .25rem .5rem; border-bottom:1px solid #e6e8eb; }
[data-testid="stMetricLabel"] { color:#777; font-size:.74rem; }
[data-testid="stMetricValue"] { color:#333; font-size:1.35rem; font-variant-numeric:tabular-nums; }
.stButton>button,[data-testid="stDownloadButton"] button { border:1px solid #d9dfe5; border-radius:4px; color:#333; background:#fff; min-height:2.35rem; }
.stButton>button:hover { border-color:#387ed1; color:#387ed1; }
[data-testid="stBaseButton-primary"] { border-color:#387ed1!important; background:#387ed1!important; }
[data-testid="stRadio"]>div { gap:.35rem; }
[data-testid="stRadio"] label { padding:.4rem .72rem; border:1px solid #e6e8eb; border-radius:4px; background:#fff; color:#555; }
[data-testid="stRadio"] label:has(input:checked) { border-color:#387ed1; color:#387ed1; background:#f5f9fe; }
[data-testid="stSelectbox"] [data-baseweb="select"]>div { min-height:2.35rem; border-color:#dfe3e8; border-radius:4px; }
[data-testid="stDataFrame"] { border:1px solid #e6e8eb; border-radius:3px; }
hr { border-color:#e6e8eb; }
.kicker { color:#777; font-size:.7rem; font-weight:650; letter-spacing:.08em; text-transform:uppercase; }
.toolbar { padding:.3rem 0 .65rem; border-bottom:1px solid #e6e8eb; }
.page-note { padding:.65rem .8rem; background:#f7f8fa; border-left:2px solid #aab1b8; color:#666; font-size:.82rem; }
@media(max-width:760px){[data-testid="stMainBlockContainer"]{padding:.9rem .85rem 2rem}}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def cached_market_data(path: str, modified_ns: int) -> pd.DataFrame:
    del modified_ns
    return load_daily_data(path)


@st.cache_data(show_spinner=False)
def cached_evaluations():
    return (load_price_evaluations(), load_volatility_evaluations())


def _fmt_price(value) -> str:
    return "—" if pd.isna(value) else f"₹{float(value):,.2f}"


def _fmt_pct(value, digits=2) -> str:
    return "—" if pd.isna(value) else f"{float(value):+.{digits}f}%"


def _freshness(latest: pd.Timestamp) -> tuple[str, int]:
    age = max(0, (date.today() - latest.date()).days)
    if age == 0:
        return "Latest available observation is from today", age
    return f"Latest available observation is {age} calendar day{'s' if age != 1 else ''} old", age


def _plot_layout(fig: go.Figure, height=360, legend=True):
    fig.update_layout(height=height, margin=dict(l=12, r=12, t=26, b=12), paper_bgcolor="#fff", plot_bgcolor="#fff",
        font=dict(family="Inter,-apple-system,Segoe UI,sans-serif", size=11, color="#555"),
        hoverlabel=dict(bgcolor="#fff", bordercolor="#dfe3e8", font_size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1) if legend else dict(visible=False))
    fig.update_xaxes(showgrid=False, linecolor="#e6e8eb", tickfont=dict(color="#777"), zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#f0f1f3", linecolor="#e6e8eb", tickfont=dict(color="#777"), zeroline=False)
    return fig


def _load_data() -> pd.DataFrame:
    try:
        return cached_market_data(str(DEFAULT_DATA_PATH), DEFAULT_DATA_PATH.stat().st_mtime_ns)
    except FileNotFoundError:
        st.error(f"Price cache not found: `{DEFAULT_DATA_PATH}`. Use Refresh data after installing yfinance.")
        st.stop()
    except Exception as exc:
        st.error(f"Could not validate price cache: {type(exc).__name__}: {exc}")
        st.stop()


def _topbar(data: pd.DataFrame):
    latest = data.index[-1]
    freshness, age = _freshness(latest)
    brand, ticker, date_col, fresh, refresh = st.columns([3.0, 1.0, 1.35, 2.1, 1.25], vertical_alignment="center")
    with brand:
        st.markdown("<div class='toolbar'><span style='color:#387ed1;font-weight:750'>▦ SMART-QUANT</span><span style='color:#9aa0a6'> &nbsp; MARKET ANALYTICS</span></div>", unsafe_allow_html=True)
    with ticker:
        st.markdown(f"<div class='toolbar'><b>{TICKER}</b> <span style='color:#777'>· NSE</span></div>", unsafe_allow_html=True)
    with date_col:
        st.markdown(f"<div class='toolbar'><span class='kicker'>Latest data</span><br><b>{latest:%d %b %Y}</b></div>", unsafe_allow_html=True)
    with fresh:
        color = "#26a69a" if age <= 3 else ("#d8913d" if age <= 7 else "#df514c")
        st.markdown(f"<div class='toolbar'><span style='color:{color}'>●</span> <span style='color:#777'>{freshness}</span></div>", unsafe_allow_html=True)
    with refresh:
        if st.button("↻ Refresh data", use_container_width=True, help="Download auto-adjusted daily OHLCV"):
            try:
                with st.spinner("Refreshing the shared daily CSV…"):
                    refreshed = refresh_daily_data()
                cached_market_data.clear()
                cached_evaluations.clear()
                st.session_state.pop("forecast_cache", None)
                st.session_state.pop("last_price_run", None)
                st.session_state.pop("last_garch_run", None)
                st.session_state.pop("latest_model_status", None)
                st.session_state.pop("latest_successful_forecast", None)
                st.session_state["latest_execution_status"] = "Idle"
                st.session_state["last_refresh_message"] = f"Cache updated through {refreshed.index[-1]:%d %b %Y}."
                st.rerun()
            except Exception as exc:
                st.error(f"Refresh failed; existing cache was kept. {type(exc).__name__}: {exc}")
    message = st.session_state.pop("last_refresh_message", None)
    if message:
        st.success(message)
    status = st.session_state.get("latest_execution_status", "Idle")
    st.caption(f"Model execution status: **{status}** · Latest market observation: {latest:%Y-%m-%d} · Forecast-generation timestamps are recorded separately.")


def _build_overview_figure(data: pd.DataFrame) -> go.Figure:
    close = data["Close"].astype(float)
    returns = np.log(close / close.shift(1)) * 100
    rolling_vol = returns.rolling(21, min_periods=15).std()
    drawdown = (close / close.cummax() - 1) * 100
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[.53, .16, .16, .15], vertical_spacing=.04,
        subplot_titles=("Adjusted close · INR", "Volume · shares", "21-session realized daily volatility · %-points", "Drawdown · %"))
    fig.add_trace(go.Scatter(x=close.index, y=close, mode="lines", name="Adjusted close", line=dict(color="#333", width=1.6), hovertemplate="%{x|%d %b %Y}<br>₹%{y:,.2f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Bar(x=data.index, y=data["Volume"], name="Volume", marker_color="#b9c9da", hovertemplate="%{x|%d %b %Y}<br>%{y:,.0f} shares<extra></extra>"), row=2, col=1)
    fig.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol, mode="lines", name="21-session volatility", line=dict(color="#d8913d", width=1.3), hovertemplate="%{x|%d %b %Y}<br>%{y:.3f} %-points<extra></extra>"), row=3, col=1)
    fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown, mode="lines", name="Drawdown", fill="tozeroy", line=dict(color="#80868b", width=1), fillcolor="rgba(128,134,139,.10)", hovertemplate="%{x|%d %b %Y}<br>%{y:.2f}%<extra></extra>"), row=4, col=1)
    fig.update_layout(showlegend=False, hovermode="x unified", dragmode="zoom")
    fig.update_annotations(font=dict(size=10, color="#7a828a"), xanchor="left", x=.01)
    fig.update_yaxes(title_text="₹", row=1, col=1)
    fig.update_yaxes(title_text="Vol", row=2, col=1)
    fig.update_yaxes(title_text="%-pts", row=3, col=1)
    fig.update_yaxes(title_text="%", row=4, col=1)
    return _plot_layout(fig, 690, legend=False)


def page_overview(data: pd.DataFrame):
    st.title("Market overview")
    close = data["Close"].astype(float)
    returns = close.pct_change() * 100
    log_returns = np.log(close / close.shift(1)) * 100
    vol21 = log_returns.rolling(21, min_periods=15).std().iloc[-1]
    freshness, _ = _freshness(data.index[-1])
    columns = st.columns(5)
    values = [
        ("Latest adjusted close", _fmt_price(close.iloc[-1]), None),
        ("Daily return", _fmt_pct(returns.iloc[-1]), returns.iloc[-1]),
        ("5-session return", _fmt_pct((close.iloc[-1] / close.iloc[-6] - 1) * 100), (close.iloc[-1] / close.iloc[-6] - 1) * 100),
        ("21-session return", _fmt_pct((close.iloc[-1] / close.iloc[-22] - 1) * 100), (close.iloc[-1] / close.iloc[-22] - 1) * 100),
        ("21-session realized daily σ", f"{vol21:.3f} %-pts" if pd.notna(vol21) else "—", None),
    ]
    for col, (label, value, delta) in zip(columns, values):
        with col:
            st.metric(label, value, delta=(f"{delta:+.2f}%" if delta is not None and pd.notna(delta) else None))
    st.caption(f"{freshness}. `Close` is treated as adjusted close, consistent with the cached auto-adjusted series. Historical volatility is the trailing 21-observation standard deviation of daily log returns; not annualized.")
    st.plotly_chart(_build_overview_figure(data), use_container_width=True, config={"displaylogo": False, "scrollZoom": True, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})


def _forecast_cache() -> dict:
    return st.session_state.setdefault("forecast_cache", {})


def _run_models(data: pd.DataFrame, horizon: int, models: tuple[str, ...], kind: str) -> dict[str, dict]:
    cache = _forecast_cache()
    origin = data.index[-1].date().isoformat()
    output, needed = {}, []
    for model in models:
        key = (kind, origin, horizon, model)
        if key in cache:
            output[model] = cache[key]
        else:
            needed.append(model)
    messages = {"arima": "Fitting ARIMA with its saved order…", "xgboost": "Loading saved XGBoost model or fitting its existing configuration…", "lstm": "Training the existing LSTM seed ensemble…", "garch": "Fitting the existing GARCH specification…"}
    if needed:
        with st.status("Generating requested forecasts", expanded=True) as status:
            for model in needed:
                status.write(messages.get(model, f"Running {model}…"))
                result = run_forecasts(data, horizon, (model,))[model]
                cache[(kind, origin, horizon, model)] = result
                output[model] = result
                st.session_state.setdefault("latest_model_status", {})[model] = result.get("status", "error")
                if result.get("status") == "success":
                    st.session_state["latest_execution_status"] = "Success"
                    st.session_state["latest_successful_forecast"] = result.get("execution_timestamp")
                else:
                    st.session_state["latest_execution_status"] = "Error"
            status.update(label="Forecast run complete", state="complete", expanded=False)
    return output


def _forecast_table(results: dict[str, dict], horizon: int) -> pd.DataFrame:
    rows = []
    for name, result in results.items():
        rows.append({
            "Model": result.get("model", name.upper()),
            "Origin": result.get("forecast_origin_date"),
            "Target": result.get("target_date") or f"T+{horizon} sessions · date unavailable",
            "Horizon": result.get("horizon_trading_days", horizon),
            "Return (%)": result.get("predicted_return_pct"),
            "Predicted close (₹)": result.get("predicted_close"),
            "Status": result.get("status", "error"),
            "Duration (s)": result.get("duration_seconds"),
            "Error": result.get("error"),
        })
    return pd.DataFrame(rows)


def _forecast_chart(data: pd.DataFrame, results: dict[str, dict], horizon: int) -> go.Figure:
    history = data["Close"].tail(180)
    date_labels = [stamp.strftime("%Y-%m-%d") for stamp in history.index]
    target_label = f"T+{horizon} sessions"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=date_labels, y=history.values, mode="lines", name="Historical adjusted close", line=dict(color="#333", width=1.5), hovertemplate="%{x}<br>₹%{y:,.2f}<extra></extra>"))
    for name in PRICE_MODELS:
        result = results.get(name)
        if not result or result.get("status") != "success" or result.get("predicted_close") is None:
            continue
        display_name = result.get("model", name.upper())
        fig.add_trace(go.Scatter(x=[target_label], y=[result["predicted_close"]], mode="markers", name=f"{display_name} endpoint", marker=dict(color=MODEL_COLORS[display_name], size=12, symbol="diamond", line=dict(color="#fff", width=1.5)), customdata=[[result.get("predicted_return_pct"), result.get("forecast_origin_date")]], hovertemplate="Endpoint only<br>₹%{y:,.2f}<br>Return %{customdata[0]:+.2f}%<br>Origin %{customdata[1]}<extra>%{fullData.name}</extra>"))
    fig.update_xaxes(type="category", categoryorder="array", categoryarray=date_labels + [target_label], tickangle=-25, title="Observed dates · forecast is shown at T+h session offset")
    fig.update_yaxes(title="Adjusted close · INR")
    fig.update_layout(hovermode="x unified", dragmode="zoom")
    return _plot_layout(fig, 470)


def page_forecasts(data: pd.DataFrame):
    st.title("Forecasting engine")
    st.caption("New forecasts run only when requested. Page navigation reuses the session results.")
    left, middle, action = st.columns([1.0, 1.5, 1.0], vertical_alignment="bottom")
    with left:
        horizon = st.selectbox("Horizon · trading days", HORIZONS, index=1, key="price_horizon")
    with middle:
        selected = st.selectbox("Price models", ["All three", "ARIMA", "XGBoost", "LSTM"], key="price_model")
    with action:
        run_clicked = st.button("Run forecast", type="primary", use_container_width=True, key="run_price_forecast")
    horizon = validate_horizon(horizon)
    model_names = PRICE_MODELS if selected == "All three" else (selected.lower(),)
    if run_clicked:
        results = _run_models(data, horizon, model_names, "price")
        st.session_state["last_price_run"] = {"origin": data.index[-1].date().isoformat(), "horizon": horizon, "models": model_names, "results": results}
    saved_run = st.session_state.get("last_price_run")
    if saved_run and (saved_run["horizon"] != horizon or tuple(saved_run["models"]) != tuple(model_names)):
        st.info(f"Showing the last completed run ({saved_run['horizon']} sessions · {', '.join(saved_run['models'])}). Press Run forecast to generate the currently selected configuration.")
    if not saved_run or saved_run["origin"] != data.index[-1].date().isoformat():
        st.info("Select a horizon and run a forecast. No historical evaluation is retrained here.")
        return
    results, run_horizon = saved_run["results"], int(saved_run["horizon"])
    valid = [item for item in results.values() if item.get("status") == "success" and item.get("predicted_return_pct") is not None and item.get("predicted_close") is not None]
    chart_col, table_col = st.columns([1.7, 1.05], gap="large")
    with chart_col:
        st.subheader("History and forecast endpoints")
        st.plotly_chart(_forecast_chart(data, results, run_horizon), use_container_width=True, config={"displaylogo": False, "scrollZoom": True})
    with table_col:
        st.subheader("Forecast comparison")
        table = _forecast_table(results, run_horizon)
        st.dataframe(table, use_container_width=True, hide_index=True, height=300,
            column_config={"Return (%)": st.column_config.NumberColumn(format="%+.2f%%"), "Predicted close (₹)": st.column_config.NumberColumn(format="₹%.2f"), "Duration (s)": st.column_config.NumberColumn(format="%.2f")})
        if valid:
            returns = [float(item["predicted_return_pct"]) for item in valid]
            consensus = float(np.mean(returns))
            spread = float(max(returns) - min(returns)) if len(returns) > 1 else 0.0
            origin = float(valid[0]["origin_close"])
            m1, m2 = st.columns(2)
            m1.metric("Experimental consensus", f"{consensus:+.2f}%", help="Equal-weight mean across successful price forecasts.")
            m2.metric("Model disagreement", f"{spread:.2f} pp", help="Spread between the largest and smallest successful forecast returns.")
            st.caption(f"Experimental consensus close: ₹{origin * (1 + consensus / 100):,.2f}. Equal weighting is a summary, not an established improvement.")
        if any(item.get("error") for item in results.values()):
            st.warning("At least one model failed. Successful outputs remain available above.")
    with st.expander("Forecast integrity and run details"):
        st.write(f"Origin: **{saved_run['origin']}** · Horizon: **{run_horizon} trading days**")
        st.write("No exact future target date is assigned because an NSE holiday calendar is not configured. The chart marks T+h endpoints without drawing a future path.")
        st.json(results, expanded=False)


def page_performance():
    st.title("Model performance & backtesting")
    price_metrics, price_predictions = cached_evaluations()[0]
    if price_metrics.empty:
        st.warning("No saved price-model metrics were found in the configured results folders.")
        return
    model_col, horizon_col = st.columns([1.2, 1.0])
    models = sorted(price_metrics["model"].dropna().unique().tolist())
    with model_col:
        selected_models = st.multiselect("Model filter", models, default=models)
    with horizon_col:
        horizon = st.selectbox("Horizon · trading days", HORIZONS, index=1, key="performance_horizon")
    filtered = price_metrics.loc[price_metrics["model"].isin(selected_models) & (price_metrics["horizon"] == horizon)].copy()
    st.caption("Saved historical walk-forward evaluation. These metrics are distinct from new forecasts. Lower price MAE/RMSE indicates lower error on this sample; overlapping multi-day horizons make observations dependent.")
    columns = ["model", "horizon", "mae", "rmse", "directional_accuracy_pct", "return_correlation", "valid_predictions", "failed_forecasts", "evaluation_start", "evaluation_end", "naive_mae", "momentum_mae"]
    renamed = {"model": "Model", "horizon": "Horizon", "mae": "MAE · INR", "rmse": "RMSE · INR", "directional_accuracy_pct": "Directional accuracy · %", "return_correlation": "Return correlation", "valid_predictions": "Valid forecasts", "failed_forecasts": "Failed forecasts", "evaluation_start": "Evaluation start", "evaluation_end": "Evaluation end", "naive_mae": "Naive MAE · INR", "momentum_mae": "Momentum MAE · INR"}
    st.dataframe(filtered[[c for c in columns if c in filtered]].rename(columns=renamed), use_container_width=True, hide_index=True,
        column_config={"MAE · INR": st.column_config.NumberColumn(format="₹%.2f"), "RMSE · INR": st.column_config.NumberColumn(format="₹%.2f"), "Naive MAE · INR": st.column_config.NumberColumn(format="₹%.2f"), "Momentum MAE · INR": st.column_config.NumberColumn(format="₹%.2f"), "Directional accuracy · %": st.column_config.NumberColumn(format="%.2f%%"), "Return correlation": st.column_config.NumberColumn(format="%.3f")})
    if price_predictions.empty or not selected_models:
        st.info("Saved prediction rows are unavailable for this selection.")
        return
    rows = price_predictions.loc[price_predictions["model"].isin(selected_models) & (price_predictions["horizon"] == horizon)].dropna(subset=["target_date", "actual_close"])
    if rows.empty:
        st.info("No saved actual-versus-predicted rows match the selected filters.")
        return
    actual = rows.drop_duplicates("target_date").sort_values("target_date")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual["target_date"], y=actual["actual_close"], name="Actual", line=dict(color="#333", width=1.7)))
    for name, group in rows.groupby("model"):
        fig.add_trace(go.Scatter(x=group["target_date"], y=group["predicted_close"], name=name, mode="lines", line=dict(color=MODEL_COLORS.get(name, "#80868b"), width=1.25)))
    for column, label in (("naive_close", "Naive"), ("momentum_close", "Momentum")):
        if column in rows and rows[column].notna().any():
            baseline = rows.drop_duplicates("target_date").sort_values("target_date")
            fig.add_trace(go.Scatter(x=baseline["target_date"], y=baseline[column], name=label, mode="lines", line=dict(color="#9aa0a6", width=1, dash="dot")))
    fig.update_layout(hovermode="x unified", yaxis_title="Adjusted close · INR", xaxis_title="Evaluation target date")
    st.subheader("Actual versus predicted")
    st.plotly_chart(_plot_layout(fig, 390), use_container_width=True, config={"displaylogo": False, "scrollZoom": True})
    error_col, time_col = st.columns(2)
    with error_col:
        error_fig = go.Figure()
        for name, group in rows.groupby("model"):
            error_fig.add_trace(go.Histogram(x=group["error"], name=name, opacity=.62, marker_color=MODEL_COLORS.get(name, "#80868b"), nbinsx=32))
        error_fig.update_layout(barmode="overlay", xaxis_title="Forecast error · INR", yaxis_title="Count", title="Forecast error distribution")
        st.plotly_chart(_plot_layout(error_fig, 315), use_container_width=True, config={"displaylogo": False})
    with time_col:
        error_fig = go.Figure()
        for name, group in rows.groupby("model"):
            error_fig.add_trace(go.Scatter(x=group["target_date"], y=group["error"], name=name, mode="lines", line=dict(color=MODEL_COLORS.get(name, "#80868b"), width=1.1)))
        error_fig.add_hline(y=0, line_color="#aeb4ba", line_dash="dot")
        error_fig.update_layout(title="Forecast error over time", xaxis_title="Target date", yaxis_title="Predicted − actual · INR")
        st.plotly_chart(_plot_layout(error_fig, 315), use_container_width=True, config={"displaylogo": False})
    sources = ", ".join(sorted(filtered["source"].dropna().unique()))
    st.caption(f"Sources: {sources}. Comparisons are descriptive; no statistical-significance claim is made here.")


def page_volatility(data: pd.DataFrame):
    st.title("Volatility & risk analytics")
    st.caption("GARCH models conditional volatility; it is not a fourth price-direction model. No portfolio VaR or CVaR is calculated.")
    horizon_col, unit_col, action_col = st.columns([1.0, 1.5, 1.0], vertical_alignment="bottom")
    with horizon_col:
        horizon = st.selectbox("Horizon · trading days", HORIZONS, index=1, key="risk_horizon")
    with unit_col:
        st.markdown("<div class='kicker'>Forecast units</div><div style='padding-top:.5rem'>Cumulative horizon return %-points</div>", unsafe_allow_html=True)
    with action_col:
        run_clicked = st.button("Run GARCH forecast", type="primary", use_container_width=True, key="run_garch_forecast")
    horizon = validate_horizon(horizon)
    if run_clicked:
        result = _run_models(data, horizon, ("garch",), "volatility")["garch"]
        st.session_state["last_garch_run"] = {"origin": data.index[-1].date().isoformat(), "horizon": horizon, "result": result}
    recent = st.session_state.get("last_garch_run")
    if recent and recent["origin"] == data.index[-1].date().isoformat():
        result = recent["result"]
        if result.get("status") == "success":
            m1, m2, m3 = st.columns(3)
            m1.metric("GARCH cumulative volatility", f"{float(result['forecast_volatility']):.3f} %-pts")
            m2.metric("Forecast variance", f"{float(result['forecast_variance']):.3f} %-pts²")
            m3.metric("Forecast origin", result["forecast_origin_date"])
            st.caption(f"{result['horizon_trading_days']}-session forecast · {result.get('model_version')} · square root of summed daily forecast variances · not annualized")
        else:
            st.error(result.get("error", "GARCH forecast failed."))
    metrics, predictions = cached_evaluations()[1]
    if metrics.empty:
        st.warning("No saved volatility evaluation files were found.")
        return
    selected_metrics = metrics.loc[pd.to_numeric(metrics["horizon"], errors="coerce") == horizon]
    source = selected_metrics["source"].iloc[0] if not selected_metrics.empty else "Saved results"
    st.subheader(f"Saved volatility evaluation · {horizon} sessions")
    st.caption(f"{source}. Historical evaluation only; MAE/RMSE use cumulative return percentage points.")
    columns = [c for c in ("model", "horizon", "n_predictions", "MAE", "RMSE", "mean_QLIKE", "source") if c in selected_metrics]
    st.dataframe(selected_metrics[columns].rename(columns={"model": "Model", "horizon": "Horizon", "n_predictions": "Valid forecasts", "MAE": "MAE · %-pts", "RMSE": "RMSE · %-pts", "mean_QLIKE": "Mean QLIKE", "source": "Saved evaluation"}), use_container_width=True, hide_index=True,
        column_config={"MAE · %-pts": st.column_config.NumberColumn(format="%.4f"), "RMSE · %-pts": st.column_config.NumberColumn(format="%.4f"), "Mean QLIKE": st.column_config.NumberColumn(format="%.4f")})
    if not predictions.empty and "horizon" in predictions:
        selected = predictions.loc[pd.to_numeric(predictions["horizon"], errors="coerce") == horizon].copy()
    else:
        selected = pd.DataFrame()
    if not selected.empty and "model" in selected:
        selected["target_date"] = pd.to_datetime(selected["target_date"], errors="coerce")
        selected = selected.dropna(subset=["target_date"])
        fig = go.Figure()
        if "realized_volatility" in selected:
            actual = selected.drop_duplicates("target_date").sort_values("target_date")
            fig.add_trace(go.Scatter(x=actual["target_date"], y=actual["realized_volatility"], name="Realized volatility", line=dict(color="#333", width=1.6)))
        for name, group in selected.groupby("model"):
            column = "forecast_volatility" if "forecast_volatility" in group else "garch_forecast_volatility"
            if column in group and group[column].notna().any():
                fig.add_trace(go.Scatter(x=group["target_date"], y=group[column], name=name, mode="lines", line=dict(color=MODEL_COLORS.get(name, "#80868b"), width=1.15)))
            elif "historical_forecast_volatility" in group:
                fig.add_trace(go.Scatter(x=group["target_date"], y=group["historical_forecast_volatility"], name=name, mode="lines", line=dict(color=MODEL_COLORS.get(name, "#80868b"), width=1.15)))
        fig.update_layout(yaxis_title="Cumulative return volatility · %-points", xaxis_title="Evaluation target date", hovermode="x unified", title="Forecast versus realized volatility")
        st.plotly_chart(_plot_layout(fig, 390), use_container_width=True, config={"displaylogo": False, "scrollZoom": True})
    returns = np.log(data["Close"] / data["Close"].shift(1)) * 100
    rolling = returns.rolling(21, min_periods=15).std()
    hist_fig = go.Figure(go.Scatter(x=rolling.index, y=rolling, mode="lines", name="21-session rolling daily σ", line=dict(color="#d8913d", width=1.35), hovertemplate="%{x|%d %b %Y}<br>%{y:.3f} %-points<extra></extra>"))
    hist_fig.update_layout(title="Historical 21-session rolling daily volatility", yaxis_title="Daily return σ · %-points", xaxis_title="Date")
    st.plotly_chart(_plot_layout(hist_fig, 315, legend=False), use_container_width=True, config={"displaylogo": False, "scrollZoom": True})
    st.info("The latest saved GARCH v4 evaluation is shown when available; its source script is archived. New forecasts use the active GARCH v2 implementation. The two results are labelled separately.")


def page_diagnostics(data: pd.DataFrame):
    st.title("Diagnostics & system monitor")
    try:
        quality = inspect_data_file(DEFAULT_DATA_PATH)
    except Exception as exc:
        quality = {"error": f"{type(exc).__name__}: {exc}"}
    a, b, c, d = st.columns(4)
    a.metric("Valid observations", f"{len(data):,}")
    b.metric("Data start", f"{data.index[0]:%Y-%m-%d}")
    c.metric("Data end", f"{data.index[-1]:%Y-%m-%d}")
    d.metric("Data file size", f"{DEFAULT_DATA_PATH.stat().st_size // 1024} KB")
    if "error" in quality:
        st.error(quality["error"])
    else:
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Raw rows", f"{quality['rows_raw']:,}")
        q2.metric("Dropped rows", f"{quality['dropped_rows']:,}")
        q3.metric("Duplicate dates", str(quality["duplicate_dates"]))
        q4.metric("Invalid dates", str(quality["invalid_dates"]))
        if quality["missing_columns"] or quality["non_numeric_values"] or quality["duplicate_dates"] or quality["invalid_dates"]:
            st.warning(f"Validation: missing columns={quality['missing_columns']}; nonnumeric={quality['non_numeric_values']}; duplicate dates={quality['duplicate_dates']}; invalid dates={quality['invalid_dates']}.")
        else:
            st.success("CSV schema, dates, and numeric OHLCV values passed validation.")
    st.subheader("Model availability and provenance")
    packages = {"ARIMA": ("statsmodels",), "XGBoost": ("xgboost", "sklearn"), "LSTM": ("tensorflow", "sklearn"), "GARCH": ("arch",)}
    sources = {"ARIMA": PROJECT_ROOT / "models" / "arima_model.py", "XGBoost": PROJECT_ROOT / "models" / "xgboost_model_v2.py", "LSTM": PROJECT_ROOT / "models" / "lstm_model_v2.py", "GARCH": PROJECT_ROOT / "models" / "garch_model_v2.py"}
    rows = []
    for name in ALL_MODELS:
        display = "XGBoost" if name == "xgboost" else name.upper()
        missing = [package for package in packages[display] if importlib.util.find_spec(package) is None]
        source = sources[display]
        rows.append({"Model": display, "Source": str(source.relative_to(PROJECT_ROOT)), "Source exists": source.is_file(), "Required packages": ", ".join(packages[display]), "Missing packages": ", ".join(missing) or "None", "Latest session status": st.session_state.get("latest_model_status", {}).get(name, "Not run")})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    if not importlib.util.find_spec("tensorflow"):
        st.warning("TensorFlow is not available in this Python environment. LSTM forecasts need the dependencies in requirements.txt.")
    st.subheader("Saved evaluation and model files")
    st.dataframe(evaluation_file_inventory(), use_container_width=True, hide_index=True)
    order_path = EVALUATION_DIRS["ARIMA"] / "metrics.csv"
    if order_path.is_file():
        orders = pd.read_csv(order_path)
        if {"horizon", "selected_order"}.issubset(orders.columns):
            st.markdown("**ARIMA selected orders (saved evaluation)**")
            st.dataframe(orders[["horizon", "selected_order"]].rename(columns={"horizon": "Horizon · sessions", "selected_order": "ARIMA order"}), hide_index=True, use_container_width=True)
    xgb_meta = EVALUATION_DIRS["XGBoost"] / "run_metadata.json"
    if xgb_meta.is_file():
        metadata = json.loads(xgb_meta.read_text(encoding="utf-8"))
        with st.expander("XGBoost run metadata"):
            st.json({key: metadata.get(key) for key in ("ticker", "start", "last_data_date", "horizons", "validation_days", "retrain_every", "objective", "seed")})
    lstm_meta = EVALUATION_DIRS["LSTM"] / "lstm_5d_metadata.json"
    if lstm_meta.is_file():
        metadata = json.loads(lstm_meta.read_text(encoding="utf-8"))
        with st.expander("LSTM saved configuration"):
            st.json({key: metadata.get(key) for key in ("ticker", "horizon_trading_days", "lookback_days", "n_folds", "n_seeds", "protocol", "warning")})
            st.caption("The LSTM script saves evaluation metadata and predictions, not reusable neural-network weights.")
    st.markdown("<div class='page-note'>GARCH v4 source is under <code>models archived/</code> and remains a historical experiment. Current on-demand volatility forecasts use active <code>models/garch_model_v2.py</code>. Archived model code and research outputs are not modified.</div>", unsafe_allow_html=True)
    st.subheader("Latest session execution")
    status = st.session_state.get("latest_execution_status", "Idle")
    st.write(f"Status: **{status}** · Latest successful forecast timestamp: **{st.session_state.get('latest_successful_forecast', 'None in this session')}**")
    with st.expander("Raw data validation report"):
        st.json(quality)


def run_dashboard():
    data = _load_data()
    _topbar(data)
    page_names = ["Overview", "Forecasts", "Model Performance", "Volatility & Risk", "Diagnostics"]
    page = st.radio("Dashboard page", page_names, horizontal=True, label_visibility="collapsed", key="dashboard_page")
    st.divider()
    if page == "Overview":
        page_overview(data)
    elif page == "Forecasts":
        page_forecasts(data)
    elif page == "Model Performance":
        page_performance()
    elif page == "Volatility & Risk":
        page_volatility(data)
    else:
        page_diagnostics(data)
