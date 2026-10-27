"""Single validated loader and safe refresh path for cached daily OHLCV data."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "TCS.NS_daily.csv"
REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")
TICKER = "TCS.NS"


def load_daily_data(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Read the shared CSV, normalize its index/columns, and validate OHLCV."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Daily price data not found: {path}")
    frame = pd.read_csv(path)
    date_column = next((c for c in frame.columns if c.lower() in {"date", "datetime"}), None)
    if date_column is None:
        first = frame.columns[0]
        if first.lower().startswith("unnamed"):
            date_column = first
        else:
            raise ValueError("CSV must contain a Date or Datetime column.")
    frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
    frame = frame.dropna(subset=[date_column]).set_index(date_column)
    frame.index.name = "Date"
    frame.columns = [str(c).strip().title() for c in frame.columns]
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"CSV is missing required OHLCV columns: {', '.join(missing)}")
    frame = frame.loc[:, REQUIRED_COLUMNS].apply(pd.to_numeric, errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    frame = frame[(frame[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    if frame.empty:
        raise ValueError(f"No valid OHLCV rows found in {path}")
    return frame


def inspect_data_file(path: str | Path = DEFAULT_DATA_PATH) -> dict[str, Any]:
    """Return raw-file validation counts before the loader drops invalid rows."""
    path = Path(path)
    if not path.is_file():
        return {"path": str(path), "exists": False, "error": "CSV file not found."}
    raw = pd.read_csv(path)
    date_col = next((c for c in raw.columns if str(c).strip().lower() in {"date", "datetime"}), None)
    if date_col is None and len(raw.columns) and str(raw.columns[0]).lower().startswith("unnamed"):
        date_col = raw.columns[0]
    date_values = pd.to_datetime(raw[date_col], errors="coerce") if date_col else pd.Series(pd.NaT, index=raw.index)
    dupes = int(date_values.duplicated(keep="last").sum())
    renamed = {str(c).strip().title(): c for c in raw.columns}
    missing = [c for c in REQUIRED_COLUMNS if c not in renamed]
    non_numeric = {}
    for col in REQUIRED_COLUMNS:
        if col in renamed:
            series = pd.to_numeric(raw[renamed[col]], errors="coerce")
            count = int((series.isna() & raw[renamed[col]].notna()).sum())
            if count:
                non_numeric[col] = count
    cleaned = load_daily_data(path)
    return {
        "path": str(path), "exists": True, "rows_raw": int(len(raw)),
        "rows_valid": int(len(cleaned)), "date_column": str(date_col) if date_col else None,
        "invalid_dates": int(date_values.isna().sum()), "duplicate_dates": dupes,
        "missing_columns": missing, "non_numeric_values": non_numeric,
        "latest_date": cleaned.index[-1].date().isoformat(),
        "oldest_date": cleaned.index[0].date().isoformat(),
        "dropped_rows": int(len(raw) - len(cleaned)),
    }


def refresh_daily_data(path: str | Path = DEFAULT_DATA_PATH, ticker: str = TICKER) -> pd.DataFrame:
    """Download auto-adjusted daily OHLCV and atomically replace the shared CSV.

    A failed download or validation leaves the existing cache untouched.
    """
    import yfinance as yf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".refresh.tmp")
    try:
        downloaded = yf.download(ticker, start="2010-01-01", interval="1d",
                                 auto_adjust=True, progress=False, threads=False)
        if downloaded.empty:
            raise RuntimeError(f"No daily data returned for {ticker}.")
        if isinstance(downloaded.columns, pd.MultiIndex):
            downloaded.columns = downloaded.columns.get_level_values(0)
        missing = [column for column in REQUIRED_COLUMNS if column not in downloaded.columns]
        if missing:
            raise ValueError(f"Downloaded data is missing columns: {', '.join(missing)}")
        downloaded = downloaded.loc[:, REQUIRED_COLUMNS].copy()
        downloaded.index = pd.to_datetime(downloaded.index).tz_localize(None)
        downloaded.index.name = "Date"
        downloaded.to_csv(temporary)
        validated = load_daily_data(temporary)
        os.replace(temporary, path)
        return validated
    finally:
        if temporary.exists():
            temporary.unlink()
