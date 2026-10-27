"""Single validated loader for the project's cached TCS daily OHLCV data."""
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "TCS.NS_daily.csv"
REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")


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
