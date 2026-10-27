"""Smart-Quant Streamlit dashboard entry point.

From the project root run: streamlit run main.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT / "front end"
if str(FRONTEND_DIR) not in sys.path:
    sys.path.insert(0, str(FRONTEND_DIR))

from dashboard_app import run_dashboard

if __name__ == "__main__":
    run_dashboard()
