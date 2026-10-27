"""Run the forecasting API and serve the lightweight project frontend.

Start from the project root with: python main.py
"""
from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from forecasting import FORECASTERS, forecast, forecast_all  # noqa: E402

FRONTEND = ROOT / "front end" / "index.html"


class Handler(BaseHTTPRequestHandler):
    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            if not FRONTEND.is_file():
                self._json(404, {"error": f"Frontend not found: {FRONTEND}"})
                return
            body = FRONTEND.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/api/models":
            self._json(200, {"models": list(FORECASTERS), "horizons": [1, 5, 21, 30, 45]})
            return
        if parsed.path == "/api/forecast":
            params = parse_qs(parsed.query)
            try:
                horizon = int(params.get("horizon", ["1"])[0])
                model_name = params.get("model", ["all"])[0]
                result = forecast_all(horizon) if model_name == "all" else forecast(model_name, horizon)
                self._json(200, result)
            except (ValueError, FileNotFoundError) as exc:
                self._json(400, {"error": str(exc)})
            except Exception as exc:
                self._json(500, {"error": f"{type(exc).__name__}: {exc}"})
            return
        self._json(404, {"error": "Not found"})


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    display_host = "127.0.0.1" if host == "0.0.0.0" else host
    print(f"Smart Quant forecast UI: http://{display_host}:{port}")
    print(f"Models: {', '.join(FORECASTERS)}")
    ThreadingHTTPServer((host, port), Handler).serve_forever()
