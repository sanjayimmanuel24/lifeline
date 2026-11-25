"""get_weather tool: Open-Meteo (free, no key). Archive API for replayed days, forecast API for today."""
from __future__ import annotations

from functools import lru_cache

import httpx
import pandas as pd

from lifeline.config import DATA_DIR

LAT, LON = 40.65, -73.95


@lru_cache(maxsize=64)
def _day(date: str) -> pd.DataFrame | None:
    today = pd.Timestamp.now(tz="America/New_York").date()
    api = ("https://api.open-meteo.com/v1/forecast" if pd.Timestamp(date).date() >= today - pd.Timedelta(days=5)
           else "https://archive-api.open-meteo.com/v1/archive")
    try:
        r = httpx.get(api, params={"latitude": LAT, "longitude": LON, "start_date": date, "end_date": date,
                                   "hourly": "precipitation,temperature_2m,weather_code",
                                   "timezone": "America/New_York"}, timeout=15)
        r.raise_for_status()
        h = r.json()["hourly"]
        return pd.DataFrame({"h": pd.to_datetime(h["time"]), "precip": h["precipitation"],
                             "temp": h["temperature_2m"], "code": h.get("weather_code")})
    except Exception:
        return None


@lru_cache(maxsize=1)
def _archive() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "weather.parquet").drop_duplicates("h").set_index("h")


def get_weather(ts: pd.Timestamp, hours: int = 3) -> dict:
    """Precipitation (mm) and temperature for ts .. ts+hours. Falls back to the local archive copy."""
    df = _day(ts.strftime("%Y-%m-%d"))
    source = "open-meteo"
    if df is None or df.empty:
        df, source = _archive().reset_index(), "local-archive"
    sel = df[(df.h >= ts.floor("h")) & (df.h < ts.floor("h") + pd.Timedelta(hours=hours))]
    if len(sel) < hours:
        a = _archive()
        idx = pd.date_range(ts.floor("h"), periods=hours, freq="h")
        sel = a.reindex(idx).ffill().bfill().reset_index(names="h")
        source = "local-archive"
    return {"precip": [float(x) for x in sel.precip], "temp": [float(x) for x in sel.temp], "source": source}
