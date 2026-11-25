"""Turn hourly zone counts + weather + calendar into (target, condition) pairs.

target  x_t = counts for zones Z over the next HORIZON hours        (HORIZON, Z)
condition c_t = [hour sin/cos, day-of-week one-hot, weekend, holiday,
                 precipitation and temperature over the horizon,
                 per-zone counts in the previous 3 hours (log1p),
                 total count in the previous 3 hours]
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import holidays
import numpy as np
import pandas as pd

from lifeline.config import DATA_DIR, HORIZON, TEST_END, TEST_START, TRAIN_START, VAL_START

LAG = 3


@dataclass
class Series:
    index: pd.DatetimeIndex
    Y: np.ndarray           # (T, Z) hourly counts
    precip: np.ndarray      # (T,)
    temp: np.ndarray        # (T,)
    zones: list[str]


def load_series() -> Series:
    splits = json.loads((DATA_DIR / "splits.json").read_text())
    zones = splits["zones"]
    counts = pd.read_parquet(DATA_DIR / "hourly_counts.parquet")
    idx = pd.date_range(TRAIN_START, pd.Timestamp(TEST_END) - pd.Timedelta(hours=1), freq="h")
    Y = (counts.pivot_table(index="h", columns="zone", values="n", aggfunc="sum")
         .reindex(index=idx, columns=zones).fillna(0).to_numpy(dtype=np.float32))
    wx = pd.read_parquet(DATA_DIR / "weather.parquet").drop_duplicates("h").set_index("h").reindex(idx)
    wx = wx.interpolate().bfill().ffill()
    return Series(idx, Y, wx.precip.to_numpy(np.float32), wx.temp.to_numpy(np.float32), zones)


_US_HOLIDAYS = holidays.US(years=range(2021, 2028))


def condition(ts: pd.Timestamp, lag_counts: np.ndarray, precip: np.ndarray, temp: np.ndarray) -> np.ndarray:
    """Build one condition vector. lag_counts: (LAG, Z); precip/temp: (HORIZON,)."""
    h = ts.hour + 0.5 * (HORIZON - 1)
    dow = np.zeros(7, np.float32)
    dow[ts.dayofweek] = 1
    cal = [np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
           float(ts.dayofweek >= 5), float(ts.date() in _US_HOLIDAYS)]
    wx = [np.log1p(precip).mean(), np.log1p(precip).max(), temp.mean() / 20.0]
    lag = np.log1p(lag_counts.sum(0))
    tot = np.log1p(lag_counts.sum()) / 5.0
    return np.concatenate([cal, dow, wx, lag, [tot]]).astype(np.float32)


def build_xy(s: Series, start: str, end: str, stride: int = 1) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    pos = np.where((s.index >= start) & (s.index < end))[0]
    pos = pos[(pos >= LAG) & (pos + HORIZON <= len(s.index))][::stride]
    X = np.stack([s.Y[p:p + HORIZON] for p in pos])
    C = np.stack([condition(s.index[p], s.Y[p - LAG:p], s.precip[p:p + HORIZON], s.temp[p:p + HORIZON]) for p in pos])
    return X, C, s.index[pos]


def splits(s: Series):
    return {
        "train": build_xy(s, TRAIN_START, VAL_START),
        "val": build_xy(s, VAL_START, TEST_START),
        "test": build_xy(s, TEST_START, TEST_END),
    }
