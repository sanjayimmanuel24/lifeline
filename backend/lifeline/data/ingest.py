"""Download and prepare every dataset LIFELINE uses.

Run once:  python -m lifeline.data.ingest

Sources (all free, no key needed):
  * NYC Open Data - EMS Incident Dispatch Data (76xm-jjuj), Brooklyn, severity 1-4
  * NYC Open Data - Modified ZCTA polygons (pri4-ifjk)
  * Open-Meteo archive API - hourly precipitation and temperature
  * OpenStreetMap Overpass API - hospitals in Brooklyn
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from lifeline.config import DATA_DIR, TRAIN_START, TEST_END, TEST_START

SOCRATA = "https://data.cityofnewyork.us/resource"
EMS = f"{SOCRATA}/76xm-jjuj.json"
BASE_WHERE = "borough='BROOKLYN' AND final_severity_level_code in('1','2','3','4')"

client = httpx.Client(timeout=120)


def soql(url: str, params: dict, page: int = 50_000) -> list[dict]:
    """Fetch every page of a SoQL query."""
    rows, offset = [], 0
    while True:
        p = dict(params, **{"$limit": page, "$offset": offset})
        for attempt in range(4):
            try:
                r = client.get(url, params=p)
                r.raise_for_status()
                break
            except httpx.HTTPError:
                if attempt == 3:
                    raise
                time.sleep(2 * (attempt + 1))
        batch = r.json()
        rows.extend(batch)
        print(f"  {url.rsplit('/', 1)[-1]}: {len(rows):,} rows")
        if len(batch) < page:
            return rows
        offset += page


def hourly_counts() -> pd.DataFrame:
    rows = soql(EMS, {
        # SoQL has no hourly truncation, so group by day and hour-of-day
        "$select": "date_trunc_ymd(incident_datetime) as d, date_extract_hh(incident_datetime) as hh, "
                   "zipcode, count(*) as n",
        "$where": f"{BASE_WHERE} AND incident_datetime >= '{TRAIN_START}' "
                  f"AND incident_datetime < '{TEST_END}' AND zipcode IS NOT NULL",
        "$group": "d, hh, zipcode",
        "$order": "d, hh, zipcode",
    })
    df = pd.DataFrame(rows)
    df["h"] = pd.to_datetime(df["d"]) + pd.to_timedelta(df["hh"].astype(int), unit="h")
    df["n"] = df["n"].astype(int)
    return df[["h", "zipcode", "n"]]


def travel_stats() -> pd.DataFrame:
    rows = soql(EMS, {
        "$select": "zipcode, date_extract_hh(incident_datetime) as hour, "
                   "avg(incident_travel_tm_seconds_qy) as travel, "
                   "avg(dispatch_response_seconds_qy) as dispatch, count(*) as n",
        "$where": f"{BASE_WHERE} AND valid_incident_rspns_time_indc='Y' "
                  f"AND incident_datetime >= '{TRAIN_START}' AND incident_datetime < '{TEST_START}'",
        "$group": "zipcode, hour",
    })
    df = pd.DataFrame(rows)
    for c in ("hour", "travel", "dispatch", "n"):
        df[c] = df[c].astype(float)
    return df


INCIDENT_FIELDS = ("incident_id, incident_datetime, zipcode, final_severity_level_code, "
                   "first_assignment_datetime, first_on_scene_datetime, first_hosp_arrival_datetime, "
                   "incident_close_datetime, dispatch_response_seconds_qy, incident_travel_tm_seconds_qy, "
                   "incident_response_seconds_qy, valid_incident_rspns_time_indc, held_indicator")


def incidents_on(days: list[str]) -> pd.DataFrame:
    frames = []
    for d in days:
        nxt = (pd.Timestamp(d) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        rows = soql(EMS, {
            "$select": INCIDENT_FIELDS,
            "$where": f"{BASE_WHERE} AND incident_datetime >= '{d}' AND incident_datetime < '{nxt}'",
            "$order": "incident_datetime",
        })
        frames.append(pd.DataFrame(rows))
    df = pd.concat(frames, ignore_index=True)
    for c in ("incident_datetime", "first_assignment_datetime", "first_on_scene_datetime",
              "first_hosp_arrival_datetime", "incident_close_datetime"):
        df[c] = pd.to_datetime(df.get(c))
    for c in ("dispatch_response_seconds_qy", "incident_travel_tm_seconds_qy", "incident_response_seconds_qy"):
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    df["severity"] = df["final_severity_level_code"].astype(int)
    return df


def weather() -> pd.DataFrame:
    r = client.get("https://archive-api.open-meteo.com/v1/archive", params={
        "latitude": 40.65, "longitude": -73.95,
        "start_date": TRAIN_START, "end_date": (pd.Timestamp(TEST_END) - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        "hourly": "precipitation,temperature_2m", "timezone": "America/New_York",
    })
    r.raise_for_status()
    h = r.json()["hourly"]
    return pd.DataFrame({"h": pd.to_datetime(h["time"]), "precip": h["precipitation"], "temp": h["temperature_2m"]})


def zone_polygons(zips: list[str]) -> dict:
    rows = soql(f"{SOCRATA}/pri4-ifjk.json", {"$select": "modzcta, zcta, pop_est, the_geom"}, page=1000)
    features = []
    for r in rows:
        members = [z.strip() for z in r["zcta"].split(",")]
        if r["modzcta"] in zips:
            features.append({"type": "Feature", "geometry": r["the_geom"],
                             "properties": {"zone": r["modzcta"], "members": members,
                                            "pop": int(float(r.get("pop_est") or 0))}})
    return {"type": "FeatureCollection", "features": features}


def zip_to_modzcta() -> dict[str, str]:
    rows = soql(f"{SOCRATA}/pri4-ifjk.json", {"$select": "modzcta, zcta"}, page=1000)
    out = {}
    for r in rows:
        for z in r["zcta"].split(","):
            out[z.strip()] = r["modzcta"]
    return out


def hospitals() -> list[dict]:
    q = """[out:json][timeout:60];
    (node["amenity"="hospital"](40.566,-74.042,40.739,-73.833);
     way["amenity"="hospital"](40.566,-74.042,40.739,-73.833););
    out center tags;"""
    for endpoint in ("https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter",
                     "https://overpass.private.coffee/api/interpreter"):
        try:
            r = client.post(endpoint, data={"data": q}, headers={"User-Agent": "lifeline-student-project/0.1"})
            r.raise_for_status()
            break
        except httpx.HTTPError as e:
            print(f"  {endpoint} failed: {e.__class__.__name__}")
    else:
        print("  WARNING: no Overpass mirror reachable, hospitals layer will be empty")
        return []
    out = []
    for e in r.json()["elements"]:
        lat = e.get("lat") or e.get("center", {}).get("lat")
        lon = e.get("lon") or e.get("center", {}).get("lon")
        name = e.get("tags", {}).get("name")
        if lat and lon and name and e.get("tags", {}).get("emergency", "yes") != "no":
            out.append({"name": name, "lat": lat, "lon": lon})
    return out


def pick_days(wx: pd.DataFrame, start: str, end: str, n: int, n_rainy: int, seed: int) -> list[str]:
    """n days from [start, end): n_rainy of the wettest days plus random others."""
    daily = wx[(wx.h >= start) & (wx.h < end)].groupby(wx.h.dt.date)["precip"].sum()
    rainy = daily.sort_values(ascending=False).index[:n_rainy * 3]
    rng = np.random.default_rng(seed)
    wet = list(rng.choice(rainy, n_rainy, replace=False))
    rest = [d for d in daily.index if d not in wet]
    others = list(rng.choice(rest, n - n_rainy, replace=False))
    return sorted(str(d) for d in wet + others)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    def missing(name: str) -> bool:
        return not (DATA_DIR / name).exists()

    if missing("weather.parquet"):
        print("weather ...")
        weather().to_parquet(DATA_DIR / "weather.parquet")
    wx = pd.read_parquet(DATA_DIR / "weather.parquet")

    z2m = zip_to_modzcta()
    if missing("hourly_counts.parquet"):
        print("hourly counts ...")
        counts = hourly_counts()
        counts["zone"] = counts["zipcode"].map(z2m)
        counts = counts.dropna(subset=["zone"]).groupby(["h", "zone"], as_index=False)["n"].sum()
        totals = counts.groupby("zone")["n"].sum()
        keep = totals[totals >= 2000].index  # drop PO-box / tiny zips
        counts = counts[counts.zone.isin(keep)]
        counts.to_parquet(DATA_DIR / "hourly_counts.parquet")
    counts = pd.read_parquet(DATA_DIR / "hourly_counts.parquet")
    zones = sorted(counts.zone.unique())
    print(f"  {len(zones)} zones, {counts.n.sum():,} incidents")

    if missing("zones.geojson"):
        print("zone polygons ...")
        (DATA_DIR / "zones.geojson").write_text(json.dumps(zone_polygons(zones)))

    if missing("travel_stats.parquet"):
        print("travel stats ...")
        ts = travel_stats()
        ts["zone"] = ts["zipcode"].map(z2m)
        ts[ts.zone.isin(zones)].to_parquet(DATA_DIR / "travel_stats.parquet")

    if missing("hospitals.json"):
        print("hospitals ...")
        (DATA_DIR / "hospitals.json").write_text(json.dumps(hospitals(), indent=1))

    print("calibration days (2025) ...")
    cal_days = pick_days(wx, "2025-01-01", TEST_START, n=12, n_rainy=3, seed=7)
    cal = incidents_on(cal_days)
    cal["zone"] = cal["zipcode"].map(z2m)
    cal[cal.zone.isin(zones)].to_parquet(DATA_DIR / "calibration_incidents.parquet")

    print("held-out test days (2026) ...")
    test_days = pick_days(wx, TEST_START, TEST_END, n=30, n_rainy=6, seed=11)
    test = incidents_on(test_days)
    test["zone"] = test["zipcode"].map(z2m)
    test[test.zone.isin(zones)].to_parquet(DATA_DIR / "test_incidents.parquet")

    (DATA_DIR / "splits.json").write_text(json.dumps(
        {"zones": zones, "calibration_days": cal_days, "test_days": test_days}, indent=1))
    print("done")


if __name__ == "__main__":
    main()
