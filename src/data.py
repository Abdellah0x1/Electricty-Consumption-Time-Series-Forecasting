"""Loading and cleaning of the project datasets.

Every loader returns a DataFrame with:
- a regular DatetimeIndex (freq set, sorted, no gaps, no duplicates)
- a target column named "y"
- optional exogenous columns (weather for Tetouan)

Raw files in data/ are never modified; use save_processed() to write cleaned copies.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

PJME_FILE = DATA_DIR / "PJME_hourly.csv"
TETOUAN_FILE = DATA_DIR / "Tetuan City power consumption.csv"

TETOUAN_COLUMNS = [
    "temperature", "humidity", "wind_speed",
    "general_diffuse_flows", "diffuse_flows",
    "zone1", "zone2", "zone3",
]
TETOUAN_WEATHER = TETOUAN_COLUMNS[:5]
TETOUAN_ZONES = TETOUAN_COLUMNS[5:]


def load_pjme(freq: str = "h") -> pd.DataFrame:
    """PJM East load in MW.

    freq="h"  -> hourly, 2002-01-01 01:00 to 2018-08-03 00:00
    freq="MS" -> monthly mean, 2002-01 to 2018-07 (incomplete Aug 2018 dropped)

    Cleaning: sort by time, average the 4 duplicate timestamps (autumn DST change),
    insert the 30 missing hours (all single-hour gaps) and fill them by time interpolation.
    """
    df = pd.read_csv(PJME_FILE, parse_dates=["Datetime"])
    y = (
        df.sort_values("Datetime")
        .groupby("Datetime")["PJME_MW"].mean()
        .asfreq("h")
        .interpolate(method="time", limit=3)
    )
    if y.isna().any():
        raise ValueError(f"PJME still has {y.isna().sum()} missing hours after interpolation")

    if freq == "MS":
        counts = y.resample("MS").count()
        y = y.resample("MS").mean()
        full_month_hours = y.index.days_in_month * 24
        y = y[counts >= full_month_hours - 1].asfreq("MS")   # allow the first month to miss 00:00
    elif freq != "h":
        raise ValueError("freq must be 'h' or 'MS'")

    out = y.to_frame("y")
    out.index.name = "datetime"
    return out


def load_tetouan(target: str = "zone1", freq: str = "h", weather: bool = True) -> pd.DataFrame:
    """Tetouan (Morocco) power consumption, 2017-01-01 to 2017-12-30.

    target  : "zone1", "zone2", "zone3" or "total" (sum of the three zones)
    freq    : "10min" (raw) or "h" (hourly mean)
    weather : include the 5 weather columns as exogenous variables
    """
    df = pd.read_csv(TETOUAN_FILE)
    df["DateTime"] = pd.to_datetime(df["DateTime"], format="%m/%d/%Y %H:%M")
    df = df.set_index("DateTime").sort_index()
    df.columns = TETOUAN_COLUMNS
    df = df.asfreq("10min")
    if df.isna().any().any():
        raise ValueError("Tetouan data has gaps; expected a complete 10-minute series")

    if freq == "h":
        df = df.resample("h").mean()
    elif freq != "10min":
        raise ValueError("freq must be 'h' or '10min'")

    if target == "total":
        y = df[TETOUAN_ZONES].sum(axis=1)
    elif target in TETOUAN_ZONES:
        y = df[target]
    else:
        raise ValueError(f"target must be one of {TETOUAN_ZONES + ['total']}")

    out = y.to_frame("y")
    if weather:
        out = out.join(df[TETOUAN_WEATHER])
    out.index.name = "datetime"
    return out


def save_processed() -> None:
    """Write the cleaned datasets used in the project to data/processed/."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    load_pjme("h").to_csv(PROCESSED_DIR / "pjme_hourly.csv")
    load_pjme("MS").to_csv(PROCESSED_DIR / "pjme_monthly.csv")
    load_tetouan("zone1", "h").to_csv(PROCESSED_DIR / "tetouan_zone1_hourly.csv")
    load_tetouan("total", "h").to_csv(PROCESSED_DIR / "tetouan_total_hourly.csv")


if __name__ == "__main__":
    for name, d in {
        "pjme hourly": load_pjme("h"),
        "pjme monthly": load_pjme("MS"),
        "tetouan zone1 hourly": load_tetouan("zone1", "h"),
    }.items():
        print(f"{name:22s} {len(d):7d} rows  {d.index.min()} -> {d.index.max()}  "
              f"freq={d.index.freqstr}  cols={list(d.columns)}")
    save_processed()
    print(f"saved cleaned files to {PROCESSED_DIR}")
