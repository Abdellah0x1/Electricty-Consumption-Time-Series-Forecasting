"""Shared evaluation: fixed splits, rolling-origin forecasting and metrics.

Every model must be wrapped as a function with this signature:

    def fit_predict(train: pd.DataFrame, horizon: int,
                    exog_future: pd.DataFrame | None = None) -> pd.Series | np.ndarray

- train       : all history before the forecast origin ("y" + optional exogenous columns)
- horizon     : number of steps to forecast
- exog_future : exogenous columns for the forecast window (None if the dataset has none).
                Using it assumes a perfect weather forecast -> report with and without.
- returns     : `horizon` predictions, in time order

All metrics in the project are computed through evaluate() so results are comparable.
"""
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.data import load_pjme, load_tetouan

RESULTS_FILE = Path(__file__).resolve().parents[1] / "results" / "results.csv"
SEED = 42


@dataclass(frozen=True)
class Task:
    load: Callable[[], pd.DataFrame]
    val_start: str      # first timestamp of the validation period
    test_start: str     # first timestamp of the test period (runs to the end of the data)
    horizon: int        # default forecast length in steps
    step: int           # spacing between candidate forecast origins
    n_windows: int      # number of forecast origins used per period
    season: int         # main seasonal period (used by seasonal naive)


TASKS = {
    # train 2002-2016 | val 2017 | test 2018-01 -> 2018-08; next 24 h from midnight
    "pjme_hourly": Task(lambda: load_pjme("h"), "2017-01-01", "2018-01-01",
                        horizon=24, step=24, n_windows=30, season=168),
    # train 2002-01 -> 2015-07 | val 2015-08 -> 2016-07 | test 2016-08 -> 2018-07; next 12 months
    "pjme_monthly": Task(lambda: load_pjme("MS"), "2015-08-01", "2016-08-01",
                         horizon=12, step=1, n_windows=13, season=12),
    # train Jan -> Oct | val Nov | test Dec (1st -> 30th); next 24 h from midnight
    "tetouan_hourly": Task(lambda: load_tetouan("zone1", "h"), "2017-11-01", "2017-12-01",
                           horizon=24, step=24, n_windows=30, season=168),
}


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------- metrics

def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mape(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def compute_metrics(y_true, y_pred) -> dict:
    return {"mae": mae(y_true, y_pred), "rmse": rmse(y_true, y_pred), "mape": mape(y_true, y_pred)}


# ---------------------------------------------------------------- splits

def split(task_name: str, df: pd.DataFrame | None = None):
    """Return (train, val, test) for a task. Chronological, never shuffled."""
    task = TASKS[task_name]
    df = task.load() if df is None else df
    train = df[df.index < task.val_start]
    val = df[(df.index >= task.val_start) & (df.index < task.test_start)]
    test = df[df.index >= task.test_start]
    return train, val, test


def forecast_origins(index: pd.DatetimeIndex, start, end, horizon: int,
                     step: int, n_windows: int) -> list[int]:
    """Positions of forecast origins in [start, end] whose full window fits in the period.

    Candidates are spaced `step` apart from `start`; `n_windows` of them are picked evenly
    across the period so every season of it is represented.
    """
    in_period = np.flatnonzero((index >= start) & (index <= end))
    first, last = in_period[0], in_period[-1]
    candidates = list(range(first, last - horizon + 2, step))
    if not candidates:
        raise ValueError("period is shorter than the forecast horizon")
    if len(candidates) <= n_windows:
        return candidates
    picks = np.linspace(0, len(candidates) - 1, n_windows).round().astype(int)
    return [candidates[i] for i in np.unique(picks)]


# ---------------------------------------------------------------- evaluation

def evaluate(fit_predict, task_name: str, period: str = "test", horizon: int | None = None,
             n_windows: int | None = None, use_exog: bool = True,
             df: pd.DataFrame | None = None) -> dict:
    """Rolling-origin evaluation of `fit_predict` on the validation or test period.

    At each origin the model is refitted on all data before it (for "test" this includes the
    validation period) and forecasts `horizon` steps. Metrics are pooled over all windows.

    Returns a dict with the metrics, run info and a `forecasts` DataFrame
    (origin, datetime, y_true, y_pred) for plotting and error analysis.
    """
    task = TASKS[task_name]
    df = task.load() if df is None else df
    horizon = horizon or task.horizon
    n_windows = n_windows or task.n_windows
    if period == "val":
        start, end = pd.Timestamp(task.val_start), df.index[df.index < task.test_start][-1]
    elif period == "test":
        start, end = pd.Timestamp(task.test_start), df.index[-1]
    else:
        raise ValueError("period must be 'val' or 'test'")

    exog_cols = [c for c in df.columns if c != "y"] if use_exog else []
    data = df[["y"] + exog_cols]

    set_seed()
    frames, t0 = [], time.perf_counter()
    for pos in forecast_origins(data.index, start, end, horizon, task.step, n_windows):
        train = data.iloc[:pos]
        future = data.iloc[pos:pos + horizon]
        exog_future = future[exog_cols] if exog_cols else None

        pred = np.asarray(fit_predict(train, horizon, exog_future), dtype=float).ravel()
        if len(pred) != horizon:
            raise ValueError(f"fit_predict returned {len(pred)} values, expected {horizon}")
        if np.isnan(pred).any():
            raise ValueError("fit_predict returned NaN values")

        frames.append(pd.DataFrame({
            "origin": data.index[pos], "datetime": future.index,
            "y_true": future["y"].to_numpy(), "y_pred": pred,
        }))
    elapsed = time.perf_counter() - t0

    forecasts = pd.concat(frames, ignore_index=True)
    return {
        "task": task_name, "period": period, "horizon": horizon,
        "n_windows": len(frames), "use_exog": bool(exog_cols),
        **compute_metrics(forecasts["y_true"], forecasts["y_pred"]),
        "seconds": round(elapsed, 2),
        "forecasts": forecasts,
    }


def save_results(result: dict, model_name: str, notes: str = "", path: Path = RESULTS_FILE) -> pd.DataFrame:
    """Append one evaluation result (without the forecasts) to results/results.csv."""
    row = {"model": model_name, **{k: v for k, v in result.items() if k != "forecasts"},
           "notes": notes, "run_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}
    row_df = pd.DataFrame([row])
    path.parent.mkdir(parents=True, exist_ok=True)
    row_df.to_csv(path, mode="a", header=not path.exists(), index=False)
    return row_df


if __name__ == "__main__":
    # Sanity check with the reference baseline every model must beat.
    for name, task in TASKS.items():
        def seasonal_naive(train, horizon, exog_future=None, m=task.season):
            return np.resize(train["y"].iloc[-m:].to_numpy(), horizon)

        for period in ("val", "test"):
            r = evaluate(seasonal_naive, name, period)
            print(f"{name:15s} {period:4s} windows={r['n_windows']:2d}  "
                  f"MAE={r['mae']:9.1f}  RMSE={r['rmse']:9.1f}  MAPE={r['mape']:5.2f}%")
