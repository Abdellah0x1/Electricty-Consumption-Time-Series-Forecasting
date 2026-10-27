# Electricity Consumption Time-Series Forecasting

University project: forecasting electricity consumption with statistical and machine learning / deep learning models.
See [PROJECT_PLAN.md](PROJECT_PLAN.md) for goals, datasets, task split and timeline.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

Python 3.11 or 3.12 is recommended.

## Data

Put the raw files in `data/`:

| File | Source |
|---|---|
| `PJME_hourly.csv` | <https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption> |
| `Tetuan City power consumption.csv` | <https://archive.ics.uci.edu/dataset/849/power+consumption+of+tetouan+city> |

Then create the cleaned files in `data/processed/`:

```bash
python -m src.data
```

## Usage

Run commands from the project root. In notebooks (in `notebooks/`), add the root to the path first:

```python
import sys; sys.path.append("..")
from src.data import load_pjme, load_tetouan
from src.evaluate import evaluate, save_results, split
```

Every model is a function `fit_predict(train, horizon, exog_future=None)` that returns `horizon` predictions:

```python
def my_model(train, horizon, exog_future=None):
    ...                       # fit on train["y"] (+ exogenous columns), forecast horizon steps
    return predictions

result = evaluate(my_model, "pjme_hourly", period="val")    # tune on "val", report on "test"
save_results(result, "my_model", notes="first try")         # appends to results/results.csv
result["forecasts"]                                          # origin, datetime, y_true, y_pred
```

Tasks: `pjme_hourly` (24 h ahead), `pjme_monthly` (12 months ahead), `tetouan_hourly` (24 h ahead, with weather).
Use `use_exog=False` to evaluate Tetouan without weather, and `horizon=168` for week-ahead forecasts.

Seasonal naive baseline (the score to beat):

```bash
python -m src.evaluate
```

| Task | Period | MAE | RMSE | MAPE |
|---|---|---|---|---|
| pjme_hourly | test | 3544.8 | 4811.4 | 10.89% |
| pjme_monthly | test | 1598.7 | 2164.1 | 5.01% |
| tetouan_hourly | test | 778.7 | 1082.0 | 2.75% |

## Structure

```
data/            raw data (never edited)  ·  data/processed/ cleaned copies
notebooks/       exploration and experiments
src/data.py      loading + cleaning
src/evaluate.py  splits, rolling-origin evaluation, metrics
results/         results.csv and figures
reports/         report and slides
```
