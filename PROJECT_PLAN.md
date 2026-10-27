# Electricity Consumption Time-Series Forecasting — Project Plan

**Team:** Partner A (statistical models) · Partner B (machine learning / deep learning)
**Duration:** 6 weeks (adjust week dates to the deadline)

| Week | Dates | Phase |
|---|---|---|
| 1 | ____ | Setup and shared rules |
| 2 | ____ | Data cleaning, EDA and baselines |
| 3 | ____ | Models (part 1) |
| 4 | ____ | Models (part 2) |
| 5 | ____ | Comparison and improvements |
| 6 | ____ | Report and presentation |

---

## 1. Goal

Forecast electricity consumption at two horizons and compare statistical models against machine learning / deep learning models:

- **Short term:** next 24 hours (main) and next 7 days, from hourly data.
- **Long term:** next 12 months, from monthly data.

Research question: *Which model family works best, and how does that depend on the forecast horizon, the amount of data and the availability of weather variables?*

---

## 2. Datasets

### Main dataset — PJME (PJM East, USA)
- File: `data/PJME_hourly.csv` — source: <https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption>
- Hourly load in **MW**, 2002-01-01 → 2018-08-03, 145,366 rows (~16.5 years).
- Seasonalities: daily (24 h), weekly (168 h), yearly.
- No weather → optionally add temperature from Open-Meteo (Philadelphia as proxy): <https://open-meteo.com/en/docs/historical-weather-api>
- Known issues (checked):
  - rows **not sorted** by time
  - **4 duplicate** timestamps (daylight-saving clock change in autumn)
  - **30 missing** hours (spring clock change and gaps)

```python
df = pd.read_csv("data/PJME_hourly.csv", parse_dates=["Datetime"])
df = (df.sort_values("Datetime")
        .groupby("Datetime").mean()
        .asfreq("h")
        .interpolate(limit=3))
monthly = df["PJME_MW"].resample("MS").mean()   # ~199 months for the long-term task
```

### Second dataset — Tetouan City (Morocco)
- File: `data/Tetuan City power consumption.csv` — source: <https://archive.ics.uci.edu/dataset/849/power+consumption+of+tetouan+city>
- 10-minute data, 2017-01-01 → 2017-12-30, 52,416 rows, no gaps (31 Dec not included).
- Weather inputs: temperature, humidity, wind speed, general diffuse flows, diffuse flows.
- Targets: Zone 1, Zone 2, Zone 3 (three distribution areas of the city) → use **Zone 1** or the **total**.
- Resample to hourly (8,736 rows). Only one year → short-term forecasting only.

```python
df = pd.read_csv("data/Tetuan City power consumption.csv")
df["DateTime"] = pd.to_datetime(df["DateTime"], format="%m/%d/%Y %H:%M")
df = df.set_index("DateTime").asfreq("10min")
df.columns = ["temperature", "humidity", "wind_speed",
              "general_diffuse_flows", "diffuse_flows",
              "zone1", "zone2", "zone3"]
hourly = df.resample("h").mean()
```

### Not used
- `Electric_Production.csv` — replaced by monthly PJME (it is a production index, not consumption).
- `household_power_consumption.txt` — single household, very noisy; optional extension only.

---

## 3. Evaluation rules (fixed in week 1, never changed afterwards)

| | PJME hourly | PJME monthly | Tetouan hourly |
|---|---|---|---|
| Train | 2002 → 2016 | 2002-01 → 2015-07 | Jan → Oct |
| Validation | 2017 | 2015-08 → 2016-07 | Nov |
| Test | 2018-01 → 2018-08 | 2016-08 → 2018-07 | Dec (1st → 30th) |
| Horizon | 24 h (main), 168 h | 12 months | 24 h |
| Forecast windows per period | 30 days, spread evenly | 13 origins (val: 1) | 30 days |

These splits are defined in `TASKS` in `src/evaluate.py`. The monthly test period is 24 months so that 12-month forecasts can start from 13 different months.

- **Never shuffle** data. Splits are always chronological.
- **Rolling-origin evaluation**: in addition to the test set, forecast several 24 h / 7-day windows spread across the year (one per season), so scores are not based on a single season.
- **Metrics:** MAE, RMSE, MAPE (+ training time).
- **Baseline to beat:** seasonal naive (same hour last week for hourly; same month last year for monthly).
- **Weather honesty:** models that use measured weather in the test period assume a perfect weather forecast. Train them **with and without** weather and report both.
- Fixed random seeds. All metrics are computed **only** through `src/evaluate.py`.

---

## 4. Project structure

```
├── data/                    # raw data (not modified)
│   └── processed/           # cleaned / resampled files
├── notebooks/
│   ├── 01_eda_pjme.ipynb
│   ├── 02_eda_tetouan.ipynb
│   ├── 03_baselines.ipynb
│   ├── 04_statistical_models.ipynb
│   ├── 05_ml_models.ipynb
│   ├── 06_deep_learning.ipynb
│   └── 07_comparison.ipynb
├── src/
│   ├── data.py              # loading + cleaning for each dataset
│   ├── features.py          # lags, calendar, rolling stats, Fourier terms
│   ├── evaluate.py          # splits, rolling-origin CV, metrics
│   ├── models_stats.py      # Partner A
│   └── models_ml.py         # Partner B
├── results/                 # metrics CSVs and figures
├── reports/                 # report + slides
├── requirements.txt
├── PROJECT_PLAN.md
└── README.md
```

Shared interface — every model must follow it so results are comparable:

```python
def fit_predict(train: pd.DataFrame, horizon: int, exog_future: pd.DataFrame | None = None) -> pd.Series:
    ...
```

---

## 5. Task split and timeline

### Week 1 — Setup (both)
- [ ] Create Git repo (GitHub), add partner, create project board (GitHub Projects / Trello)
- [ ] Folder structure, `requirements.txt` (pandas, numpy, matplotlib, statsmodels, pmdarima, prophet, scikit-learn, xgboost, lightgbm, tensorflow or pytorch)
- [ ] **A:** `src/data.py` — load + clean PJME (sort, duplicates, missing hours, monthly resample)
- [ ] **B:** `src/data.py` — load + clean Tetouan (datetime parsing, column names, hourly resample)
- [ ] **Both:** `src/evaluate.py` — splits, rolling-origin CV, metrics (write together, review carefully)

### Week 2 — EDA and baselines
| Partner A | Partner B |
|---|---|
| EDA PJME: plots by hour / weekday / month / year, trend, holidays, outliers | EDA Tetouan: zone comparison, weather vs consumption correlations, daily/weekly profiles |
| STL decomposition, ACF/PACF, stationarity tests (ADF, KPSS) | Optional: download Open-Meteo temperature for PJME and merge |
| Baselines on both datasets: naive, seasonal naive, moving average | `src/features.py`: lags (1, 2, 24, 168), calendar features, rolling mean/std, holidays |

**Checkpoint:** baselines table in `results/` for all three tasks.

### Weeks 3–4 — Models
| Partner A — Statistical | Partner B — ML / DL |
|---|---|
| Holt-Winters / ETS (monthly PJME) | Linear regression on features (reference) |
| SARIMA (monthly PJME) | Random Forest |
| SARIMAX with Fourier terms (hourly, daily + weekly cycles) + weather exog | XGBoost / LightGBM (with and without weather) |
| Prophet (multiple seasonality + holidays) | LSTM / GRU (sliding windows, scaling, early stopping) |
| TBATS (optional) | Recursive vs direct multi-step forecasting |
| Residual diagnostics (Ljung-Box) | Feature importance (built-in importances / SHAP) |

**Checkpoint (end of week 4):** every model runs through `evaluate.py` on all applicable tasks; results saved as CSV.

### Week 5 — Comparison and improvements (both)
- [ ] Final results table: models × datasets × horizons × metrics
- [ ] Forecast plots (actual vs predicted) with prediction intervals where available
- [ ] Error analysis: which hours / seasons / holidays are hardest?
- [ ] Ensemble of best 2–3 models
- [ ] Optional hybrid: SARIMAX + XGBoost on residuals
- [ ] Check: does the winner on PJME also win on Tetouan?

### Week 6 — Report and presentation
| Partner A | Partner B |
|---|---|
| Report: introduction, datasets, EDA, statistical methods | Report: ML/DL methods, results, discussion, limitations |
| Slides: first half | Slides: second half + optional Streamlit demo |
| README: how to reproduce | Final code clean-up |

**Both:** proofread the whole report, rehearse the presentation.

---

## 6. Working together

- **Branches:** one per task (`feat/sarimax`, `feat/xgboost`, …); merge via pull request reviewed by the other partner.
- **Weekly sync (30 min):** done / blocked / latest numbers.
- **Data:** never edit files in `data/` by hand — all cleaning in `src/data.py`.
- **Notebooks** explore; reusable code goes into `src/`.
- **Results** saved as CSV in `results/` with model name, dataset, horizon, metrics, date.

---

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| SARIMA too slow on hourly data | Use Fourier terms / Prophet for hourly; SARIMA only on monthly |
| LSTM training slow | Use Google Colab GPU; start with small networks |
| Data leakage (future info in features or scaling) | Fit scalers on train only; lag features only use the past; review `features.py` together |
| Tetouan has only one year | Short-term only; state it in limitations |
| Monthly PJME test period short | Add rolling-origin CV on monthly series |
| Falling behind schedule | Week 4 checkpoint: drop optional items (TBATS, hybrid, demo) first |

---

## 8. Deliverables checklist

- [ ] GitHub repo with README (setup + how to reproduce)
- [ ] EDA notebooks for both datasets
- [ ] Baselines + ≥ 3 statistical + ≥ 3 ML/DL models
- [ ] Final comparison table and figures
- [ ] Report (PDF)
- [ ] Presentation slides
- [ ] Optional: Streamlit demo
