# Data Scientist Agent 2.0

An autonomous data science agent with a web interface. Upload a CSV, describe what you
want in one sentence, and the agent profiles the data, cleans it, engineers and selects
features, detects the problem type, picks and trains suitable models, compares them,
explains the winner, and then judges honestly whether your objective was actually met.

Built with **Python, FastAPI, HTML, CSS, MySQL and the Gemini API** — nothing else.

---

## What it actually does

```
UPLOAD → UNDERSTAND → CLEAN → FRAME → ENGINEER → SELECT → TRAIN → EVALUATE → VALIDATE → REPORT
```

1. **Dataset understanding** — rows, columns, dtypes, missing values, duplicates,
   cardinality, constant columns, identifier columns, free-text columns, date columns,
   summary statistics, and a 0–100 data quality score.
2. **Data cleaning** — trims whitespace, converts text placeholders (`N/A`, `?`, `-`) to
   real nulls, merges case-variant categories, parses numbers stored as text (`$1,234`),
   parses dates, replaces infinities, removes duplicates, and drops constant, identifier,
   free-text and near-empty columns. Every action is recorded with a reason.
3. **AI reasoning** — a compact *profile* of the dataset (never the raw file) goes to
   Gemini, which returns structured JSON: problem type, target column, recommended models,
   relevant features, columns to drop, leakage risks, evaluation metric, and success
   criteria. Every answer is then validated against the real dataset.
4. **Problem type detection** — decided from the target column's actual shape. The data
   always wins a disagreement with the AI.
5. **Target detection** — if the agent is not confident enough, it stops and asks you which
   column to predict instead of quietly modelling the wrong one.
6. **Feature engineering** — calendar parts from dates, log transforms for skewed columns,
   ratios and differences between the most informative numeric columns, and rare-category
   grouping. Capped and explained, never hundreds of features.
7. **Feature selection** — variance filtering, correlation redundancy, mutual information
   ranking, and leakage detection. Every removal has a stated reason.
8. **Model training** — only models compatible with the detected problem type, drawn from a
   fixed library. A failing model is recorded and skipped; it never stops the run.
9. **Evaluation** — accuracy, precision, recall, F1, ROC-AUC and a confusion matrix for
   classification; MAE, MSE, RMSE and R² for regression. Plus a naive baseline.
10. **Objective validation** — the part that matters. See below.
11. **Dashboard and report** — charts, tables, explanations, and a downloadable report.

## Three design decisions worth explaining in an interview

**1. Gemini recommends; it never executes.**
`agent/models.py` contains a hand-written dictionary of every estimator the system can run.
Model names returned by the AI are resolved against it; anything unknown, or wrong for the
problem type, or too slow for the dataset size is rejected — and the rejection is shown to
the user. No code produced by a language model is ever executed.

**2. Nothing is fitted before the split.**
The train/test split happens *before* feature engineering, feature selection and
imputation. Missing-value imputation lives inside the scikit-learn pipeline so the imputers
are fitted on the training fold only. This is why the reported scores are defensible:
the test rows never influenced a single decision.

**3. Model performance and objective fulfilment are separate questions.**
A churn model can be 94% accurate and never once predict churn. `validate_objective()` runs
independent checks — absolute quality, does it beat the naive baseline, does it catch the
class the objective cares about, is it stable across folds — and returns FULFILLED,
PARTIALLY FULFILLED or NOT FULFILLED with the reasoning behind it.

The agent also refuses work it cannot do properly. Clustering, anomaly detection and
time-series forecasting are detected and explained rather than forced through a random
split that would produce a confident, meaningless number.

---

## Setup

### 1. Requirements

- Python 3.10 or newer
- MySQL 8.0 (or MariaDB 10.5+)
- A Gemini API key — **optional**, see below

### 2. Install

```bash
cd dsagent
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Database

```bash
mysql -u root -p < schema.sql
```

The app also creates the database and tables automatically on startup, so this step is
optional — but running it once makes any permission problem obvious immediately.

### 4. Configuration

```bash
cp .env.example .env
```

Then edit `.env` with your MySQL password. If you have a Gemini key
(free at <https://aistudio.google.com/apikey>) add it as `GEMINI_API_KEY`.

**Without a Gemini key everything still works.** A built-in rule-based analyst produces the
same structured recommendation, and the UI says which one was used. This is deliberate: a
demo should never depend on an API key, a quota or the venue's wifi.

### 5. Run

```bash
uvicorn main:app --reload
```

Open <http://127.0.0.1:8000>.

### 6. Try it immediately

`sample_data/` contains two ready-to-use CSVs:

| File | Objective to paste in | What it exercises |
|---|---|---|
| `customer_churn.csv` | *Predict whether a customer will churn.* | Binary classification on deliberately messy data — currency stored as text, inconsistent casing, a constant column, an ID column, free text, duplicates, an infinity and missing values |
| `house_prices.csv` | *Predict house prices from the property details.* | Regression, feature engineering and the actual-vs-predicted chart |

Try *"Forecast monthly revenue over the next year"* on the churn file too — the agent will
correctly refuse to run the standard workflow and explain why.

---

## Project layout

```
dsagent/
├── main.py                 FastAPI routes — thin, one job each
├── config.py               All settings, read once from .env
├── db.py                   MySQL helpers — plain functions, no ORM
├── schema.sql              Database schema (single source of truth)
├── agent/
│   ├── profiling.py        Dataset understanding and quality scoring
│   ├── cleaning.py         Cleaning, with a reason for every action
│   ├── gemini.py           Gemini prompt, JSON validation, local fallback
│   ├── features.py         Feature engineering, selection, leakage detection
│   ├── models.py           The fixed, safe model library
│   ├── training.py         Preprocessing, training, metrics, objective validation
│   ├── charts.py           matplotlib charts (server-rendered PNGs)
│   ├── report.py           The written report
│   └── pipeline.py         The orchestrator — read this one first
├── templates/              Jinja2 templates
├── static/css/style.css    One stylesheet
└── uploads/                Uploaded CSVs (metadata and path go to MySQL)
```

`agent/pipeline.py` knows nothing about FastAPI or MySQL. It takes a dataframe and an
objective, calls a progress callback, and returns a dictionary — which is what makes it
testable on its own, and readable in one sitting.

## Routes

| Method | Route | Purpose |
|---|---|---|
| GET | `/` | Landing page |
| GET | `/workspace` | Upload a dataset and state the objective |
| POST | `/upload` | Validate, store, create the analysis, start the agent |
| GET | `/analysis/{id}` | Live progress |
| GET | `/analysis/{id}/status` | JSON polled by the progress page |
| POST | `/analysis/{id}/target` | Confirm the target column when the agent asks |
| GET | `/analysis/{id}/results` | Results dashboard |
| GET | `/report/{id}` | Full written report |
| GET | `/models` | The model library the agent may use |
| GET | `/history` | Past analyses |
| GET/POST | `/login` | Minimal account, only for downloading |
| GET | `/download-report/{id}` | Download the report (one-use token) |

## Database tables

`users` · `datasets` · `analyses` · `analysis_steps` · `model_results` · `reports`

Uploaded CSVs are stored on the filesystem; MySQL holds their metadata and path. The
database is the analysis trail — what the agent decided and why — not a copy of your data.

## Authentication

Running an analysis and viewing the dashboard and report need **no account at all**. An
account is asked for only when you download the report, and it is deliberately the smallest
thing that works: a salted PBKDF2 hash in MySQL, and a one-use token held in memory that
expires after ten minutes. No JWT, no OAuth, no cookies, no sessions — because nothing else
in the app sits behind a login. A production system with more protected surface would use
real server-side sessions instead.

## Model library

16 models plus a baseline for each task:

*Classification* — Logistic Regression, Decision Tree, Random Forest, Extra Trees, K-Nearest
Neighbors, Gaussian Naive Bayes, Support Vector Classifier, Gradient Boosting,
HistGradientBoosting.

*Regression* — Linear, Ridge, Lasso, Decision Tree, Random Forest, Extra Trees, Gradient
Boosting, HistGradientBoosting, K-Nearest Neighbors.

*Baselines* — most-frequent class, and mean prediction. Always trained, because "is this
model any good?" needs something to compare against.

## Known limitations

- CSV only. The loader is structured so other formats can be added without touching the
  pipeline, but nothing else is supported today.
- No hyperparameter search. Every model uses sensible defaults, so there is headroom left
  on every result.
- Datasets above `MAX_TRAINING_ROWS` (default 50,000) are sampled before training. Profiling
  still uses every row.
- Clustering, anomaly detection and time-series forecasting are detected and explained, not
  performed.
- Analyses run as FastAPI background tasks inside the web process. That is the right size
  for this project; a production deployment would move them to a separate worker.
