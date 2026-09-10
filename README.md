# Market Basket Analysis — Online Retail

Apriori-based market basket analysis on the UCI [Online Retail dataset](https://archive.ics.uci.edu/dataset/352/online+retail),
with a Streamlit app that recommends products based on what a customer already
has in their basket, optionally filtered by country.

## What's in here

```
data/
  raw/           <- put Online_Retail.csv here (not committed, see below)
  processed/      <- cleaned data cache written by scripts/train.py
eda/
  eda.py           <- standalone EDA: charts + eda/output/report.html
  output/           <- generated charts + report (not committed)
src/mba/
  config.py        <- paths + tunable thresholds (top-N, support, etc.)
  preprocessing.py <- shared data loading/cleaning
  basket.py        <- sparse one-hot basket matrix construction
  apriori_engine.py<- Apriori mining + association rule generation
  recommend.py     <- rule lookup + fallback logic used by the app
scripts/
  train.py         <- offline precompute step, writes models/*
models/            <- cached rules/artifacts (not committed, generated)
app/
  streamlit_app.py <- the recommender UI
tests/
  test_mba.py      <- unit tests for cleaning + recommendation fallback logic
```

## How it works

1. **Clean the data** — drop cancelled orders, non-positive quantities/prices,
   missing descriptions, and non-product line items (postage, fees, manual
   adjustments). One basket = one invoice.
2. **Bound the item universe** — Apriori's cost explodes with the number of
   distinct items. Instead of running it over all ~4,000 products, the item
   universe is capped to the **top 200 most-frequent products** (`CATALOG_CAP`
   in `config.py`). The **top 30** of those are what's offered as selectable
   products in the UI; consequents can be any of the 200, so recommendations
   stay richer than a 30x30 matrix while Apriori stays fast.
3. **Mine rules offline** — `scripts/train.py` runs Apriori once for "All
   Countries" and once per country with enough invoices (>= 30), in parallel
   across CPU cores, and pickles the resulting association rules to `models/`.
   Countries with too little data are skipped and fall back to "All
   Countries" rules at recommendation time.
4. **Serve from cache** — the Streamlit app never runs Apriori itself. It
   loads the cached rule tables and does a simple, fast subset lookup:
   antecedent ⊆ selected items, ranked by lift then confidence.

Full training over the ~528k-row cleaned dataset (15 scopes: All Countries +
14 qualifying countries) takes about 25 seconds on a typical laptop.

## Setup (local, no Docker)

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Place the dataset:

```
data/raw/Online_Retail.csv
```

Train the models (required once before running the app):

```bash
python scripts/train.py
```

Run the EDA report:

```bash
python eda/eda.py
# open eda/output/report.html
```

Run the app:

```bash
streamlit run app/streamlit_app.py
```

Run tests:

```bash
pytest tests/
```

## Docker

The container's entrypoint trains models automatically on first run if
`models/` is empty but `data/raw/Online_Retail.csv` is present via the
mounted volume; on later runs it reuses the cached artifacts.

```bash
docker compose up --build
```

or without compose:

```bash
docker build -t market-basket-analysis .
docker run -p 8501:8501 -v "%cd%/data:/app/data" -v "%cd%/models:/app/models" market-basket-analysis
```

Then open http://localhost:8501.

## Configuration

Key tunables live in `src/mba/config.py`:

| Setting | Meaning |
|---|---|
| `TOP_N_PRODUCTS` | How many products are selectable in the UI (default 30) |
| `CATALOG_CAP` | Item universe size fed into Apriori (default 200) |
| `BASE_MIN_SUPPORT` | min_support for large scopes (All Countries, UK) |
| `MIN_INVOICES_PER_COUNTRY` | Countries below this invoice count are skipped |
| `MAX_ITEMSET_LEN` | Caps itemset size Apriori searches (bounds combinatorics) |

## Data

`Online_Retail.csv` is **not** committed to this repo (see `.gitignore`).
Download it from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/352/online+retail)
and place it at `data/raw/Online_Retail.csv` before running training or EDA.

## Future improvements

- **FP-Growth** as a drop-in alternative to Apriori (`mlxtend.frequent_patterns.fpgrowth`) —
  scales better if the catalog cap is raised.
- **Customer-level baskets** (not just per-invoice) to capture repeat-purchase patterns.
- **Seasonality-aware rules** — this dataset spans Dec 2010–Dec 2011 and includes a
  holiday season; separate rule sets per season/month could sharpen recommendations.
- **Incremental/online rule updates** instead of a full retrain as new sales data arrives.
- **A REST API layer** (Flask/FastAPI) if this needs to plug into an actual storefront's
  checkout flow rather than a standalone demo UI.
- **Additional rule-quality metrics** (conviction, leverage) beyond lift/confidence.
- **Click-through analytics** on shown recommendations to validate and retune thresholds
  against real uplift, not just support/confidence/lift.
- **Redis or similar caching layer** if this is deployed at a scale beyond a single
  Streamlit instance reading local pickles.
