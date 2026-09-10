"""Central configuration for paths and tunable thresholds."""

from pathlib import Path

# ---- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
EDA_OUTPUT_DIR = PROJECT_ROOT / "eda" / "output"

RAW_CSV_PATH = DATA_RAW_DIR / "Online_Retail.csv"
CLEANED_PARQUET_PATH = DATA_PROCESSED_DIR / "cleaned_transactions.parquet"

RULES_ALL_PATH = MODELS_DIR / "rules_all.pkl"
RULES_COUNTRY_TEMPLATE = "rules_{country}.pkl"
TOP30_PRODUCTS_PATH = MODELS_DIR / "top30_products.json"
CATALOG_POPULARITY_PATH = MODELS_DIR / "catalog_popularity.json"
COUNTRIES_META_PATH = MODELS_DIR / "countries.json"
TRAIN_META_PATH = MODELS_DIR / "meta.json"

ALL_COUNTRIES_LABEL = "All Countries"

# ---- Non-product StockCodes to exclude (postage, fees, manual adjustments) -
NON_PRODUCT_STOCK_CODES = {
    "POST",       # Postage
    "DOT",        # Dotcom postage
    "D",          # Discount
    "M",          # Manual
    "m",
    "C2",         # Carriage
    "BANK CHARGES",
    "PADS",       # Pads to match orders
    "CRUK",       # Charity commission
    "AMAZONFEE",
    "S",          # Samples
}

# ---- Top-N / catalog sizing --------------------------------------------------
TOP_N_PRODUCTS = 30          # selectable products in the UI
CATALOG_CAP = 200            # bound the item universe fed into Apriori (speed)

# ---- Apriori / rule-mining thresholds ---------------------------------------
BASE_MIN_SUPPORT = 0.02      # default min_support for "All Countries" and large countries
MIN_SUPPORT_FLOOR = 0.03     # smallest min_support allowed for adaptive per-country runs
MIN_INVOICES_PER_COUNTRY = 30  # countries with fewer invoices are skipped (insufficient data)
MIN_CONFIDENCE = 0.2
MIN_LIFT = 1.0
MAX_ITEMSET_LEN = 3
TOP_K_RECOMMENDATIONS = 5

# Countries with fewer invoices than this get a boosted min_support floor so
# Apriori doesn't explode combinatorially on a still-large but sparse item set.
ADAPTIVE_SUPPORT_MIN_ITEMSET_COUNT = 3
