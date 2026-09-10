"""Offline precompute step: clean data, mine Apriori rules, cache artifacts.

Run this once (and whenever the raw data changes) before starting the
Streamlit app. The app only ever reads what this script writes to models/ —
it never runs Apriori itself, which is what keeps the UI fast.

Usage:
    python scripts/train.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from mba import basket, apriori_engine, config, preprocessing  # noqa: E402


def ensure_dirs():
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def build_and_save_rules(scope_name: str, transactions, allowed_products, out_path):
    onehot, _ = basket.build_basket_matrix(transactions, allowed_products)
    if onehot is None or onehot.empty:
        rules = pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])
        num_baskets = 0
    else:
        num_baskets = len(onehot)
        rules = apriori_engine.build_rules_for_scope(onehot, num_baskets)
    rules.to_pickle(out_path)
    return scope_name, num_baskets, len(rules)


def country_out_path(country: str) -> Path:
    safe = country.replace(" ", "_").replace("/", "_")
    return config.MODELS_DIR / config.RULES_COUNTRY_TEMPLATE.format(country=safe)


def main():
    t0 = time.time()
    ensure_dirs()

    print("Loading and cleaning raw transactions...")
    transactions = preprocessing.load_clean_transactions()
    transactions.to_parquet(config.CLEANED_PARQUET_PATH, index=False)
    print(f"  {len(transactions):,} clean rows across {transactions['InvoiceNo'].nunique():,} invoices")

    catalog_products = basket.top_products_by_invoice_count(transactions, config.CATALOG_CAP)
    top30_products = catalog_products[: config.TOP_N_PRODUCTS]

    popularity_counts = (
        transactions[transactions["Description"].isin(catalog_products)]
        .groupby("Description")["InvoiceNo"]
        .nunique()
        .sort_values(ascending=False)
    )
    catalog_popularity = popularity_counts.index.tolist()

    with open(config.TOP30_PRODUCTS_PATH, "w", encoding="utf-8") as f:
        json.dump(top30_products, f, indent=2)
    with open(config.CATALOG_POPULARITY_PATH, "w", encoding="utf-8") as f:
        json.dump(catalog_popularity, f, indent=2)

    print(f"Top-{config.TOP_N_PRODUCTS} products selected; catalog cap = {config.CATALOG_CAP} products.")

    print("Mining 'All Countries' rules...")
    _, all_baskets, all_rules_n = build_and_save_rules(
        config.ALL_COUNTRIES_LABEL, transactions, catalog_products, config.RULES_ALL_PATH
    )
    print(f"  All Countries: {all_baskets:,} baskets -> {all_rules_n:,} rules")

    country_invoice_counts = transactions.groupby("Country")["InvoiceNo"].nunique().sort_values(ascending=False)
    qualifying_countries = country_invoice_counts[
        country_invoice_counts >= config.MIN_INVOICES_PER_COUNTRY
    ].index.tolist()

    print(f"Mining rules for {len(qualifying_countries)} qualifying countries (parallelized)...")

    def _job(country):
        country_tx = transactions[transactions["Country"] == country]
        out_path = country_out_path(country)
        return build_and_save_rules(country, country_tx, catalog_products, out_path)

    results = Parallel(n_jobs=-1, prefer="processes")(delayed(_job)(c) for c in qualifying_countries)

    countries_meta = {}
    for country, num_baskets, num_rules in results:
        countries_meta[country] = {
            "num_invoices": int(num_baskets),
            "num_rules": int(num_rules),
            "has_rules": num_rules > 0,
        }
        print(f"  {country}: {num_baskets:,} invoices -> {num_rules:,} rules")

    skipped = country_invoice_counts[country_invoice_counts < config.MIN_INVOICES_PER_COUNTRY].index.tolist()
    for country in skipped:
        countries_meta[country] = {
            "num_invoices": int(country_invoice_counts[country]),
            "num_rules": 0,
            "has_rules": False,
        }

    with open(config.COUNTRIES_META_PATH, "w", encoding="utf-8") as f:
        json.dump(countries_meta, f, indent=2)

    meta = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_rows": len(transactions),
        "total_invoices": int(transactions["InvoiceNo"].nunique()),
        "catalog_cap": config.CATALOG_CAP,
        "top_n_products": config.TOP_N_PRODUCTS,
        "num_countries_with_rules": sum(1 for v in countries_meta.values() if v["has_rules"]),
        "num_countries_skipped": len(skipped),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    with open(config.TRAIN_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone in {meta['elapsed_seconds']}s. Artifacts written to {config.MODELS_DIR}")


if __name__ == "__main__":
    main()
