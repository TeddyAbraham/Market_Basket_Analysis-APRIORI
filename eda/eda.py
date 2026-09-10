"""Standalone EDA for the Online Retail dataset.

Produces PNG charts under eda/output/figures/ and a single self-contained
eda/output/report.html (images embedded as base64) summarizing the dataset.
Does not depend on scripts/train.py or any cached model artifacts.

Usage:
    python eda/eda.py
"""

import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from mba import config, preprocessing  # noqa: E402

sns.set_theme(style="whitegrid")
FIG_DIR = config.EDA_OUTPUT_DIR / "figures"


def savefig(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, bbox_inches="tight", dpi=110)
    plt.close(fig)
    return path


def img_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def data_quality_summary(raw: pd.DataFrame, clean: pd.DataFrame) -> dict:
    cancelled = raw["InvoiceNo"].astype(str).str.startswith("C").sum()
    return {
        "Raw rows": f"{len(raw):,}",
        "Clean rows": f"{len(clean):,}",
        "Rows removed": f"{len(raw) - len(clean):,}",
        "Missing Description (raw)": f"{raw['Description'].isna().sum():,}",
        "Missing CustomerID (raw)": f"{raw['CustomerID'].isna().sum():,}",
        "Cancelled-order rows (raw)": f"{cancelled:,}",
        "Unique invoices (clean)": f"{clean['InvoiceNo'].nunique():,}",
        "Unique products (clean)": f"{clean['Description'].nunique():,}",
        "Unique countries (clean)": f"{clean['Country'].nunique():,}",
        "Date range (clean)": f"{clean['InvoiceDate'].min().date()} to {clean['InvoiceDate'].max().date()}",
    }


def plot_top_products(clean: pd.DataFrame, n=20):
    top = clean.groupby("Description")["InvoiceNo"].nunique().sort_values(ascending=False).head(n)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.barplot(x=top.values, y=top.index, ax=ax, color="#4C72B0")
    ax.set_xlabel("Number of invoices")
    ax.set_ylabel("")
    ax.set_title(f"Top {n} products by invoice frequency")
    return savefig(fig, "top_products.png")


def plot_top_countries(clean: pd.DataFrame, n=15):
    top = clean.groupby("Country")["InvoiceNo"].nunique().sort_values(ascending=False).head(n)
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(x=top.values, y=top.index, ax=ax, color="#55A868")
    ax.set_xlabel("Number of invoices")
    ax.set_ylabel("")
    ax.set_title(f"Top {n} countries by invoice count")
    return savefig(fig, "top_countries.png")


def plot_basket_size_distribution(clean: pd.DataFrame):
    basket_sizes = clean.groupby("InvoiceNo")["Description"].nunique()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(basket_sizes[basket_sizes <= 50], bins=50, ax=ax, color="#C44E52")
    ax.set_xlabel("Distinct products per invoice")
    ax.set_ylabel("Number of invoices")
    ax.set_title("Basket size distribution (capped at 50 items for readability)")
    return savefig(fig, "basket_size_distribution.png")


def plot_monthly_trend(clean: pd.DataFrame):
    monthly = clean.set_index("InvoiceDate").resample("MS")["InvoiceNo"].nunique()
    fig, ax = plt.subplots(figsize=(10, 5))
    monthly.plot(ax=ax, marker="o", color="#8172B2")
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of invoices")
    ax.set_title("Monthly order volume")
    return savefig(fig, "monthly_trend.png")


def plot_price_distribution(clean: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(clean[clean["UnitPrice"] <= 20]["UnitPrice"], bins=40, ax=ax, color="#CCB974")
    ax.set_xlabel("Unit price (GBP, capped at 20 for readability)")
    ax.set_ylabel("Number of line items")
    ax.set_title("Unit price distribution")
    return savefig(fig, "price_distribution.png")


def build_html_report(summary: dict, fig_paths: list, countries_top: pd.Series):
    rows = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in summary.items())
    images = "".join(
        f'<div class="figure"><img src="data:image/png;base64,{img_to_base64(p)}" alt="{p.stem}"></div>'
        for p in fig_paths
    )
    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Online Retail — EDA Report</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 2rem; color: #222; background: #fafafa; }}
  h1 {{ margin-bottom: 0.2rem; }}
  .subtitle {{ color: #666; margin-top: 0; }}
  table {{ border-collapse: collapse; margin: 1.5rem 0; }}
  th, td {{ text-align: left; padding: 6px 14px; border-bottom: 1px solid #ddd; }}
  th {{ color: #444; font-weight: 600; }}
  .figure {{ margin: 1.5rem 0; background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  .figure img {{ max-width: 100%; }}
</style>
</head>
<body>
  <h1>Online Retail — Exploratory Data Analysis</h1>
  <p class="subtitle">Generated by eda/eda.py</p>
  <h2>Data quality summary</h2>
  <table>{rows}</table>
  <h2>Charts</h2>
  {images}
</body>
</html>
"""
    report_path = config.EDA_OUTPUT_DIR / "report.html"
    report_path.write_text(html, encoding="utf-8")
    return report_path


def main():
    print("Loading raw data...")
    raw = preprocessing.load_raw()
    print("Cleaning...")
    clean = preprocessing.clean_transactions(raw)

    print("Computing summary + charts...")
    summary = data_quality_summary(raw, clean)
    fig_paths = [
        plot_top_products(clean),
        plot_top_countries(clean),
        plot_basket_size_distribution(clean),
        plot_monthly_trend(clean),
        plot_price_distribution(clean),
    ]
    countries_top = clean.groupby("Country")["InvoiceNo"].nunique().sort_values(ascending=False)

    report_path = build_html_report(summary, fig_paths, countries_top)
    print(f"Report written to {report_path}")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
