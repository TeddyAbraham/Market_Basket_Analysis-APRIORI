"""Load and clean the Online Retail transactions.

Shared by scripts/train.py and eda/eda.py so both operate on the exact same
definition of a "clean" transaction.
"""

import pandas as pd

from mba import config


def load_raw(csv_path=None) -> pd.DataFrame:
    """Read the raw Online Retail CSV, handling its UTF-8 BOM."""
    path = csv_path or config.RAW_CSV_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data file not found at {path}. Place Online_Retail.csv there "
            "before running training or EDA."
        )
    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype={"StockCode": str, "CustomerID": str},
    )
    df.columns = [c.strip() for c in df.columns]
    return df


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the standard cleaning rules and return a tidy transactions frame.

    Rules applied (see plan for rationale):
      - drop rows with missing Description
      - drop cancelled orders (InvoiceNo starting with "C")
      - drop non-positive Quantity or UnitPrice
      - drop known non-product StockCodes (postage/fees/manual adjustments)
      - normalize whitespace in Description and Country
    """
    out = df.copy()

    out["InvoiceNo"] = out["InvoiceNo"].astype(str).str.strip()
    out["StockCode"] = out["StockCode"].astype(str).str.strip()
    out["Description"] = out["Description"].astype(str).str.strip()
    out["Country"] = out["Country"].astype(str).str.strip()

    out = out[out["Description"].notna()]
    out = out[~out["Description"].isin(["", "nan", "NaN"])]

    out = out[~out["InvoiceNo"].str.startswith("C")]

    out["Quantity"] = pd.to_numeric(out["Quantity"], errors="coerce")
    out["UnitPrice"] = pd.to_numeric(out["UnitPrice"], errors="coerce")
    out = out[(out["Quantity"] > 0) & (out["UnitPrice"] > 0)]

    out = out[~out["StockCode"].isin(config.NON_PRODUCT_STOCK_CODES)]

    out["InvoiceDate"] = pd.to_datetime(out["InvoiceDate"], format="%d/%m/%y %H:%M", errors="coerce")

    out = out.dropna(subset=["InvoiceNo", "Description", "InvoiceDate"])

    return out.reset_index(drop=True)


def load_clean_transactions(csv_path=None) -> pd.DataFrame:
    """Convenience wrapper: load raw + clean in one call."""
    return clean_transactions(load_raw(csv_path))
