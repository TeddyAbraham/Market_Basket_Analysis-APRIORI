"""Build sparse one-hot basket (invoice x product) matrices for Apriori."""

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder


def top_products_by_invoice_count(transactions: pd.DataFrame, n: int) -> list:
    """Rank products by number of distinct invoices containing them."""
    counts = (
        transactions.groupby("Description")["InvoiceNo"]
        .nunique()
        .sort_values(ascending=False)
    )
    return counts.head(n).index.tolist()


def build_transaction_list(transactions: pd.DataFrame, allowed_products=None) -> list:
    """Group rows into a list of baskets (lists of product descriptions).

    If ``allowed_products`` is given, items outside that set are dropped from
    each basket before it's returned — this is how the item universe fed into
    Apriori is bounded for speed (see config.CATALOG_CAP).
    """
    df = transactions
    if allowed_products is not None:
        allowed = set(allowed_products)
        df = df[df["Description"].isin(allowed)]

    baskets = df.groupby("InvoiceNo")["Description"].apply(lambda s: list(set(s)))
    baskets = baskets[baskets.map(len) > 0]
    return baskets.tolist()


def build_basket_matrix(transactions: pd.DataFrame, allowed_products=None):
    """Return (sparse one-hot DataFrame, TransactionEncoder) for Apriori input."""
    basket_list = build_transaction_list(transactions, allowed_products)
    if not basket_list:
        return None, None

    encoder = TransactionEncoder()
    sparse_array = encoder.fit(basket_list).transform(basket_list, sparse=True)
    onehot = pd.DataFrame.sparse.from_spmatrix(sparse_array, columns=encoder.columns_)
    return onehot, encoder
