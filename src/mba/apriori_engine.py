"""Apriori frequent-itemset mining and association-rule generation."""

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

from mba import config


def adaptive_min_support(num_baskets: int) -> float:
    """Pick a min_support that still yields itemsets for small baskets counts.

    A fixed min_support tuned for ~25k UK invoices would find nothing in a
    country with a few hundred invoices, so scale the floor up as the basket
    count shrinks while never going below the base support for large scopes.
    """
    if num_baskets <= 0:
        return config.BASE_MIN_SUPPORT
    support_for_n_baskets = config.ADAPTIVE_SUPPORT_MIN_ITEMSET_COUNT / num_baskets
    return max(config.BASE_MIN_SUPPORT, min(support_for_n_baskets, 0.5))


def mine_frequent_itemsets(onehot: pd.DataFrame, min_support: float) -> pd.DataFrame:
    return apriori(
        onehot,
        min_support=min_support,
        use_colnames=True,
        low_memory=True,
        max_len=config.MAX_ITEMSET_LEN,
    )


def mine_rules(frequent_itemsets: pd.DataFrame) -> pd.DataFrame:
    if frequent_itemsets.empty:
        return pd.DataFrame(
            columns=["antecedents", "consequents", "support", "confidence", "lift"]
        )
    rules = association_rules(
        frequent_itemsets,
        metric="lift",
        min_threshold=config.MIN_LIFT,
        num_itemsets=len(frequent_itemsets),
    )
    rules = rules[rules["confidence"] >= config.MIN_CONFIDENCE]
    rules = rules.sort_values(["lift", "confidence"], ascending=False).reset_index(drop=True)
    return rules[["antecedents", "consequents", "support", "confidence", "lift"]]


def build_rules_for_scope(onehot: pd.DataFrame, num_baskets: int) -> pd.DataFrame:
    """Run the full itemset -> rules pipeline for one scope (All / one country)."""
    min_support = adaptive_min_support(num_baskets)
    itemsets = mine_frequent_itemsets(onehot, min_support)
    return mine_rules(itemsets)
