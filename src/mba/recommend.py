"""Load precomputed rules/artifacts and answer "what should we recommend?".

Nothing in this module runs Apriori — it only reads what scripts/train.py
already computed and cached to disk, which is what keeps the app fast.
"""

import json

import pandas as pd

from mba import config


def _country_rules_path(country: str):
    safe = country.replace(" ", "_").replace("/", "_")
    return config.MODELS_DIR / config.RULES_COUNTRY_TEMPLATE.format(country=safe)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_top30_products() -> list:
    return load_json(config.TOP30_PRODUCTS_PATH)


def load_catalog_popularity() -> dict:
    return load_json(config.CATALOG_POPULARITY_PATH)


def load_countries_meta() -> dict:
    return load_json(config.COUNTRIES_META_PATH)


def load_rules(scope: str) -> pd.DataFrame:
    """Load cached rules for a scope ("All Countries" or a country name).

    Returns an empty DataFrame (not an error) if no artifact exists for that
    scope, so callers can fall back gracefully.
    """
    path = config.RULES_ALL_PATH if scope == config.ALL_COUNTRIES_LABEL else _country_rules_path(scope)
    if not path.exists():
        return pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])
    return pd.read_pickle(path)


def _rules_matching_selection(rules: pd.DataFrame, selected: frozenset) -> pd.DataFrame:
    if rules.empty:
        return rules
    mask = rules["antecedents"].apply(lambda ante: ante.issubset(selected))
    return rules[mask]


def _rank_consequents(matching_rules: pd.DataFrame, selected: frozenset, top_k: int) -> list:
    consequent_best = {}
    for _, row in matching_rules.iterrows():
        for item in row["consequents"]:
            if item in selected:
                continue
            candidate = (row["lift"], row["confidence"], row["support"])
            if item not in consequent_best or candidate > consequent_best[item][:3]:
                consequent_best[item] = (row["lift"], row["confidence"], row["support"], item)

    ranked = sorted(consequent_best.values(), key=lambda t: (t[0], t[1]), reverse=True)
    return [
        {"product": item, "lift": round(lift, 3), "confidence": round(conf, 3), "support": round(sup, 4)}
        for lift, conf, sup, item in ranked[:top_k]
    ]


def recommend(selected_items, country: str = None, top_k: int = None) -> dict:
    """Return top-k recommended products for the given selection + country.

    Fallback chain: country-specific rules -> All Countries rules -> global
    popularity (excluding already-selected items). Always returns a result,
    never raises for "no rules found" — that's communicated via `note`.
    """
    top_k = top_k or config.TOP_K_RECOMMENDATIONS
    selected = frozenset(selected_items)
    scope = country or config.ALL_COUNTRIES_LABEL

    result = {"recommendations": [], "scope_used": scope, "fallback_used": False, "note": ""}

    if scope != config.ALL_COUNTRIES_LABEL:
        country_rules = load_rules(scope)
        matches = _rules_matching_selection(country_rules, selected)
        if not matches.empty:
            result["recommendations"] = _rank_consequents(matches, selected, top_k)
            if result["recommendations"]:
                return result

    all_rules = load_rules(config.ALL_COUNTRIES_LABEL)
    matches = _rules_matching_selection(all_rules, selected)
    if not matches.empty:
        recs = _rank_consequents(matches, selected, top_k)
        if recs:
            result["recommendations"] = recs
            result["scope_used"] = config.ALL_COUNTRIES_LABEL
            result["fallback_used"] = scope != config.ALL_COUNTRIES_LABEL
            if result["fallback_used"]:
                result["note"] = (
                    f"No strong association found specifically for {scope}; "
                    "showing results from All Countries instead."
                )
            return result

    popularity = load_catalog_popularity()
    fallback_items = [p for p in popularity if p not in selected]
    result["recommendations"] = [
        {"product": p, "lift": None, "confidence": None, "support": None}
        for p in fallback_items[:top_k]
    ]
    result["fallback_used"] = True
    result["note"] = "No strong association rules found for this selection; showing globally popular items instead."
    return result
