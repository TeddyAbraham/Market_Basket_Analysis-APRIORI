"""Streamlit UI for product recommendations from precomputed Apriori rules.

Select one or more of the top-30 products (optionally scoped to a country)
and see recommended products to cross-sell. All rule mining happens offline
in scripts/train.py — this app only reads cached artifacts, so it stays fast.

Usage:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st  # noqa: E402

from mba import config, recommend  # noqa: E402

st.set_page_config(page_title="Market Basket Recommender", page_icon="🛒", layout="centered")


@st.cache_resource
def load_artifacts():
    top30 = recommend.load_top30_products()
    countries_meta = recommend.load_countries_meta()
    return top30, countries_meta


def artifacts_missing() -> bool:
    return not config.TOP30_PRODUCTS_PATH.exists() or not config.RULES_ALL_PATH.exists()


def main():
    st.title("🛒 Market Basket Recommender")
    st.caption("Apriori-based product recommendations — Online Retail dataset")

    if artifacts_missing():
        st.error(
            "No trained model artifacts found. Run `python scripts/train.py` first "
            "(after placing Online_Retail.csv in `data/raw/`)."
        )
        st.stop()

    top30, countries_meta = load_artifacts()

    country_options = [config.ALL_COUNTRIES_LABEL] + sorted(
        c for c, meta in countries_meta.items() if meta.get("has_rules")
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        selected_products = st.multiselect(
            "Select product(s) from the top 30:",
            options=top30,
            help="Pick one or more products a customer has in their basket.",
        )
    with col2:
        country = st.selectbox("Country filter:", options=country_options)

    if country != config.ALL_COUNTRIES_LABEL:
        meta = countries_meta.get(country, {})
        st.caption(f"{country}: {meta.get('num_invoices', '?'):,} invoices, {meta.get('num_rules', '?'):,} rules")

    if not selected_products:
        st.info("Select at least one product to see recommendations.")
        return

    result = recommend.recommend(selected_products, country=country)

    st.subheader("Recommended products")
    if result["note"]:
        st.warning(result["note"])

    if not result["recommendations"]:
        st.write("No recommendations available.")
        return

    for rec in result["recommendations"]:
        if rec["lift"] is not None:
            st.markdown(
                f"**{rec['product']}**  \n"
                f"lift: {rec['lift']} · confidence: {rec['confidence']} · support: {rec['support']}"
            )
        else:
            st.markdown(f"**{rec['product']}**  \n_popular item (no direct rule found)_")
        st.divider()

    with st.expander("Scope used for these recommendations"):
        st.write(result["scope_used"])


if __name__ == "__main__":
    main()
