import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mba import preprocessing, recommend  # noqa: E402


def make_raw_frame():
    return pd.DataFrame(
        {
            "InvoiceNo": ["536365", "536365", "C536366", "536367", "536368"],
            "StockCode": ["85123A", "71053", "85123A", "POST", "84029G"],
            "Description": ["WHITE HANGING HEART", "WHITE METAL LANTERN", "WHITE HANGING HEART", "Postage", "KNITTED FLAG BOTTLE"],
            "Quantity": [6, 6, -1, 1, 0],
            "InvoiceDate": ["01/12/10 8:26", "01/12/10 8:26", "01/12/10 9:00", "01/12/10 9:10", "01/12/10 9:20"],
            "UnitPrice": [2.55, 3.39, 2.55, 5.0, 3.39],
            "CustomerID": ["17850", "17850", "17850", "17851", "17852"],
            "Country": ["United Kingdom", "United Kingdom", "United Kingdom", "United Kingdom", "France"],
        }
    )


def test_clean_transactions_drops_cancelled_negative_and_nonproduct_rows():
    raw = make_raw_frame()
    clean = preprocessing.clean_transactions(raw)

    assert "C536366" not in clean["InvoiceNo"].values
    assert (clean["Quantity"] > 0).all()
    assert "POST" not in clean["StockCode"].values
    assert len(clean) == 2
    assert set(clean["Description"]) == {"WHITE HANGING HEART", "WHITE METAL LANTERN"}


def test_clean_transactions_keeps_valid_rows_untouched():
    raw = make_raw_frame().iloc[[0, 1]]
    clean = preprocessing.clean_transactions(raw)
    assert len(clean) == 2


def test_recommend_falls_back_to_popularity_when_no_artifacts(monkeypatch, tmp_path):
    monkeypatch.setattr(recommend.config, "RULES_ALL_PATH", tmp_path / "missing_all.pkl")
    monkeypatch.setattr(recommend.config, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(recommend.config, "RULES_COUNTRY_TEMPLATE", "missing_{country}.pkl")
    monkeypatch.setattr(
        recommend,
        "load_catalog_popularity",
        lambda: ["ITEM A", "ITEM B", "ITEM C"],
    )

    result = recommend.recommend(["ITEM A"], country="Nowhereland")

    assert result["fallback_used"] is True
    assert result["recommendations"]
    assert all(r["product"] != "ITEM A" for r in result["recommendations"])


def test_recommend_uses_rules_when_available(monkeypatch):
    rules = pd.DataFrame(
        [
            {
                "antecedents": frozenset({"ITEM A"}),
                "consequents": frozenset({"ITEM B"}),
                "support": 0.1,
                "confidence": 0.5,
                "lift": 2.0,
            }
        ]
    )
    monkeypatch.setattr(recommend, "load_rules", lambda scope: rules)

    result = recommend.recommend(["ITEM A"], country=recommend.config.ALL_COUNTRIES_LABEL)

    assert result["fallback_used"] is False
    assert result["recommendations"][0]["product"] == "ITEM B"
