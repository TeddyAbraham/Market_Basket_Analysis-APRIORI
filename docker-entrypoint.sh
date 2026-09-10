#!/bin/sh
set -e

if [ ! -f "/app/models/rules_all.pkl" ]; then
    if [ ! -f "/app/data/raw/Online_Retail.csv" ]; then
        echo "ERROR: /app/data/raw/Online_Retail.csv not found. Mount your data/ folder as a volume."
        exit 1
    fi
    echo "No cached model artifacts found — running scripts/train.py..."
    python scripts/train.py
else
    echo "Using cached model artifacts in /app/models."
fi

exec streamlit run app/streamlit_app.py --server.address=0.0.0.0 --server.port=8501 --server.headless=true
