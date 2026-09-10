FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY app/ app/
COPY scripts/ scripts/
COPY eda/eda.py eda/eda.py
COPY docker-entrypoint.sh .
RUN chmod +x docker-entrypoint.sh

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

EXPOSE 8501

ENTRYPOINT ["./docker-entrypoint.sh"]
