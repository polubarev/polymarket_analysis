FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY polymarket_pipeline /app/polymarket_pipeline
COPY scripts/cloud_run_entrypoint.sh /app/scripts/cloud_run_entrypoint.sh

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

RUN chmod +x /app/scripts/cloud_run_entrypoint.sh

ENTRYPOINT ["/app/scripts/cloud_run_entrypoint.sh"]
