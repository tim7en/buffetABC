# ============================================================
# SPY-gate × QQQ Ensemble DCA Signal API
# Build context must be the repository root:
#   docker build -f production/spy_qqq_signal_api/Dockerfile .
# ============================================================
FROM python:3.11-slim

# ---- System dependencies -------------------------------------------- #
# gcc/g++ required by scikit-learn wheel compilation fallback
# libgomp1 required by OpenMP (random forest parallelism)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Python dependencies -------------------------------------------- #
COPY production/spy_qqq_signal_api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---- Research tools (imported by pipeline.py) ----------------------- #
# Copy the tools directory (regime analysis, model compare, leverage)
COPY tools/qqq_macro_ml_regime_analysis.py   /app/tools/
COPY tools/qqq_macro_walkforward_model_compare.py /app/tools/
COPY tools/qqq_macro_walkforward_leverage.py /app/tools/

# ---- Cache / download scripts --------------------------------------- #
# The macro download script is run as a subprocess during daily refresh
COPY cache/download_daily_macro_data.py /app/cache_scripts/download_daily_macro_data.py
COPY cache/download_qqq_daily.py        /app/cache_scripts/download_qqq_daily.py

# ---- FastAPI application -------------------------------------------- #
COPY production/spy_qqq_signal_api/app/ /app/app/

# ---- Persistent data volume ----------------------------------------- #
# Mount a named volume here so signal_state.json and parquet caches
# survive container restarts.
VOLUME ["/data"]

# ---- Runtime configuration ------------------------------------------ #
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DATA_DIR=/data \
    # Override with a real secret in production:
    REFRESH_TOKEN="" \
    MAX_STATE_AGE_H=25

EXPOSE 8000

# ---- Health check --------------------------------------------------- #
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--log-level", "info", \
     "--access-log"]
