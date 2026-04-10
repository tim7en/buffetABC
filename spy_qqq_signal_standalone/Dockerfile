# ============================================================
# SPY-gate × QQQ 2x DCA Signal API  —  standalone build
# Build context must be this folder:
#   docker build -t spy-qqq-signal-api .
# Or via docker-compose:
#   docker-compose up --build
# ============================================================
FROM python:3.11-slim-bookworm

# ---- System dependencies -------------------------------------------- #
# All Python packages ship as manylinux wheels (no gcc needed).
# scikit-learn bundles its own OpenMP runtime inside the wheel.
# No apt-get required — skip to keep the image portable across Docker hosts.

WORKDIR /app

# ---- Python dependencies -------------------------------------------- #
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --progress-bar off -r requirements.txt

# ---- Research tools (imported by pipeline.py) ----------------------- #
COPY tools/ /app/tools/

# ---- Cache / download scripts --------------------------------------- #
COPY cache_scripts/ /app/cache_scripts/

# ---- FastAPI application -------------------------------------------- #
COPY app/ /app/app/

# ---- Persistent data volume ----------------------------------------- #
# Mount a named volume here so signal_state.json and parquet caches
# survive container restarts.
VOLUME ["/data"]

# ---- Runtime configuration ------------------------------------------ #
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DATA_DIR=/data \
    REFRESH_TOKEN="" \
    MAX_STATE_AGE_H=25 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

EXPOSE 8000

HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()"

CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--log-level", "info", \
     "--access-log"]
