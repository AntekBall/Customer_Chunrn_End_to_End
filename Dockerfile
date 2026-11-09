FROM ghcr.io/astral-sh/uv:0.13.7 AS uv
FROM python:3.12.12-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /usr/local/bin/uv
ENV UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --only-group serving --no-install-project \
    && uv cache clean

COPY src ./src
COPY sql ./sql
RUN useradd --create-home --uid 10001 churn
USER churn
EXPOSE 8000 8050
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]