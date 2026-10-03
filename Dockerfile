FROM python:3.11-slim

# uv = the fast installer we use locally
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never PYTHONUNBUFFERED=1

WORKDIR /app

# Install production dependencies exactly as locked (no dev tools, no notebooks, no MLflow)
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

# The trained model and its decision file travel with the image
COPY artifacts ./artifacts

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000

# Cloud hosts (Render, etc.) tell us the port through $PORT; default to 8000 locally
CMD ["sh", "-c", "uvicorn churn_predictor.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
