# ── Stage 1: install Python dependencies ─────────────────────────────────────
FROM python:3.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy only the dependency manifest so this layer is cached unless deps change
COPY pyproject.toml uv.lock ./

# Install production deps into an isolated venv; skip the project package itself
RUN uv sync --frozen --no-dev --no-install-project

# ── Stage 2: minimal runtime image ───────────────────────────────────────────
FROM python:3.13-slim AS runtime

# libsndfile1  – required by soundfile (librosa audio I/O)
# ffmpeg       – required by torchaudio to decode MP3 files
RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the pre-built venv from the builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy only the files the API actually imports
COPY api.py eval.py predict_keys.py model.py dataset.py ./

# Model checkpoint – required at startup
COPY checkpoints/ ./checkpoints/

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
