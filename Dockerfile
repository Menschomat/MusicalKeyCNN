# ── Stage 1: install Python dependencies ─────────────────────────────────────
FROM python:3.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy only the dependency manifest so this layer is cached unless deps change
COPY pyproject.toml uv.lock ./

# Install production deps into an isolated venv; skip the project package itself
RUN uv sync --frozen --no-dev --no-install-project

# Pre-download beat-this model checkpoint (~77 MB) into the torch hub cache.
# Placed after uv sync so this layer is invalidated only when deps change,
# not when application source files change.
RUN .venv/bin/python -c "from beat_this.inference import Audio2Beats; Audio2Beats()"

# ── Stage 2: minimal runtime image ───────────────────────────────────────────
FROM python:3.13-slim AS runtime

# libsndfile1  – required by soundfile (librosa audio I/O)
# ffmpeg is intentionally omitted: torchaudio 2.x bundles its own FFmpeg
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the pre-built venv from the builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy the pre-downloaded beat-this checkpoint so the first request doesn't trigger a network call
COPY --from=builder /root/.cache/torch/hub/checkpoints/beat_this-final0.ckpt /root/.cache/torch/hub/checkpoints/beat_this-final0.ckpt

# Copy only the files the API actually imports
COPY api.py audio_utils.py constants.py eval.py predict_keys.py predict_bpm.py model.py dataset.py ./

# Model checkpoint – required at startup
COPY checkpoints/ ./checkpoints/

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
