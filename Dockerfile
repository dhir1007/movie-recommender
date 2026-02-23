# Dockerfile - fixed for implicit + OpenBLAS + libgomp1

# Stage 1: Builder (with compilers + BLAS dev)
FROM python:3.12-slim AS builder

# Install build tools + OpenBLAS dev for compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    gfortran \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Poetry
RUN pip install --no-cache-dir poetry

# Copy dependency files (cache layer)
COPY pyproject.toml poetry.lock* ./

# Install Python deps (implicit will compile here)
RUN poetry config virtualenvs.create false \
    && poetry install --no-root --no-interaction --no-ansi

# Stage 2: Runtime (minimal image + libgomp1 for OpenMP)
FROM python:3.12-slim

# Install runtime deps: OpenBLAS + libgomp1 (for implicit multi-threading)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy app code
COPY api/ api/
COPY src/ src/

# Expose port
EXPOSE 8000

# Run uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]