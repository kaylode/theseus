# syntax = docker/dockerfile:1
#
# Multi-stage Dockerfile for Theseus v2.0
# Requires Docker BuildKit (DOCKER_BUILDKIT=1)
#
ARG BASE_IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04

# =============================================================================
# Stage 1: System dependencies
# =============================================================================
FROM ${BASE_IMAGE} AS base

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    ccache \
    cmake \
    curl \
    git \
    gcc \
    wget \
    libjpeg-dev \
    zip \
    unzip \
    bzip2 \
    ffmpeg \
    libsm6 \
    libxext6 \
    libpng-dev \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast Python package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# =============================================================================
# Stage 2: Python dependencies
# =============================================================================
FROM base AS deps

WORKDIR /workspace
COPY pyproject.toml setup.py ./
COPY theseus/__init__.py theseus/__init__.py

# Install PyTorch with CUDA 12.4 support
RUN uv pip install --system \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# Install theseus with all optional dependencies
COPY ./ /workspace/
RUN uv pip install --system -e ".[all,dev]"

# Install DVC for data versioning
RUN uv pip install --system dvc dvc-gdrive

# =============================================================================
# Stage 3: Data + Runtime
# =============================================================================
FROM deps AS runtime

WORKDIR /workspace

# Pull data from GDrive (requires credentials secret)
RUN --mount=type=secret,id=credentials \
    CREDENTIALS=$(cat /run/secrets/credentials) \
    && echo "$CREDENTIALS" > /workspace/credentials.json
RUN dvc remote modify gdrive --local gdrive_user_credentials_file /workspace/credentials.json
RUN dvc pull

ENTRYPOINT ["/bin/bash"]
