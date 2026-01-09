FROM nvcr.io/nvidia/pytorch:24.07-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y \
    curl \
    wget \
    libsndfile1 \
    ffmpeg \
    sox \
    libsox-fmt-all \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY *.py ./

# Install dependencies with uv
RUN uv pip install --system --no-cache \
    nemo-toolkit[asr]==2.5.3 \
    omegaconf>=2.3.0 \
    hydra-core>=1.3.0 \
    fastapi>=0.104.0 \
    uvicorn[standard]>=0.24.0 \
    websockets>=12.0 \
    pydantic>=2.0.0 \
    scipy>=1.10.0 \
    librosa>=0.10.0 \
    soundfile>=0.12.0

EXPOSE 8765

CMD ["python", "server.py", "--model", "/app/model.nemo", "--port", "8765", "--host", "0.0.0.0"]
