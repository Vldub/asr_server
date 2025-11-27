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

WORKDIR /app

# Install NeMo and dependencies
RUN pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
    "numpy<2.0.0" "protobuf<5.0" --index-url https://download.pytorch.org/whl/cu121

RUN pip install nemo-toolkit[asr]==2.5.3 \
    omegaconf>=2.3.0 \
    hydra-core>=1.3.0 \
    fastapi>=0.104.0 \
    uvicorn[standard]>=0.24.0 \
    websockets>=12.0 \
    librosa>=0.10.0 \
    soundfile>=0.12.0

COPY *.py ./

EXPOSE 8765

CMD ["python", "server.py", "--model", "/app/model.nemo", "--port", "8765", "--host", "0.0.0.0"]



