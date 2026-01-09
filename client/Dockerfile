FROM python:3.11-slim

# Установка системных зависимостей для работы с аудио
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    portaudio19-dev \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Создание рабочей директории
WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY client.py .

# Install dependencies with uv
RUN uv pip install --system --no-cache \
    websockets>=12.0 \
    numpy>=1.24.0 \
    soundfile>=0.12.0 \
    pyaudio>=0.2.14

# Установка прав на выполнение
RUN chmod +x client.py

# Точка входа
ENTRYPOINT ["python", "client.py"]
