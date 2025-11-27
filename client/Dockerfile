FROM python:3.11-slim

# Установка системных зависимостей для работы с аудио
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    portaudio19-dev \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Создание рабочей директории
WORKDIR /app

# Копирование requirements и установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование кода приложения
COPY client.py .

# Установка прав на выполнение
RUN chmod +x client.py

# Точка входа
ENTRYPOINT ["python", "client.py"]

