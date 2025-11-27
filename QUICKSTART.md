# Быстрый старт

## 1. Подготовка модели

Скопируйте вашу NeMo модель в директорию сервера:

```bash
cp /path/to/your/model.nemo streaming_asr_server/model.nemo
```

## 2. Запуск сервера

```bash
cd streaming_asr_server
docker compose up -d
```

Проверка работы:

```bash
curl http://localhost:8765/health
```

## 3. Запуск клиента

В другом терминале:

```bash
cd streaming_asr_client

# Установка зависимостей
pip install -r requirements.txt

# Транскрипция файла
python client.py --server ws://localhost:8765 --audio /path/to/audio.wav

# Или транскрипция с микрофона
python client.py --server ws://localhost:8765 --microphone
```

## Готово!

Сервер обрабатывает аудио и возвращает транскрипции в реальном времени.



