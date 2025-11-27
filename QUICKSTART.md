# Быстрый старт

## 1. Подготовка модели

Укажите путь к вашей NeMo модели в файле `docker-compose.yml`:

```yaml
volumes:
  - /path/to/your/model.nemo:/app/model.nemo:ro
```

Например:
```yaml
volumes:
  - /home/user/models/stt_en_fastconformer_hybrid_large_streaming_multi.nemo:/app/model.nemo:ro
```

## 2. Запуск сервера

```bash
docker compose up -d asr-server
```

Проверка работы:

```bash
curl http://localhost:8765/health
```

Ожидаемый ответ:
```json
{
  "status": "ok",
  "model_loaded": true,
  "active_sessions": 0
}
```

## 3. Использование клиента

### Вариант A: Через Docker (рекомендуется)

#### Транскрипция аудио файла

Поместите аудио файл в директорию `client/audio/` или укажите полный путь:

```bash
# Если файл в client/audio/
docker compose run --rm asr-client \
  --server ws://asr-server:8765/ws/transcribe \
  --audio /app/audio/your_audio.wav

# Или с полным путем (если файл вне контейнера)
docker compose run --rm asr-client \
  --server ws://asr-server:8765/ws/transcribe \
  --audio /home/vlad/path/to/audio.wav
```

#### Транскрипция с микрофона

```bash
docker compose run --rm -it asr-client \
  --server ws://asr-server:8765/ws/transcribe \
  --microphone
```

### Вариант B: Локально (без Docker)

```bash
cd client

# Установка зависимостей
pip install -r requirements.txt

# Транскрипция файла
python client.py \
  --server ws://localhost:8765/ws/transcribe \
  --audio /path/to/audio.wav

# Или транскрипция с микрофона
python client.py \
  --server ws://localhost:8765/ws/transcribe \
  --microphone
```

## 4. Просмотр результатов

Транскрипции выводятся в консоль в реальном времени. Финальная транскрипция также сохраняется в файл:
- Для файлов: `{имя_файла}_transcription.txt`
- Для микрофона: `microphone_transcription_{timestamp}.txt`

## Готово!

Сервер обрабатывает аудио и возвращает транскрипции в реальном времени.

## Дополнительные команды

### Остановка сервера
```bash
docker compose stop asr-server
```

### Просмотр логов сервера
```bash
docker compose logs -f asr-server
```

### Перезапуск сервера
```bash
docker compose restart asr-server
```
