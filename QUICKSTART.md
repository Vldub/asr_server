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

Поместите аудио файл в директорию `client/audio/`:

```bash
# Файл должен быть в client/audio/
docker compose --profile client run --rm asr-client \
  --server ws://asr-server:8765/ws/transcribe \
  --audio /app/audio/your_audio.wav
```

#### Транскрипция с микрофона

```bash
# Примечание: для работы микрофона в Docker требуется дополнительная настройка
# Рекомендуется использовать локальный клиент для микрофона
docker compose --profile client run --rm -it asr-client \
  --server ws://asr-server:8765/ws/transcribe \
  --microphone
```

### Вариант B: Локально с uv

```bash
cd client

# Транскрипция файла (uv автоматически установит зависимости)
uv run client.py \
  --server ws://localhost:8765/ws/transcribe \
  --audio /path/to/audio.wav

# Или транскрипция с микрофона
uv run --extra microphone client.py \
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
