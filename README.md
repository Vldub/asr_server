# Streaming ASR Server

Онлайн стриминговый ASR сервер на основе NeMo с поддержкой множественных сессий.

## Структура проекта

```
.
├── server/              # Серверная часть
│   ├── server.py       # WebSocket сервер на FastAPI
│   ├── asr_engine.py   # ASR движок для обработки аудио
│   ├── Dockerfile      # Docker образ для сервера
│   └── requirements.txt
├── client/              # Клиентская часть
│   ├── client.py       # Клиент для подключения к серверу
│   ├── Dockerfile      # Docker образ для клиента
│   ├── requirements.txt
│   └── audio/          # Директория для аудио файлов
├── docker-compose.yml   # Общий docker-compose для сервера и клиента
├── README.md
└── QUICKSTART.md        # Быстрый старт
```

## Особенности

- ✅ Поддержка множественных параллельных сессий
- ✅ WebSocket API для реального времени
- ✅ Обработка аудио чанков в реальном времени
- ✅ Промежуточные транскрипции
- ✅ Автоматическое управление жизненным циклом сессий
- ✅ Docker Compose для удобного запуска сервера и клиента
- ✅ Поддержка транскрипции файлов и микрофона

## Требования

- Docker и Docker Compose (версия 3.8+)
- NVIDIA GPU с CUDA (рекомендуется для сервера)
- NeMo модель (.nemo файл)
- Для клиента: Python 3.11+ (при локальном использовании)

## Быстрый старт

См. [QUICKSTART.md](QUICKSTART.md) для подробных инструкций.

### Краткая версия:

1. **Настройте путь к модели** в `docker-compose.yml`:
   ```yaml
   volumes:
     - /path/to/your/model.nemo:/app/model.nemo:ro
   ```

2. **Запустите сервер**:
   ```bash
   docker compose up -d asr-server
   ```

3. **Проверьте работу**:
   ```bash
   curl http://localhost:8765/health
   ```

4. **Используйте клиент**:
   ```bash
   # Через Docker
   docker compose run --rm asr-client \
     --server ws://asr-server:8765/ws/transcribe \
     --audio /app/audio/your_audio.wav
   
   # Или локально
   cd client
   pip install -r requirements.txt
   python client.py --server ws://localhost:8765/ws/transcribe --audio /path/to/audio.wav
   ```

## Использование

### Запуск сервера

#### Через Docker Compose (рекомендуется)

```bash
docker compose up -d asr-server
```

#### Локально (требует установленного NeMo)

```bash
cd server
pip install -r requirements.txt
python server.py --model /path/to/model.nemo --port 8765 --host 0.0.0.0
```

### Использование клиента

#### Через Docker

```bash
# Транскрипция аудио файла
docker compose run --rm asr-client \
  --server ws://asr-server:8765/ws/transcribe \
  --audio /app/audio/your_audio.wav

# Транскрипция с микрофона
docker compose run --rm -it asr-client \
  --server ws://asr-server:8765/ws/transcribe \
  --microphone
```

**Важно:** При использовании Docker для работы с файлами:
- Поместите файлы в `client/audio/` для доступа через `/app/audio/`
- Или используйте полный путь, если файл смонтирован через volumes в `docker-compose.yml`

#### Локально (без Docker)

```bash
cd client
pip install -r requirements.txt

# Транскрипция файла
python client.py \
  --server ws://localhost:8765/ws/transcribe \
  --audio /path/to/audio.wav

# Транскрипция с микрофона
python client.py \
  --server ws://localhost:8765/ws/transcribe \
  --microphone
```

### WebSocket API

#### Подключение

```
ws://localhost:8765/ws/transcribe
```

#### Создание сессии

Отправьте JSON:
```json
{
  "action": "start_session",
  "session_id": "unique_session_id",
  "sample_rate": 16000
}
```

#### Отправка аудио

Отправьте аудио данные как binary (bytes):
- Формат: float32 numpy array
- Sample rate: должен соответствовать указанному при создании сессии

#### Получение транскрипций

Сервер отправляет транскрипции в формате:
```json
{
  "type": "transcription",
  "session_id": "unique_session_id",
  "text": "распознанный текст",
  "timestamp": 1234567890.123
}
```

#### Завершение сессии

Отправьте JSON:
```json
{
  "action": "end_session"
}
```

## REST API

### Health Check

```bash
GET /health
```

Ответ:
```json
{
  "status": "ok",
  "model_loaded": true,
  "active_sessions": 2
}
```

### Список сессий

```bash
GET /sessions
```

Ответ:
```json
{
  "sessions": [
    {
      "session_id": "session_123",
      "sample_rate": 16000,
      "created_at": 1234567890.0,
      "last_activity": 1234567895.0,
      "idle_time": 5.0
    }
  ]
}
```

## Конфигурация

### Сервер

Можно настроить через переменные окружения или аргументы командной строки:

- `--model` - путь к модели (обязательно)
- `--port` - порт сервера (по умолчанию: 8765)
- `--host` - хост (по умолчанию: 0.0.0.0)
- `--device` - CUDA устройство (0, 1, ... или -1 для CPU)

В `docker-compose.yml` можно изменить:
- Порт сервера (в секции `ports`)
- Путь к модели (в секции `volumes`)
- CUDA устройство (в `command` или `CUDA_VISIBLE_DEVICES`)

### Клиент

- `--server` - WebSocket URL сервера (обязательно, должен включать `/ws/transcribe`)
- `--audio` - путь к аудио файлу
- `--microphone` - использовать микрофон
- `--chunk-size` - размер чанка в миллисекундах (по умолчанию: 100)
- `--sample-rate` - частота дискретизации для микрофона (по умолчанию: 16000)
- `--output` - путь к файлу для сохранения результатов
- `--debug` - включить детальное логирование

## Архитектура

- `server/asr_engine.py` - ASR движок для обработки аудио с поддержкой стриминга
- `server/server.py` - WebSocket сервер на FastAPI
- `client/client.py` - Клиент для подключения к серверу (поддержка файлов и микрофона)

## Troubleshooting

### Проблема: Модель не загружается

Убедитесь что:
1. Файл модели существует и доступен
2. Модель совместима с NeMo 2.5.3
3. Достаточно памяти GPU/CPU
4. Путь к модели правильно указан в `docker-compose.yml` (секция `volumes` сервиса `asr-server`)

### Проблема: Низкая производительность

- Используйте GPU вместо CPU (убедитесь, что `CUDA_VISIBLE_DEVICES` настроен правильно)
- Уменьшите количество одновременных сессий
- Увеличьте размер чанков на стороне клиента (`--chunk-size`)

### Проблема: Сессии не закрываются

Сессии автоматически закрываются при:
- Разрыве WebSocket соединения
- Получении команды `end_session`
- Превышении времени неактивности (5 минут)

### Проблема: Клиент не может подключиться к серверу

- Убедитесь, что сервер запущен: `docker compose ps`
- Проверьте логи: `docker compose logs asr-server`
- Проверьте сеть: оба контейнера должны быть в одной сети `asr-network`
- Для локального использования клиента укажите `ws://localhost:8765/ws/transcribe`
- Убедитесь, что URL включает `/ws/transcribe` в конце

### Проблема: Файл не найден при использовании Docker клиента

- Поместите файл в `client/audio/` и используйте путь `/app/audio/filename.wav`
- Или используйте полный путь, если файл смонтирован через volumes
- Проверьте, что файл существует: `docker compose run --rm asr-client ls -la /app/audio/`

### Проблема: Ошибка при транскрипции с микрофона

- Убедитесь, что микрофон доступен в системе
- Для Docker может потребоваться дополнительная настройка (передача устройств)
- Попробуйте использовать локальный клиент вместо Docker

## Управление сервисами

### Просмотр статуса
```bash
docker compose ps
```

### Просмотр логов
```bash
# Логи сервера
docker compose logs -f asr-server

# Логи клиента (если запущен)
docker compose logs asr-client
```

### Остановка
```bash
docker compose stop
```

### Перезапуск
```bash
docker compose restart asr-server
```

### Полная остановка и удаление
```bash
docker compose down
```

## Лицензия

См. лицензию NeMo.
