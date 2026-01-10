# Triton Inference Server для Streaming ASR

Интеграция NeMo ASR модели с NVIDIA Triton Inference Server для высокопроизводительного streaming распознавания речи.

## 🚀 Преимущества Triton

- **Dynamic Batching** — автоматическая группировка запросов для GPU
- **Sequence Batching** — поддержка streaming с сохранением состояния
- **TensorRT оптимизация** — до 3-5x ускорение на GPU
- **Multi-GPU** — автоматическое распределение нагрузки
- **Prometheus метрики** — встроенный мониторинг
- **gRPC/HTTP API** — стандартные протоколы

## 📁 Структура

```
triton/
├── model_repository/
│   └── streaming_asr/
│       ├── 1/
│       │   └── model.py      # Python backend для streaming
│       └── config.pbtxt      # Конфигурация модели
├── scripts/
│   └── export_model.py       # Экспорт в ONNX (опционально)
├── client.py                 # gRPC клиент
├── docker-compose.yml        # Docker Compose
├── Dockerfile.client         # Образ клиента
├── pyproject.toml
└── README.md
```

## 🏃 Быстрый старт

### 1. Настройка модели

Укажите путь к вашей NeMo модели в `docker-compose.yml`:

```yaml
volumes:
  - /path/to/your/model.nemo:/models/model.nemo:ro
```

### 2. Запуск Triton сервера

```bash
# GPU версия
docker compose up -d triton-server

# CPU версия
docker compose --profile cpu up -d triton-server-cpu
```

### 3. Проверка работы

```bash
# Health check
curl http://localhost:8000/v2/health/ready

# Информация о модели
curl http://localhost:8000/v2/models/streaming_asr
```

### 4. Использование клиента

```bash
# Установка зависимостей
pip install tritonclient[grpc] soundfile numpy

# Транскрипция файла
python client.py --server localhost:8001 --audio ../client/audio/test.wav

# Транскрипция с микрофона
pip install pyaudio
python client.py --server localhost:8001 --microphone
```

## 📡 API

### gRPC Endpoint

```
localhost:8001
```

### Streaming ASR

Для streaming используется Sequence Batching:

```python
import tritonclient.grpc.aio as grpcclient

client = grpcclient.InferenceServerClient("localhost:8001")

# Начало сессии
result = await client.infer(
    model_name="streaming_asr",
    inputs=[audio_input],
    sequence_id=session_id,
    sequence_start=True,  # Первый чанк
    sequence_end=False,
)

# Промежуточные чанки
result = await client.infer(
    model_name="streaming_asr",
    inputs=[audio_input],
    sequence_id=session_id,
    sequence_start=False,
    sequence_end=False,
)

# Завершение сессии
result = await client.infer(
    model_name="streaming_asr",
    inputs=[audio_input],
    sequence_id=session_id,
    sequence_start=False,
    sequence_end=True,  # Последний чанк
)
```

### HTTP REST API

```bash
# Простой инференс (не streaming)
curl -X POST http://localhost:8000/v2/models/streaming_asr/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [{
      "name": "audio_signal",
      "shape": [16000],
      "datatype": "FP32",
      "data": [0.1, 0.2, ...]
    }]
  }'
```

## 📊 Мониторинг

### Prometheus метрики

```bash
curl http://localhost:8002/metrics
```

Основные метрики:
- `nv_inference_request_success` — успешные запросы
- `nv_inference_request_failure` — ошибки
- `nv_inference_compute_infer_duration_us` — время инференса
- `nv_inference_queue_duration_us` — время в очереди

### Пример Grafana dashboard

```promql
# Throughput (req/s)
rate(nv_inference_request_success{model="streaming_asr"}[1m])

# Latency (ms)
histogram_quantile(0.95, rate(nv_inference_compute_infer_duration_us{model="streaming_asr"}[1m])) / 1000
```

## ⚡ Оптимизация

### TensorRT (GPU)

Для максимальной производительности экспортируйте модель в TensorRT:

```bash
# Экспорт в ONNX
python scripts/export_model.py --model /path/to/model.nemo --output-dir model_repository

# Triton автоматически сконвертирует в TensorRT при загрузке
```

### Параметры config.pbtxt

```protobuf
# Увеличение batch размера
max_batch_size: 16

# Dynamic batching
dynamic_batching {
  preferred_batch_size: [ 4, 8, 16 ]
  max_queue_delay_microseconds: 100000
}

# Несколько GPU инстансов
instance_group [
  { count: 2, kind: KIND_GPU, gpus: [ 0, 1 ] }
]
```

## 🔧 Troubleshooting

### Модель не загружается

```bash
# Проверьте логи
docker compose logs triton-server

# Проверьте конфигурацию
curl http://localhost:8000/v2/models/streaming_asr/config
```

### Ошибка sequence batching

Убедитесь что:
1. `sequence_id` уникален для каждой сессии
2. `sequence_start=True` только для первого чанка
3. `sequence_end=True` только для последнего чанка

### Низкая производительность

1. Проверьте использование GPU: `nvidia-smi`
2. Включите TensorRT оптимизацию
3. Увеличьте batch size
4. Используйте FP16 precision

## 📈 Ожидаемая производительность

| Конфигурация | RTF (1 сессия) | RTF (10 сессий) | Throughput |
|--------------|----------------|-----------------|------------|
| CPU (PyTorch) | 0.25 | 0.70 | 5 req/s |
| GPU (PyTorch) | 0.08 | 0.30 | 15 req/s |
| GPU + Triton | 0.04 | 0.12 | 40 req/s |
| GPU + TensorRT | 0.02 | 0.08 | 60 req/s |

## 📚 Документация

- [Triton Inference Server](https://github.com/triton-inference-server/server)
- [Triton Python Backend](https://github.com/triton-inference-server/python_backend)
- [NeMo ASR](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/intro.html)



