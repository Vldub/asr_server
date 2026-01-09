#!/usr/bin/env python3
"""
WebSocket сервер для онлайн стриминговой транскрипции речи.

Запуск:
    python server.py --model /path/to/model.nemo --port 8765
"""

import argparse
import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional, Any

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator

from asr_engine import StreamingASREngine

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============== Pydantic Models ==============

class StartSessionRequest(BaseModel):
    """Запрос на создание сессии."""
    action: str = Field(..., pattern="^start_session$")
    session_id: Optional[str] = None
    sample_rate: int = Field(default=16000, ge=8000, le=48000)
    
    @field_validator('sample_rate')
    @classmethod
    def validate_sample_rate(cls, v: int) -> int:
        allowed_rates = [8000, 16000, 22050, 44100, 48000]
        if v not in allowed_rates:
            # Выбираем ближайший допустимый sample rate
            closest = min(allowed_rates, key=lambda x: abs(x - v))
            logger.warning(f"Sample rate {v} не в списке поддерживаемых, используем {closest}")
            return closest
        return v


class EndSessionRequest(BaseModel):
    """Запрос на завершение сессии."""
    action: str = Field(..., pattern="^end_session$")


class TranscriptionResponse(BaseModel):
    """Ответ с транскрипцией."""
    type: str = "transcription"
    session_id: str
    text: str
    timestamp: float


class StatusResponse(BaseModel):
    """Ответ со статусом."""
    type: str = "status"
    status: str
    session_id: Optional[str] = None
    final_transcription: Optional[str] = None


class ErrorResponse(BaseModel):
    """Ответ с ошибкой."""
    type: str = "error"
    error: str


# ============== Rate Limiting ==============

class RateLimiter:
    """Простой rate limiter для ограничения нагрузки."""
    
    def __init__(
        self,
        max_sessions: int = 10,
        max_chunk_size_bytes: int = 256 * 1024,  # 256KB (увеличено для длинных чанков)
        inactivity_timeout: float = 5.0,  # Таймаут неактивности транскрипций (секунды)
    ):
        self.max_sessions = max_sessions
        self.max_chunk_size_bytes = max_chunk_size_bytes
        self.inactivity_timeout = inactivity_timeout
    
    def check_session_limit(self, current_sessions: int) -> tuple[bool, str]:
        """Проверка лимита сессий."""
        if current_sessions >= self.max_sessions:
            return False, f"Достигнут лимит сессий: {self.max_sessions}"
        return True, ""
    
    def check_chunk_size(self, chunk_bytes: int) -> tuple[bool, str]:
        """Проверка размера чанка."""
        if chunk_bytes > self.max_chunk_size_bytes:
            return False, f"Размер чанка ({chunk_bytes} байт) превышает лимит ({self.max_chunk_size_bytes} байт)"
        return True, ""


# ============== Application State ==============

class AppState:
    """Состояние приложения."""
    def __init__(self):
        self.asr_engine: Optional[StreamingASREngine] = None
        self.config: dict = {}
        self._async_lock = asyncio.Lock()
        self.rate_limiter = RateLimiter()
    
    @property
    def async_lock(self) -> asyncio.Lock:
        return self._async_lock


# ============== Lifespan ==============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения."""
    # Startup
    import torch
    
    config = app.state.config
    device_id = config.get("device", 0)
    device = torch.device(f"cuda:{device_id}" if device_id >= 0 and torch.cuda.is_available() else "cpu")
    
    logger.info(f"Инициализация ASR сервера...")
    logger.info(f"Модель: {config['model']}")
    logger.info(f"Устройство: {device}")
    
    app.state.asr_engine = StreamingASREngine(
        model_path=config["model"],
        device=device,
        compute_dtype=torch.float32,
    )
    
    logger.info("ASR сервер готов!")
    
    yield  # Приложение работает
    
    # Shutdown
    logger.info("Завершение работы ASR сервера...")
    if app.state.asr_engine:
        # Закрываем все активные сессии
        with app.state.asr_engine.lock:
            session_ids = list(app.state.asr_engine.sessions.keys())
        for session_id in session_ids:
            try:
                app.state.asr_engine.close_session(session_id)
            except Exception as e:
                logger.error(f"Ошибка закрытия сессии {session_id}: {e}")
    logger.info("ASR сервер остановлен")


# ============== FastAPI App ==============

app = FastAPI(title="NeMo Streaming ASR Server")
app.state = AppState()


# ============== Dependency ==============

def get_asr_engine(request: Request) -> StreamingASREngine:
    """Получение ASR engine из состояния приложения."""
    engine = request.app.state.asr_engine
    if engine is None:
        raise RuntimeError("ASR Engine не инициализирован")
    return engine


# ============== Audio Validation ==============

def validate_audio_chunk(audio_bytes: bytes, expected_dtype: np.dtype = np.float32) -> tuple[np.ndarray, str]:
    """
    Валидация и конвертация аудио чанка.
    
    Returns:
        tuple: (audio_array, error_message or None)
    """
    if len(audio_bytes) == 0:
        return None, "Пустой аудио чанк"
    
    # Проверка размера (должен быть кратен размеру float32)
    if len(audio_bytes) % 4 != 0:
        return None, f"Некорректный размер аудио данных: {len(audio_bytes)} байт"
    
    try:
        audio_array = np.frombuffer(audio_bytes, dtype=expected_dtype)
    except Exception as e:
        return None, f"Ошибка декодирования аудио: {e}"
    
    # Проверка на NaN и Inf
    if np.any(np.isnan(audio_array)) or np.any(np.isinf(audio_array)):
        return None, "Аудио содержит NaN или Inf значения"
    
    # Проверка амплитуды (должна быть в разумных пределах)
    max_amplitude = np.abs(audio_array).max()
    if max_amplitude > 10.0:  # Слишком большая амплитуда
        logger.warning(f"Большая амплитуда аудио: {max_amplitude:.3f}, нормализация")
        audio_array = audio_array / max_amplitude
    
    return audio_array, None


# ============== Routes ==============

@app.get("/")
async def get():
    """HTML страница с информацией о сервере."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>NeMo Streaming ASR Server</title>
        <meta charset="utf-8">
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
            }
            #sessions {
                font-weight: bold;
                color: #0066cc;
            }
        </style>
    </head>
    <body>
        <h1>NeMo Streaming ASR Server</h1>
        <p>WebSocket сервер для онлайн транскрипции речи</p>
        <p>Используйте клиентский скрипт для подключения</p>
        <p>Активных сессий: <span id="sessions">0</span></p>
        <script>
            async function updateSessionCount() {
                try {
                    const response = await fetch("/health");
                    const data = await response.json();
                    const count = data.active_sessions || 0;
                    document.getElementById("sessions").textContent = count;
                } catch (error) {
                    console.error("Ошибка обновления счетчика сессий:", error);
                }
            }
            
            // Обновляем счетчик при загрузке страницы
            updateSessionCount();
            
            // Обновляем счетчик каждую секунду
            setInterval(updateSessionCount, 1000);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.websocket("/ws/transcribe")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint для стриминговой транскрипции."""
    await websocket.accept()
    
    asr_engine = websocket.app.state.asr_engine
    if asr_engine is None:
        await websocket.send_text(json.dumps(
            ErrorResponse(error="ASR Engine не инициализирован").model_dump()
        ))
        await websocket.close()
        return
    
    session_id: Optional[str] = None
    transcription_queue: Optional[asyncio.Queue] = None
    send_task: Optional[asyncio.Task] = None
    session_closed = False
    callback = None
    
    try:
        # Ожидаем первое сообщение с созданием сессии
        message = await websocket.receive_text()
        
        try:
            data = json.loads(message)
        except json.JSONDecodeError as e:
            await websocket.send_text(json.dumps(
                ErrorResponse(error=f"Некорректный JSON: {e}").model_dump()
            ))
            return
        
        if data.get("action") == "start_session":
            # Валидация через Pydantic
            try:
                start_request = StartSessionRequest(**data)
            except Exception as e:
                await websocket.send_text(json.dumps(
                    ErrorResponse(error=f"Ошибка валидации: {e}").model_dump()
                ))
                return
            
            # Проверка rate limiting для сессий
            rate_limiter = websocket.app.state.rate_limiter
            current_sessions = asr_engine.get_session_count()
            allowed, error_msg = rate_limiter.check_session_limit(current_sessions)
            if not allowed:
                logger.warning(f"Rate limit: {error_msg}")
                await websocket.send_text(json.dumps(
                    ErrorResponse(error=error_msg).model_dump()
                ))
                return
            
            session_id = start_request.session_id or f"session_{int(time.time() * 1000)}"
            sample_rate = start_request.sample_rate
            
            logger.info(f"Создание сессии {session_id} с sample_rate={sample_rate}")
            
            transcription_queue = asyncio.Queue()
            last_transcription_time = time.time()
            last_transcription_text = ""
            inactivity_timeout = websocket.app.state.rate_limiter.inactivity_timeout
            
            async def send_transcriptions():
                """Задача для отправки транскрипций из очереди."""
                nonlocal last_transcription_time, last_transcription_text
                while True:
                    try:
                        transcription_data = await transcription_queue.get()
                        if transcription_data is None:
                            logger.debug(f"[{session_id}] Завершение отправки транскрипций")
                            break
                        
                        new_text = transcription_data["text"]
                        
                        # Обновляем время только если транскрипция изменилась
                        if new_text != last_transcription_text:
                            last_transcription_time = time.time()
                            last_transcription_text = new_text
                        
                        response = TranscriptionResponse(
                            session_id=transcription_data["session_id"],
                            text=new_text,
                            timestamp=time.time()
                        )
                        await websocket.send_text(json.dumps(response.model_dump()))
                        logger.debug(f"[{session_id}] Транскрипция отправлена клиенту: \"{new_text}\"")
                        transcription_queue.task_done()
                    except Exception as e:
                        logger.error(f"Ошибка отправки транскрипции: {e}")
            
            send_task = asyncio.create_task(send_transcriptions())
            inactivity_triggered = False
            
            async def check_inactivity():
                """Проверяет таймаут неактивности транскрипций."""
                nonlocal inactivity_triggered
                while True:
                    await asyncio.sleep(1.0)  # Проверяем каждую секунду
                    
                    # Проверяем только если есть накопленный текст (сессия активна)
                    if last_transcription_text:
                        elapsed = time.time() - last_transcription_time
                        if elapsed >= inactivity_timeout:
                            logger.info(f"[{session_id}] Таймаут неактивности: {elapsed:.1f}s без новых транскрипций")
                            inactivity_triggered = True
                            
                            # Отправляем уведомление клиенту
                            try:
                                response = StatusResponse(
                                    status="inactivity_timeout",
                                    session_id=session_id,
                                    final_transcription=last_transcription_text
                                )
                                await websocket.send_text(json.dumps(response.model_dump()))
                            except Exception:
                                pass
                            break
            
            inactivity_task = asyncio.create_task(check_inactivity())
            
            transcription_running = False
            
            async def periodic_transcription():
                """Периодически запускает транскрипцию накопленного буфера."""
                nonlocal transcription_running, last_transcription_time, last_transcription_text
                
                while True:
                    await asyncio.sleep(0.1)  # Проверяем каждые 100ms для быстрой реакции
                    
                    if transcription_running:
                        continue
                    
                    try:
                        # Проверяем есть ли достаточно данных
                        if not asr_engine.has_pending_audio(session_id):
                            continue
                        
                        transcription_running = True
                        
                        # Выполняем транскрипцию в executor (не блокируем event loop)
                        loop = asyncio.get_event_loop()
                        transcription = await loop.run_in_executor(
                            None,
                            lambda: asr_engine.transcribe_pending(session_id)
                        )
                        
                        if transcription:
                            # Обновляем время последней транскрипции если текст изменился
                            if transcription != last_transcription_text:
                                last_transcription_time = time.time()
                                last_transcription_text = transcription
                            
                            # Отправляем транскрипцию
                            transcription_queue.put_nowait({
                                "session_id": session_id,
                                "text": transcription
                            })
                    except Exception as e:
                        logger.error(f"[{session_id}] Ошибка периодической транскрипции: {e}")
                    finally:
                        transcription_running = False
            
            transcription_task = asyncio.create_task(periodic_transcription())
            
            def callback(sid: str, text: str):
                """Callback для добавления транскрипции в очередь."""
                try:
                    logger.info(f"[{sid}] Транскрипция: \"{text}\"")
                    transcription_queue.put_nowait({
                        "session_id": sid,
                        "text": text
                    })
                except Exception as e:
                    logger.error(f"Ошибка добавления транскрипции в очередь: {e}")
            
            asr_engine.create_session(
                session_id=session_id,
                sample_rate=sample_rate,
                callback=callback
            )
            
            response = StatusResponse(
                status="session_created",
                session_id=session_id
            )
            await websocket.send_text(json.dumps(response.model_dump()))
        else:
            await websocket.send_text(json.dumps(
                ErrorResponse(error="Первое сообщение должно быть start_session").model_dump()
            ))
            return
        
        # Основной цикл обработки сообщений
        while True:
            # Проверяем флаг таймаута неактивности
            if inactivity_triggered:
                logger.info(f"[{session_id}] Закрытие сессии по таймауту неактивности")
                
                # Завершаем задачи
                if transcription_queue and send_task:
                    await transcription_queue.put(None)
                    try:
                        await asyncio.wait_for(send_task, timeout=1.0)
                    except asyncio.TimeoutError:
                        send_task.cancel()
                
                final_transcription = asr_engine.close_session(session_id)
                session_closed = True
                
                response = StatusResponse(
                    status="session_closed",
                    session_id=session_id,
                    final_transcription=final_transcription or last_transcription_text
                )
                await websocket.send_text(json.dumps(response.model_dump()))
                break
            
            try:
                message = await asyncio.wait_for(websocket.receive(), timeout=1.0)
            except asyncio.TimeoutError:
                continue  # Продолжаем цикл для проверки таймаута
            except WebSocketDisconnect:
                logger.info(f"WebSocket соединение разорвано для сессии {session_id}")
                break
            
            try:
                if "text" in message:
                    try:
                        data = json.loads(message["text"])
                    except json.JSONDecodeError:
                        continue
                    
                    if data.get("action") == "end_session":
                        # Завершаем задачу отправки
                        if transcription_queue and send_task:
                            await transcription_queue.put(None)
                            try:
                                await asyncio.wait_for(send_task, timeout=1.0)
                            except asyncio.TimeoutError:
                                send_task.cancel()
                        
                        final_transcription = asr_engine.close_session(session_id)
                        session_closed = True
                        logger.info(f"[{session_id}] Финальная транскрипция: \"{final_transcription}\"")
                        
                        response = StatusResponse(
                            status="session_closed",
                            final_transcription=final_transcription
                        )
                        await websocket.send_text(json.dumps(response.model_dump()))
                        break
                
                elif "bytes" in message:
                    audio_bytes = message["bytes"]
                    
                    # Rate limiting: проверка размера чанка
                    rate_limiter = websocket.app.state.rate_limiter
                    allowed, error_msg = rate_limiter.check_chunk_size(len(audio_bytes))
                    if not allowed:
                        logger.warning(f"[{session_id}] Rate limit: {error_msg}")
                        await websocket.send_text(json.dumps(
                            ErrorResponse(error=error_msg).model_dump()
                        ))
                        continue
                    
                    # Валидация аудио данных
                    audio_array, error = validate_audio_chunk(audio_bytes)
                    if error:
                        logger.warning(f"[{session_id}] Ошибка валидации аудио: {error}")
                        await websocket.send_text(json.dumps(
                            ErrorResponse(error=error).model_dump()
                        ))
                        continue
                    
                    # Добавляем чанк в буфер (быстро, не блокирует)
                    asr_engine.add_audio_chunk(
                        session_id=session_id,
                        audio_chunk=audio_array,
                        sample_rate=sample_rate
                    )
                    
            except WebSocketDisconnect:
                logger.info(f"WebSocket соединение разорвано для сессии {session_id}")
                break
            except Exception as e:
                logger.error(f"Ошибка обработки сообщения: {e}")
                await websocket.send_text(json.dumps(
                    ErrorResponse(error=str(e)).model_dump()
                ))
    
    except Exception as e:
        logger.error(f"Ошибка в WebSocket handler: {e}")
    finally:
        # Отменяем фоновые задачи
        for task_name, task in [('inactivity_task', locals().get('inactivity_task')), 
                                 ('transcription_task', locals().get('transcription_task'))]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        if session_id and not session_closed:
            try:
                if transcription_queue and send_task:
                    try:
                        await transcription_queue.put(None)
                        await asyncio.wait_for(send_task, timeout=1.0)
                    except asyncio.TimeoutError:
                        send_task.cancel()
                
                asr_engine.close_session(session_id)
                logger.info(f"Сессия {session_id} закрыта")
            except Exception as e:
                logger.error(f"Ошибка закрытия сессии: {e}")


@app.get("/health")
async def health_check(request: Request):
    """Проверка здоровья сервера."""
    asr_engine = request.app.state.asr_engine
    return {
        "status": "ok",
        "model_loaded": asr_engine is not None,
        "active_sessions": asr_engine.get_session_count() if asr_engine else 0
    }


@app.get("/sessions")
async def get_sessions(request: Request):
    """Получение списка активных сессий."""
    asr_engine = request.app.state.asr_engine
    if not asr_engine:
        return {"sessions": []}
    
    with asr_engine.lock:
        sessions = []
        for session_id, (config, _) in asr_engine.sessions.items():
            sessions.append({
                "session_id": session_id,
                "sample_rate": config.sample_rate,
                "created_at": config.created_at,
                "last_activity": config.last_activity,
                "idle_time": time.time() - config.last_activity
            })
    
    return {"sessions": sessions}


def create_app(model_path: str, device: int = 0) -> FastAPI:
    """Создание приложения с конфигурацией."""
    app.state.config = {
        "model": model_path,
        "device": device
    }
    # Устанавливаем lifespan после конфигурации
    app.router.lifespan_context = lifespan
    return app


def main():
    parser = argparse.ArgumentParser(description="Онлайн стриминговый ASR сервер")
    parser.add_argument("--model", required=True, help="Путь к .nemo модели")
    parser.add_argument("--port", type=int, default=8765, help="Порт для WebSocket сервера")
    parser.add_argument("--host", default="0.0.0.0", help="Хост для сервера")
    parser.add_argument("--device", type=int, default=0, help="CUDA устройство (или -1 для CPU)")
    parser.add_argument("--max-sessions", type=int, default=10, help="Максимальное количество одновременных сессий")
    parser.add_argument("--max-chunk-size", type=int, default=262144, help="Максимальный размер аудио чанка в байтах (256KB)")
    parser.add_argument("--inactivity-timeout", type=float, default=5.0, help="Таймаут неактивности транскрипций в секундах")
    args = parser.parse_args()
    
    # Конфигурируем приложение
    app.state.config = {
        "model": args.model,
        "device": args.device
    }
    app.state.rate_limiter = RateLimiter(
        max_sessions=args.max_sessions,
        max_chunk_size_bytes=args.max_chunk_size,
        inactivity_timeout=args.inactivity_timeout,
    )
    app.router.lifespan_context = lifespan
    
    logger.info(f"Запуск WebSocket сервера на {args.host}:{args.port}")
    logger.info(f"Rate limiting: max_sessions={args.max_sessions}, max_chunk_size={args.max_chunk_size}")
    logger.info(f"Inactivity timeout: {args.inactivity_timeout}s")
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )


if __name__ == "__main__":
    main()
