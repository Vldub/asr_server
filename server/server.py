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
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from asr_engine import StreamingASREngine

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()
asr_engine: Optional[StreamingASREngine] = None


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
    session_id = None
    transcription_queue = None
    send_task = None
    session_closed = False
    
    try:
        message = await websocket.receive_text()
        data = json.loads(message)
        
        if data.get("action") == "start_session":
            session_id = data.get("session_id") or f"session_{int(time.time() * 1000)}"
            sample_rate = data.get("sample_rate", 16000)
            
            logger.info(f"Создание сессии {session_id} с sample_rate={sample_rate}")
            
            transcription_queue = asyncio.Queue()
            
            async def send_transcriptions():
                """Задача для отправки транскрипций из очереди."""
                while True:
                    try:
                        transcription_data = await transcription_queue.get()
                        if transcription_data is None:
                            logger.debug(f"[{session_id}] Завершение отправки транскрипций")
                            break
                        
                        response = {
                            "type": "transcription",
                            "session_id": transcription_data["session_id"],
                            "text": transcription_data["text"],
                            "timestamp": time.time()
                        }
                        await websocket.send_text(json.dumps(response))
                        logger.debug(f"[{session_id}] Транскрипция отправлена клиенту: \"{transcription_data['text']}\"")
                        transcription_queue.task_done()
                    except Exception as e:
                        logger.error(f"Ошибка отправки транскрипции: {e}")
            
            send_task = asyncio.create_task(send_transcriptions())
            
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
            
            await websocket.send_text(json.dumps({
                "type": "status",
                "status": "session_created",
                "session_id": session_id
            }))
        
        while True:
            try:
                message = await websocket.receive()
                
                if "text" in message:
                    data = json.loads(message["text"])
                    
                    if data.get("action") == "end_session":
                        if transcription_queue and send_task:
                            await transcription_queue.put(None)
                            try:
                                await asyncio.wait_for(send_task, timeout=1.0)
                            except asyncio.TimeoutError:
                                send_task.cancel()
                        
                        final_transcription = asr_engine.close_session(session_id)
                        session_closed = True
                        logger.info(f"[{session_id}] Финальная транскрипция: \"{final_transcription}\"")
                        await websocket.send_text(json.dumps({
                            "type": "status",
                            "status": "session_closed",
                            "final_transcription": final_transcription
                        }))
                        break
                
                elif "bytes" in message:
                    audio_bytes = message["bytes"]
                    audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
                    
                    with asr_engine.lock:
                        if session_id in asr_engine.sessions:
                            sample_rate = asr_engine.sessions[session_id][1].sample_rate
                        else:
                            sample_rate = 16000
                    
                    # Логирование входящего аудио-чанка
                    chunk_duration = len(audio_array) / sample_rate
                    logger.info(f"[{session_id}] Получен аудио-чанк: {len(audio_array)} samples, {chunk_duration:.3f}s, {len(audio_bytes)} bytes")
                    
                    # Накапливаем данные и периодически вызываем транскрипцию
                    transcription = asr_engine.process_audio_chunk(
                        session_id=session_id,
                        audio_chunk=audio_array,
                        sample_rate=sample_rate,
                        return_transcription=True  # Периодически вызываем транскрипцию при накоплении данных
                    )
                    
                    # Если получена транскрипция, отправляем через callback
                    if transcription and callback:
                        logger.info(f"[{session_id}] Промежуточная транскрипция: \"{transcription}\"")
                        callback(session_id, transcription)
                    
            except WebSocketDisconnect:
                logger.info(f"WebSocket соединение разорвано для сессии {session_id}")
                break
            except Exception as e:
                logger.error(f"Ошибка обработки сообщения: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "error": str(e)
                }))
    
    except Exception as e:
        logger.error(f"Ошибка в WebSocket handler: {e}")
    finally:
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
async def health_check():
    """Проверка здоровья сервера."""
    return {
        "status": "ok",
        "model_loaded": asr_engine is not None,
        "active_sessions": asr_engine.get_session_count() if asr_engine else 0
    }


@app.get("/sessions")
async def get_sessions():
    """Получение списка активных сессий."""
    if not asr_engine:
        return {"sessions": []}
    
    with asr_engine.lock:
        sessions = []
        for session_id, (_, config, _) in asr_engine.sessions.items():
            sessions.append({
                "session_id": session_id,
                "sample_rate": config.sample_rate,
                "created_at": config.created_at,
                "last_activity": config.last_activity,
                "idle_time": time.time() - config.last_activity
            })
    
    return {"sessions": sessions}


def main():
    parser = argparse.ArgumentParser(description="Онлайн стриминговый ASR сервер")
    parser.add_argument("--model", required=True, help="Путь к .nemo модели")
    parser.add_argument("--port", type=int, default=8765, help="Порт для WebSocket сервера")
    parser.add_argument("--host", default="0.0.0.0", help="Хост для сервера")
    parser.add_argument("--device", type=int, default=0, help="CUDA устройство (или -1 для CPU)")
    args = parser.parse_args()
    
    global asr_engine
    
    import torch
    device = torch.device(f"cuda:{args.device}" if args.device >= 0 and torch.cuda.is_available() else "cpu")
    
    logger.info(f"Инициализация ASR сервера...")
    logger.info(f"Модель: {args.model}")
    logger.info(f"Устройство: {device}")
    
    asr_engine = StreamingASREngine(
        model_path=args.model,
        device=device,
        compute_dtype=torch.float32,
        online_normalization=False,
        pad_and_drop_preencoded=False,
    )
    
    logger.info("ASR сервер готов!")
    logger.info(f"Запуск WebSocket сервера на {args.host}:{args.port}")
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )


if __name__ == "__main__":
    main()

