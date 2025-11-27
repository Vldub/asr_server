#!/usr/bin/env python3
"""
Клиент для онлайн стримингового ASR сервера.

Использование:
    # Транскрипция аудио файла
    python client.py --server ws://localhost:8765 --audio audio.wav
    
    # Транскрипция с микрофона
    python client.py --server ws://localhost:8765 --microphone
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import websockets
from websockets.exceptions import ConnectionClosed

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def transcribe_audio_file(server_url: str, audio_file: str, chunk_size_ms: int = 100, output_file: str = None):
    """Транскрипция аудио файла через стриминг."""
    import soundfile as sf
    
    logger.info(f"Загрузка аудио файла: {audio_file}")
    audio_data, sample_rate = sf.read(audio_file)
    
    # Конвертация стерео в моно (если нужно)
    if len(audio_data.shape) > 1:
        if audio_data.shape[1] == 2:
            logger.info("Конвертация стерео в моно")
            audio_data = np.mean(audio_data, axis=1)
        else:
            # Берем первый канал для многоканального аудио
            logger.info(f"Многоканальное аудио ({audio_data.shape[1]} каналов), берем первый канал")
            audio_data = audio_data[:, 0]
    
    # Нормализация в float32
    if audio_data.dtype != np.float32:
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_data = audio_data.astype(np.float32) / 2147483648.0
        else:
            audio_data = audio_data.astype(np.float32)
    
    # Нормализация амплитуды (если нужно)
    max_val = np.abs(audio_data).max()
    if max_val > 1.0:
        logger.warning(f"Амплитуда превышает 1.0 (max={max_val:.3f}), нормализация")
        audio_data = audio_data / max_val
    
    logger.info(f"Sample rate: {sample_rate}, Длительность: {len(audio_data)/sample_rate:.2f}s")
    
    chunk_size_samples = int((chunk_size_ms / 1000.0) * sample_rate)
    
    async with websockets.connect(server_url) as websocket:
        session_id = f"file_{int(time.time() * 1000)}"
        
        logger.info(f"Создание сессии {session_id}...")
        await websocket.send(json.dumps({
            "action": "start_session",
            "session_id": session_id,
            "sample_rate": sample_rate
        }))
        
        response = await websocket.recv()
        data = json.loads(response)
        logger.info(f"Ответ сервера: {data}")
        
        # Проверяем правильный формат ответа от сервера
        if data.get("type") == "status" and data.get("status") == "session_created":
            logger.info(f"Сессия успешно создана: {data.get('session_id')}")
        elif data.get("status") == "session_created":
            # Поддержка старого формата для обратной совместимости
            logger.info(f"Сессия успешно создана: {data.get('session_id')}")
        else:
            logger.error(f"Ошибка создания сессии: {data}")
            return
        
        transcriptions = []
        transcription_count = 0
        final_transcription = ""  # Отдельная переменная для финальной транскрипции
        
        async def receive_transcriptions():
            """Получение транскрипций от сервера."""
            nonlocal transcription_count, final_transcription
            try:
                async for message in websocket:
                    # Логируем все входящие сообщения для отладки
                    try:
                        if isinstance(message, bytes):
                            logger.debug(f"Получено бинарное сообщение: {len(message)} байт")
                            continue
                        
                        data = json.loads(message)
                        logger.debug(f"Получено сообщение от сервера: {data}")
                        
                        # Проверяем различные возможные форматы транскрипций
                        text = None
                        
                        if data.get("type") == "transcription":
                            text = data.get("text", "") or data.get("transcription", "")
                            if text:
                                transcription_count += 1
                                logger.info(f"\n{'='*60}")
                                logger.info(f"📝 ТРАНСКРИПЦИЯ #{transcription_count}: {text}")
                                logger.info(f"{'='*60}\n")
                                transcriptions.append(text)
                            else:
                                logger.warning(f"Получена пустая транскрипция: {data}")
                        
                        # Проверяем наличие промежуточных транскрипций в других полях
                        elif "transcription" in data and data.get("type") != "status":
                            text = data.get("transcription", "")
                            if text:
                                transcription_count += 1
                                logger.info(f"\n{'='*60}")
                                logger.info(f"📝 ТРАНСКРИПЦИЯ #{transcription_count}: {text}")
                                logger.info(f"{'='*60}\n")
                                transcriptions.append(text)
                        
                        elif data.get("type") == "error":
                            logger.error(f"Ошибка сервера: {data.get('error')}")
                        
                        elif data.get("type") == "status":
                            status = data.get("status", "")
                            logger.info(f"Статус: {status}")
                            
                            # Проверяем наличие промежуточной транскрипции в статусе
                            if "transcription" in data:
                                text = data.get("transcription", "")
                                if text:
                                    transcription_count += 1
                                    logger.info(f"\n{'='*60}")
                                    logger.info(f"📝 ТРАНСКРИПЦИЯ #{transcription_count}: {text}")
                                    logger.info(f"{'='*60}\n")
                                    transcriptions.append(text)
                            
                            # Проверяем наличие final_transcription в сообщении session_closed
                            if status == "session_closed":
                                final_text = data.get("final_transcription", "") or data.get("transcription", "")
                                if final_text:
                                    transcription_count += 1
                                    logger.info(f"\n{'='*60}")
                                    logger.info(f"📝 ФИНАЛЬНАЯ ТРАНСКРИПЦИЯ #{transcription_count}: {final_text}")
                                    logger.info(f"{'='*60}\n")
                                    # Сохраняем финальную транскрипцию отдельно
                                    final_transcription = final_text
                                    # Также добавляем в список для отображения всех транскрипций
                                    transcriptions.append(final_text)
                                else:
                                    logger.warning("Сервер вернул пустую финальную транскрипцию")
                        
                        else:
                            # Проверяем, может ли быть транскрипция в любом поле
                            for key in ["text", "transcription", "result", "output"]:
                                if key in data and data[key]:
                                    text = str(data[key])
                                    if text and text.strip():
                                        transcription_count += 1
                                        logger.info(f"\n{'='*60}")
                                        logger.info(f"📝 ТРАНСКРИПЦИЯ #{transcription_count} (из поля '{key}'): {text}")
                                        logger.info(f"{'='*60}\n")
                                        transcriptions.append(text)
                                        break
                            
                            if not text:
                                # Логируем неизвестные типы сообщений только если нет транскрипции
                                logger.debug(f"Получено сообщение: {data}")
                    
                    except json.JSONDecodeError as e:
                        logger.warning(f"Не удалось распарсить JSON сообщение: {message[:100]}... Ошибка: {e}")
            
            except ConnectionClosed:
                logger.info("Соединение закрыто")
            except Exception as e:
                logger.error(f"Ошибка при получении сообщений: {e}", exc_info=True)
        
        receive_task = asyncio.create_task(receive_transcriptions())
        
        logger.info("Отправка аудио чанков...")
        total_chunks = (len(audio_data) + chunk_size_samples - 1) // chunk_size_samples
        
        for i in range(0, len(audio_data), chunk_size_samples):
            chunk = audio_data[i:i + chunk_size_samples]
            chunk_num = i // chunk_size_samples + 1
            
            await websocket.send(chunk.tobytes())
            
            # Показываем прогресс и текущее количество полученных транскрипций
            if chunk_num % 10 == 0:
                logger.info(f"Отправлено {chunk_num}/{total_chunks} чанков | Получено транскрипций: {transcription_count}")
            elif chunk_num % 5 == 0:
                logger.debug(f"Отправлен чанк {chunk_num}/{total_chunks} | Транскрипций: {transcription_count}")
            
            # Небольшая задержка для предотвращения перегрузки, но не слишком большая
            # Для файлов отправляем быстрее, чтобы сервер мог накапливать данные
            await asyncio.sleep(0.01)  # 10ms задержка вместо chunk_size_ms
        
        logger.info("Все чанки отправлены. Завершение сессии...")
        
        await websocket.send(json.dumps({
            "action": "end_session"
        }))
        
        # Ждем получения всех транскрипций от сервера
        # Увеличиваем время ожидания и проверяем, есть ли еще сообщения
        logger.info("Ожидание финальных транскрипций от сервера...")
        
        # Ждем до 10 секунд для получения всех транскрипций
        for _ in range(20):  # 20 * 0.5 = 10 секунд
            await asyncio.sleep(0.5)
            # Проверяем, не закрыто ли соединение
            try:
                # Проверяем состояние соединения через закрытие
                if websocket.close_code is not None:
                    logger.info("Соединение закрыто сервером")
                    break
            except:
                pass
        
        # Даем еще немного времени на обработку последних сообщений
        await asyncio.sleep(1)
        
        # Отменяем задачу получения, если она еще работает
        if not receive_task.done():
            receive_task.cancel()
            try:
                await receive_task
            except asyncio.CancelledError:
                pass
        
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 ИТОГО получено транскрипций: {len(transcriptions)}")
        logger.info(f"{'='*60}\n")
        
        if transcriptions:
            logger.info(f"\n{'='*60}")
            logger.info(f"📝 ВСЕ ТРАНСКРИПЦИИ:")
            logger.info(f"{'='*60}")
            for idx, trans in enumerate(transcriptions, 1):
                logger.info(f"{idx}. {trans}")
            logger.info(f"{'='*60}\n")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"📄 ФИНАЛЬНАЯ ТРАНСКРИПЦИЯ:")
        logger.info(f"{'='*60}")
        
        # Используем финальную транскрипцию, если она есть, иначе берем последнюю из списка
        if not final_transcription and transcriptions:
            final_transcription = transcriptions[-1]
        
        if final_transcription:
            logger.info(f"{final_transcription}")
        else:
            logger.warning("Транскрипция пустая")
        logger.info(f"{'='*60}\n")
        
        # Сохранение в файл - только финальная транскрипция
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(final_transcription)
            logger.info(f"Финальная транскрипция сохранена в файл: {output_file}")
        else:
            # Автоматическое имя файла на основе входного файла
            audio_path = Path(audio_file)
            output_path = audio_path.parent / f"{audio_path.stem}_transcription.txt"
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(final_transcription)
            logger.info(f"Финальная транскрипция сохранена в файл: {output_path}")
        
        return final_transcription


async def transcribe_microphone(server_url: str, sample_rate: int = 16000, chunk_size_ms: int = 100, output_file: str = None):
    """Транскрипция с микрофона в реальном времени."""
    try:
        import pyaudio
    except ImportError:
        logger.error("pyaudio не установлен. Установите: pip install pyaudio")
        return
    
    chunk_size_samples = int((chunk_size_ms / 1000.0) * sample_rate)
    
    p = pyaudio.PyAudio()
    transcriptions = []
    final_transcription = ""  # Отдельная переменная для финальной транскрипции
    
    try:
        stream = p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=sample_rate,
            input=True,
            frames_per_buffer=chunk_size_samples
        )
        
        logger.info("Микрофон готов. Начинаю запись...")
        logger.info("Нажмите Ctrl+C для остановки")
        
        async with websockets.connect(server_url) as websocket:
            session_id = f"mic_{int(time.time() * 1000)}"
            
            await websocket.send(json.dumps({
                "action": "start_session",
                "session_id": session_id,
                "sample_rate": sample_rate
            }))
            
            response = await websocket.recv()
            data = json.loads(response)
            logger.info(f"Сессия создана: {data}")
            
            async def receive_transcriptions():
                nonlocal final_transcription
                try:
                    async for message in websocket:
                        data = json.loads(message)
                        
                        if data.get("type") == "transcription":
                            text = data.get("text", "")
                            if text:
                                print(f"\rТранскрипция: {text}" + " " * 50, end="", flush=True)
                                transcriptions.append(text)
                        
                        elif data.get("type") == "status":
                            status = data.get("status", "")
                            # Проверяем наличие final_transcription в сообщении session_closed
                            if status == "session_closed":
                                final_text = data.get("final_transcription", "") or data.get("transcription", "")
                                if final_text:
                                    final_transcription = final_text
                                    logger.info(f"\nФинальная транскрипция: {final_text}")
                        
                        elif data.get("type") == "error":
                            logger.error(f"Ошибка: {data.get('error')}")
                except ConnectionClosed:
                    logger.info("\nСоединение закрыто")
            
            receive_task = asyncio.create_task(receive_transcriptions())
            
            try:
                while True:
                    audio_data = stream.read(chunk_size_samples, exception_on_overflow=False)
                    audio_array = np.frombuffer(audio_data, dtype=np.float32)
                    
                    await websocket.send(audio_array.tobytes())
                    await asyncio.sleep(0.01)
            
            except KeyboardInterrupt:
                logger.info("\nОстановка записи...")
                await websocket.send(json.dumps({"action": "end_session"}))
                await asyncio.sleep(1)
                receive_task.cancel()
        
        stream.stop_stream()
        
        # Сохранение результатов - только финальная транскрипция
        # Используем финальную транскрипцию, если она есть, иначе берем последнюю из списка
        if not final_transcription and transcriptions:
            final_transcription = transcriptions[-1]
        
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(final_transcription)
            logger.info(f"Финальная транскрипция сохранена в файл: {output_file}")
        else:
            # Автоматическое имя файла с timestamp
            output_path = Path(f"microphone_transcription_{int(time.time())}.txt")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(final_transcription)
            logger.info(f"Финальная транскрипция сохранена в файл: {output_path}")
        
    finally:
        stream.close()
        p.terminate()
        logger.info("Микрофон закрыт")


def main():
    parser = argparse.ArgumentParser(description="Клиент для стримингового ASR сервера")
    parser.add_argument("--server", required=True, help="WebSocket URL сервера (ws://host:port/ws/transcribe)")
    parser.add_argument("--audio", help="Путь к аудио файлу для транскрипции")
    parser.add_argument("--microphone", action="store_true", help="Использовать микрофон")
    parser.add_argument("--chunk-size", type=int, default=100, help="Размер чанка в миллисекундах")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Частота дискретизации для микрофона")
    parser.add_argument("--output", help="Путь к файлу для сохранения результатов (по умолчанию: автоматически)")
    parser.add_argument("--debug", action="store_true", help="Включить детальное логирование для отладки")
    
    args = parser.parse_args()
    
    # Включаем debug режим если нужно
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.debug("Включен режим отладки")
    
    if not args.server.startswith("ws://") and not args.server.startswith("wss://"):
        args.server = f"ws://{args.server}"
    
    if not args.server.endswith("/ws/transcribe"):
        args.server = f"{args.server}/ws/transcribe"
    
    logger.info(f"Подключение к серверу: {args.server}")
    
    if args.audio:
        if not Path(args.audio).exists():
            logger.error(f"Файл не найден: {args.audio}")
            sys.exit(1)
        
        asyncio.run(transcribe_audio_file(args.server, args.audio, args.chunk_size, args.output))
    
    elif args.microphone:
        asyncio.run(transcribe_microphone(args.server, args.sample_rate, args.chunk_size, args.output))
    
    else:
        parser.error("Необходимо указать --audio или --microphone")


if __name__ == "__main__":
    main()



