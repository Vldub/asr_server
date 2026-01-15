#!/usr/bin/env python3
"""
Клиент для Triton Inference Server с поддержкой streaming ASR.

Использование:
    # Транскрипция файла
    python client.py --server localhost:8001 --audio audio.wav
    
    # С микрофоном
    python client.py --server localhost:8001 --microphone
"""

import argparse
import asyncio
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TritonStreamingASRClient:
    """Клиент для streaming ASR через Triton gRPC."""
    
    def __init__(self, server_url: str = "localhost:8001", model_name: str = "streaming_asr"):
        """
        Args:
            server_url: URL Triton gRPC сервера (host:port)
            model_name: Имя модели в Triton
        """
        import tritonclient.grpc.aio as grpcclient
        
        self.server_url = server_url
        self.model_name = model_name
        self.client: Optional[grpcclient.InferenceServerClient] = None
        self.grpcclient = grpcclient
    
    async def connect(self):
        """Подключение к серверу."""
        self.client = self.grpcclient.InferenceServerClient(url=self.server_url)
        
        # Проверяем доступность сервера
        if not await self.client.is_server_live():
            raise RuntimeError(f"Triton сервер недоступен: {self.server_url}")
        
        if not await self.client.is_server_ready():
            raise RuntimeError("Triton сервер не готов")
        
        # Проверяем доступность модели
        if not await self.client.is_model_ready(self.model_name):
            raise RuntimeError(f"Модель {self.model_name} не готова")
        
        logger.info(f"Подключено к Triton: {self.server_url}, модель: {self.model_name}")
    
    async def transcribe_streaming(
        self,
        audio_chunks,
        sample_rate: int = 16000,
        chunk_size_ms: int = 100
    ):
        """
        Streaming транскрипция аудио.
        
        Args:
            audio_chunks: Итератор/генератор аудио чанков (np.ndarray float32)
            sample_rate: Sample rate аудио
            chunk_size_ms: Размер чанка в мс (для логирования)
            
        Yields:
            str: Промежуточные транскрипции
        """
        sequence_id = int(uuid.uuid4().int & 0xFFFFFFFF)  # 32-bit ID
        
        logger.info(f"Начало streaming сессии: {sequence_id}")
        
        is_first = True
        chunk_count = 0
        
        async for audio_chunk in audio_chunks:
            chunk_count += 1
            
            # Подготавливаем входные данные
            audio_input = self.grpcclient.InferInput(
                "audio_signal", audio_chunk.shape, "FP32"
            )
            audio_input.set_data_from_numpy(audio_chunk.astype(np.float32))
            
            # Флаги sequence
            inputs = [audio_input]
            
            # Выполняем инференс
            result = await self.client.infer(
                model_name=self.model_name,
                inputs=inputs,
                sequence_id=sequence_id,
                sequence_start=is_first,
                sequence_end=False,
            )
            
            is_first = False
            
            # Получаем транскрипцию
            transcription = result.as_numpy("transcription")[0]
            if isinstance(transcription, bytes):
                transcription = transcription.decode('utf-8')
            
            if transcription:
                yield transcription
        
        # Финальный запрос с END флагом
        empty_chunk = np.array([], dtype=np.float32)
        audio_input = self.grpcclient.InferInput(
            "audio_signal", [0], "FP32"
        )
        audio_input.set_data_from_numpy(empty_chunk)
        
        result = await self.client.infer(
            model_name=self.model_name,
            inputs=[audio_input],
            sequence_id=sequence_id,
            sequence_start=False,
            sequence_end=True,
        )
        
        final_transcription = result.as_numpy("transcription")[0]
        if isinstance(final_transcription, bytes):
            final_transcription = final_transcription.decode('utf-8')
        
        logger.info(f"Сессия завершена: {sequence_id}, чанков: {chunk_count}")
        
        if final_transcription:
            yield final_transcription


async def audio_file_chunks(audio_file: str, chunk_size_ms: int = 100):
    """Генератор аудио чанков из файла."""
    import soundfile as sf
    
    logger.info(f"Загрузка файла: {audio_file}")
    audio_data, sample_rate = sf.read(audio_file)
    
    # Конвертация в моно
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    # Конвертация в float32
    if audio_data.dtype != np.float32:
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_data = audio_data.astype(np.float32) / 2147483648.0
        else:
            audio_data = audio_data.astype(np.float32)
    
    # Нормализация
    max_val = np.abs(audio_data).max()
    if max_val > 1.0:
        audio_data = audio_data / max_val
    
    logger.info(f"Sample rate: {sample_rate}, Длительность: {len(audio_data)/sample_rate:.2f}s")
    
    chunk_size_samples = int((chunk_size_ms / 1000.0) * sample_rate)
    
    for i in range(0, len(audio_data), chunk_size_samples):
        chunk = audio_data[i:i + chunk_size_samples]
        yield chunk
        await asyncio.sleep(0.01)  # Небольшая задержка для имитации streaming


async def microphone_chunks(sample_rate: int = 16000, chunk_size_ms: int = 100):
    """Генератор аудио чанков с микрофона."""
    try:
        import pyaudio
    except ImportError:
        logger.error("pyaudio не установлен: pip install pyaudio")
        return
    
    chunk_size_samples = int((chunk_size_ms / 1000.0) * sample_rate)
    
    p = pyaudio.PyAudio()
    
    stream = p.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=sample_rate,
        input=True,
        frames_per_buffer=chunk_size_samples
    )
    
    logger.info("Микрофон готов. Нажмите Ctrl+C для остановки")
    
    try:
        while True:
            audio_data = stream.read(chunk_size_samples, exception_on_overflow=False)
            audio_chunk = np.frombuffer(audio_data, dtype=np.float32)
            yield audio_chunk
            await asyncio.sleep(0.001)
    except KeyboardInterrupt:
        logger.info("Остановка записи")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


async def transcribe_file(server_url: str, audio_file: str, chunk_size_ms: int = 100):
    """Транскрипция аудио файла."""
    client = TritonStreamingASRClient(server_url)
    await client.connect()
    
    transcriptions = []
    
    chunks = audio_file_chunks(audio_file, chunk_size_ms)
    
    async for transcription in client.transcribe_streaming(chunks, chunk_size_ms=chunk_size_ms):
        if transcription:
            transcriptions.append(transcription)
            logger.info(f"📝 Транскрипция: {transcription}")
    
    final = transcriptions[-1] if transcriptions else ""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"📄 ФИНАЛЬНАЯ ТРАНСКРИПЦИЯ: {final}")
    logger.info(f"{'='*60}\n")
    
    # Сохраняем в файл
    audio_path = Path(audio_file)
    output_path = audio_path.parent / f"{audio_path.stem}_triton_transcription.txt"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(final)
    logger.info(f"Сохранено: {output_path}")
    
    return final


async def transcribe_microphone(server_url: str, sample_rate: int = 16000, chunk_size_ms: int = 100):
    """Транскрипция с микрофона."""
    client = TritonStreamingASRClient(server_url)
    await client.connect()
    
    chunks = microphone_chunks(sample_rate, chunk_size_ms)
    
    try:
        async for transcription in client.transcribe_streaming(chunks, sample_rate, chunk_size_ms):
            if transcription:
                print(f"\r📝 {transcription}" + " " * 20, end="", flush=True)
    except KeyboardInterrupt:
        print("\n")
        logger.info("Запись остановлена")


def main():
    parser = argparse.ArgumentParser(description="Triton Streaming ASR Client")
    parser.add_argument("--server", default="localhost:8001", help="Triton gRPC URL (host:port)")
    parser.add_argument("--model", default="streaming_asr", help="Имя модели")
    parser.add_argument("--audio", help="Путь к аудио файлу")
    parser.add_argument("--microphone", action="store_true", help="Использовать микрофон")
    parser.add_argument("--chunk-size", type=int, default=80, help="Размер чанка в мс")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Sample rate для микрофона")
    
    args = parser.parse_args()
    
    if args.audio:
        if not Path(args.audio).exists():
            logger.error(f"Файл не найден: {args.audio}")
            sys.exit(1)
        asyncio.run(transcribe_file(args.server, args.audio, args.chunk_size))
    elif args.microphone:
        asyncio.run(transcribe_microphone(args.server, args.sample_rate, args.chunk_size))
    else:
        parser.error("Укажите --audio или --microphone")


if __name__ == "__main__":
    main()



