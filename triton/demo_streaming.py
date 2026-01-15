#!/usr/bin/env python3
"""
Демонстрация streaming ASR в реальном времени.
Показывает как транскрипция обновляется по мере обработки аудио.
"""

import argparse
import asyncio
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf


async def demo_streaming(server_url: str, audio_file: str, chunk_size_ms: int = 100):
    """Демонстрация streaming с выводом в реальном времени."""
    import tritonclient.grpc.aio as grpcclient
    
    # Загрузка аудио
    print(f"📂 Загрузка: {audio_file}")
    audio_data, sample_rate = sf.read(audio_file)
    
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    if audio_data.dtype != np.float32:
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        else:
            audio_data = audio_data.astype(np.float32)
    
    audio_duration = len(audio_data) / sample_rate
    chunk_size_samples = int((chunk_size_ms / 1000.0) * sample_rate)
    total_chunks = (len(audio_data) + chunk_size_samples - 1) // chunk_size_samples
    
    print(f"⏱️  Длительность: {audio_duration:.2f}s")
    print(f"📊 Чанков: {total_chunks} (по {chunk_size_ms}ms)")
    print(f"🔌 Сервер: {server_url}")
    print()
    print("=" * 60)
    print("🎤 STREAMING ТРАНСКРИПЦИЯ")
    print("=" * 60)
    print()
    
    # Подключение
    client = grpcclient.InferenceServerClient(url=server_url)
    
    if not await client.is_server_live():
        print("❌ Сервер недоступен!")
        return
    
    sequence_id = int(uuid.uuid4().int & 0xFFFFFFFF)
    
    start_time = time.perf_counter()
    is_first = True
    chunk_num = 0
    last_transcription = ""
    
    for i in range(0, len(audio_data), chunk_size_samples):
        chunk = audio_data[i:i + chunk_size_samples].astype(np.float32)
        chunk_num += 1
        is_last = (i + chunk_size_samples >= len(audio_data))
        
        audio_input = grpcclient.InferInput("audio_signal", chunk.shape, "FP32")
        audio_input.set_data_from_numpy(chunk)
        
        result = await client.infer(
            model_name="streaming_asr",
            inputs=[audio_input],
            sequence_id=sequence_id,
            sequence_start=is_first,
            sequence_end=is_last,
        )
        
        is_first = False
        
        transcription = result.as_numpy("transcription")[0]
        if isinstance(transcription, bytes):
            transcription = transcription.decode("utf-8")
        
        elapsed = time.perf_counter() - start_time
        audio_time = (i + len(chunk)) / sample_rate
        
        # Выводим прогресс
        progress = chunk_num / total_chunks * 100
        
        # Очищаем предыдущие строки и выводим обновление
        sys.stdout.write(f"\r⏳ [{progress:5.1f}%] Аудио: {audio_time:.2f}s | Время: {elapsed:.2f}s | RTF: {elapsed/audio_time:.2f}")
        sys.stdout.flush()
        
        # Если транскрипция изменилась - показываем
        if transcription and transcription != last_transcription:
            print()  # Новая строка
            print(f"📝 [{elapsed:.2f}s] {transcription}")
            last_transcription = transcription
        
        # Небольшая задержка для имитации реального streaming
        await asyncio.sleep(0.01)
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    print()
    print()
    print("=" * 60)
    print("✅ РЕЗУЛЬТАТ")
    print("=" * 60)
    print(f"📝 Финальная транскрипция:")
    print(f"   \"{last_transcription}\"")
    print()
    print(f"⏱️  Общее время: {total_time:.2f}s")
    print(f"📊 RTF: {total_time / audio_duration:.3f}")
    if total_time < audio_duration:
        print(f"🚀 Быстрее реального времени на {(1 - total_time/audio_duration)*100:.1f}%")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Демо streaming ASR")
    parser.add_argument("--server", default="localhost:8001", help="Triton gRPC URL")
    parser.add_argument("--audio", required=True, help="Путь к аудио файлу")
    parser.add_argument("--chunk-size", type=int, default=2000, help="Размер чанка в мс")
    
    args = parser.parse_args()
    
    if not Path(args.audio).exists():
        print(f"❌ Файл не найден: {args.audio}")
        return
    
    asyncio.run(demo_streaming(
        server_url=args.server,
        audio_file=args.audio,
        chunk_size_ms=args.chunk_size
    ))


if __name__ == "__main__":
    main()


