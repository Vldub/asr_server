#!/usr/bin/env python3
"""
Имитация реального голосового потока.
Отправляет аудио в реальном времени с задержками, как с микрофона.
"""

import argparse
import asyncio
import sys
import time
import uuid

import numpy as np
import soundfile as sf


async def simulate_realtime_stream(
    server_url: str,
    audio_file: str,
    chunk_size_ms: int = 100,
    realtime_factor: float = 1.0
):
    """
    Имитация реального голосового потока.
    
    Args:
        server_url: URL Triton сервера
        audio_file: Путь к аудио файлу
        chunk_size_ms: Размер чанка в мс
        realtime_factor: 1.0 = реальное время, 0.5 = в 2 раза быстрее
    """
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
    chunk_delay = (chunk_size_ms / 1000.0) * realtime_factor
    
    print(f"⏱️  Длительность аудио: {audio_duration:.2f}s")
    print(f"📊 Чанков: {total_chunks} (по {chunk_size_ms}ms)")
    print(f"🔌 Сервер: {server_url}")
    print(f"⚡ Realtime factor: {realtime_factor}x")
    print()
    
    # Подключение
    client = grpcclient.InferenceServerClient(url=server_url)
    
    if not await client.is_server_live():
        print("❌ Сервер недоступен!")
        return
    
    sequence_id = int(uuid.uuid4().int & 0xFFFFFFFF)
    
    print("=" * 70)
    print("🎤 ИМИТАЦИЯ ГОЛОСОВОГО ПОТОКА")
    print("=" * 70)
    print()
    print("Формат: [Время аудио] [Время обработки] Транскрипция")
    print("-" * 70)
    print()
    
    stream_start = time.perf_counter()
    is_first = True
    chunk_num = 0
    last_transcription = ""
    last_display_time = 0
    
    for i in range(0, len(audio_data), chunk_size_samples):
        chunk = audio_data[i:i + chunk_size_samples].astype(np.float32)
        chunk_num += 1
        is_last = (i + chunk_size_samples >= len(audio_data))
        
        # Время аудио
        audio_time = (i + len(chunk)) / sample_rate
        
        # Имитация реального времени - ждём перед отправкой
        target_time = (chunk_num * chunk_size_ms / 1000.0) * realtime_factor
        elapsed = time.perf_counter() - stream_start
        if elapsed < target_time:
            await asyncio.sleep(target_time - elapsed)
        
        # Отправляем чанк
        send_start = time.perf_counter()
        
        audio_input = grpcclient.InferInput("audio_signal", chunk.shape, "FP32")
        audio_input.set_data_from_numpy(chunk)
        
        result = await client.infer(
            model_name="streaming_asr",
            inputs=[audio_input],
            sequence_id=sequence_id,
            sequence_start=is_first,
            sequence_end=is_last,
        )
        
        send_time = time.perf_counter() - send_start
        is_first = False
        
        transcription = result.as_numpy("transcription")[0]
        if isinstance(transcription, bytes):
            transcription = transcription.decode("utf-8")
        
        elapsed = time.perf_counter() - stream_start
        
        # Отображаем прогресс и транскрипцию
        progress = audio_time / audio_duration * 100
        
        # Визуальная полоса прогресса
        bar_width = 20
        filled = int(bar_width * progress / 100)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        # Статус строка
        status = f"\r🎙️ [{bar}] {audio_time:5.2f}s/{audio_duration:.2f}s | ⏱️ {elapsed:5.2f}s | 📡 {send_time*1000:3.0f}ms"
        sys.stdout.write(status)
        sys.stdout.flush()
        
        # Если транскрипция изменилась - показываем на новой строке
        if transcription and transcription != last_transcription:
            # Показываем что изменилось
            if len(transcription) > len(last_transcription):
                added = transcription[len(last_transcription):].strip()
                if added and (elapsed - last_display_time) > 0.1:  # Не чаще чем раз в 100ms
                    print()
                    print(f"   📝 +\"{added}\"")
                    last_display_time = elapsed
            else:
                # Полное изменение
                print()
                print(f"   🔄 \"{transcription}\"")
                last_display_time = elapsed
            
            last_transcription = transcription
    
    stream_end = time.perf_counter()
    total_time = stream_end - stream_start
    
    print()
    print()
    print("=" * 70)
    print("✅ ПОТОК ЗАВЕРШЁН")
    print("=" * 70)
    print()
    print(f"📝 Финальная транскрипция:")
    print(f"   \"{last_transcription}\"")
    print()
    print(f"⏱️  Время стриминга: {total_time:.2f}s")
    print(f"🎵 Длительность аудио: {audio_duration:.2f}s")
    print(f"📊 Latency (относительно realtime): {(total_time - audio_duration * realtime_factor):.2f}s")
    
    if realtime_factor == 1.0:
        print(f"🚀 Обработка успевает за реальным временем: {'✅ Да' if total_time <= audio_duration * 1.1 else '❌ Нет'}")


async def run_multiple_scenarios(server_url: str, audio_file: str):
    """Запуск нескольких сценариев."""
    
    print("=" * 70)
    print("🎬 ТЕСТ СЦЕНАРИЕВ ГОЛОСОВОГО ПОТОКА")
    print("=" * 70)
    print()
    
    scenarios = [
        {"name": "Реальное время (100ms чанки)", "chunk_ms": 100, "rtf": 1.0},
        {"name": "Реальное время (200ms чанки)", "chunk_ms": 200, "rtf": 1.0},
        {"name": "Ускоренное x2 (100ms чанки)", "chunk_ms": 100, "rtf": 0.5},
        {"name": "Ускоренное x5 (100ms чанки)", "chunk_ms": 100, "rtf": 0.2},
    ]
    
    results = []
    
    for scenario in scenarios:
        print(f"\n{'='*70}")
        print(f"📋 Сценарий: {scenario['name']}")
        print(f"{'='*70}\n")
        
        await simulate_realtime_stream(
            server_url=server_url,
            audio_file=audio_file,
            chunk_size_ms=scenario["chunk_ms"],
            realtime_factor=scenario["rtf"]
        )
        
        await asyncio.sleep(1)


def main():
    parser = argparse.ArgumentParser(description="Имитация голосового потока")
    parser.add_argument("--server", default="localhost:8001", help="Triton gRPC URL")
    parser.add_argument("--audio", required=True, help="Путь к аудио файлу")
    parser.add_argument("--chunk-size", type=int, default=100, help="Размер чанка в мс")
    parser.add_argument("--realtime", type=float, default=1.0, 
                       help="Realtime factor: 1.0=реальное время, 0.5=x2 быстрее")
    parser.add_argument("--scenarios", action="store_true", 
                       help="Запустить все сценарии")
    
    args = parser.parse_args()
    
    if args.scenarios:
        asyncio.run(run_multiple_scenarios(
            server_url=args.server,
            audio_file=args.audio
        ))
    else:
        asyncio.run(simulate_realtime_stream(
            server_url=args.server,
            audio_file=args.audio,
            chunk_size_ms=args.chunk_size,
            realtime_factor=args.realtime
        ))


if __name__ == "__main__":
    main()

