#!/usr/bin/env python3
"""
Тест влияния размера чанков клиента на качество транскрипции и RTF.
Буфер на сервере накапливает до оптимального размера (1040ms),
поэтому транскрипция должна быть одинаковой независимо от размера чанков.
"""

import argparse
import asyncio
import time
import uuid

import numpy as np
import soundfile as sf


async def transcribe_with_chunk_size(
    server_url: str,
    audio_data: np.ndarray,
    sample_rate: int,
    chunk_size_ms: int
) -> tuple[str, float, int]:
    """
    Транскрипция с заданным размером чанков.
    
    Returns:
        (transcription, rtf, num_chunks)
    """
    import tritonclient.grpc.aio as grpcclient
    
    client = grpcclient.InferenceServerClient(url=server_url)
    
    if not await client.is_server_live():
        raise RuntimeError("Сервер недоступен")
    
    sequence_id = int(uuid.uuid4().int & 0xFFFFFFFF)
    chunk_size_samples = int((chunk_size_ms / 1000.0) * sample_rate)
    
    start_time = time.perf_counter()
    is_first = True
    chunk_count = 0
    last_transcription = ""
    
    for i in range(0, len(audio_data), chunk_size_samples):
        chunk = audio_data[i:i + chunk_size_samples].astype(np.float32)
        chunk_count += 1
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
        
        if transcription:
            last_transcription = transcription
        
        # Имитация реального времени - небольшая задержка
        await asyncio.sleep(0.001)
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    audio_duration = len(audio_data) / sample_rate
    rtf = total_time / audio_duration
    
    return last_transcription, rtf, chunk_count


async def run_comparison(server_url: str, audio_file: str, chunk_sizes: list[int]):
    """Запуск сравнительного теста."""
    
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
    
    print(f"⏱️  Длительность: {audio_duration:.2f}s")
    print(f"🔌 Сервер: {server_url}")
    print()
    
    # Таблица результатов
    print("=" * 90)
    print(f"{'Чанк (ms)':<12} {'Чанков':<10} {'Время (s)':<12} {'RTF':<10} {'Транскрипция'}")
    print("=" * 90)
    
    results = []
    
    for chunk_ms in chunk_sizes:
        try:
            transcription, rtf, num_chunks = await transcribe_with_chunk_size(
                server_url, audio_data, sample_rate, chunk_ms
            )
            total_time = rtf * audio_duration
            
            # Сокращаем транскрипцию для таблицы
            short_trans = transcription[:40] + "..." if len(transcription) > 40 else transcription
            
            print(f"{chunk_ms:<12} {num_chunks:<10} {total_time:<12.3f} {rtf:<10.3f} {short_trans}")
            
            results.append({
                "chunk_ms": chunk_ms,
                "num_chunks": num_chunks,
                "rtf": rtf,
                "time": total_time,
                "transcription": transcription
            })
            
            # Небольшая пауза между тестами
            await asyncio.sleep(0.5)
            
        except Exception as e:
            print(f"{chunk_ms:<12} ОШИБКА: {e}")
    
    print("=" * 90)
    print()
    
    # Проверка одинаковости транскрипций
    print("📊 АНАЛИЗ РЕЗУЛЬТАТОВ")
    print("-" * 60)
    
    if results:
        base_trans = results[0]["transcription"]
        all_same = all(r["transcription"] == base_trans for r in results)
        
        if all_same:
            print("✅ Все транскрипции ОДИНАКОВЫЕ (буферизация работает корректно)")
        else:
            print("⚠️  Транскрипции ОТЛИЧАЮТСЯ:")
            for r in results:
                match = "✓" if r["transcription"] == base_trans else "✗"
                print(f"   {match} {r['chunk_ms']}ms: {r['transcription'][:60]}...")
        
        print()
        print("📈 RTF по размеру чанков:")
        
        min_rtf = min(r["rtf"] for r in results)
        max_rtf = max(r["rtf"] for r in results)
        
        for r in results:
            bar_len = int((r["rtf"] / max_rtf) * 30)
            bar = "█" * bar_len
            optimal = " ← лучший" if r["rtf"] == min_rtf else ""
            print(f"   {r['chunk_ms']:>5}ms: {bar} {r['rtf']:.3f}{optimal}")
        
        print()
        print(f"📉 Разброс RTF: {min_rtf:.3f} - {max_rtf:.3f} (разница {((max_rtf/min_rtf)-1)*100:.1f}%)")
        
        # Финальная транскрипция
        print()
        print("📝 Финальная транскрипция:")
        print(f"   \"{base_trans}\"")


def main():
    parser = argparse.ArgumentParser(description="Сравнение размеров чанков")
    parser.add_argument("--server", default="localhost:8001", help="Triton gRPC URL")
    parser.add_argument("--audio", required=True, help="Путь к аудио файлу")
    parser.add_argument(
        "--chunk-sizes", 
        type=int, 
        nargs="+",
        default=[20, 80, 200, 500, 1000],
        help="Размеры чанков для теста (мс)"
    )
    
    args = parser.parse_args()
    
    asyncio.run(run_comparison(
        server_url=args.server,
        audio_file=args.audio,
        chunk_sizes=args.chunk_sizes
    ))


if __name__ == "__main__":
    main()

