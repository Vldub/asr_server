#!/usr/bin/env python3
"""
Измеряет WER между транскрипциями при разных размерах чанков.
"""

import argparse
import asyncio
import time
import uuid
from itertools import combinations

import numpy as np
import soundfile as sf


def calculate_wer(reference: str, hypothesis: str) -> tuple[float, int, int, int, int]:
    """
    Вычисляет Word Error Rate.
    
    Returns:
        (wer, substitutions, deletions, insertions, ref_words)
    """
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()
    
    if len(ref_words) == 0:
        return (1.0 if len(hyp_words) > 0 else 0.0), 0, 0, len(hyp_words), 0
    
    # Динамическое программирование для расстояния Левенштейна
    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1), dtype=np.int32)
    
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j
    
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = min(
                    d[i-1][j] + 1,      # deletion
                    d[i][j-1] + 1,      # insertion
                    d[i-1][j-1] + 1     # substitution
                )
    
    # Backtrack для подсчёта S, D, I
    i, j = len(ref_words), len(hyp_words)
    s, d_count, ins = 0, 0, 0
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_words[i-1] == hyp_words[j-1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and d[i][j] == d[i-1][j-1] + 1:
            s += 1
            i -= 1
            j -= 1
        elif i > 0 and d[i][j] == d[i-1][j] + 1:
            d_count += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    
    wer = (s + d_count + ins) / len(ref_words)
    return wer, s, d_count, ins, len(ref_words)


async def get_transcription(server_url: str, audio_data: np.ndarray, sample_rate: int, chunk_size_ms: int) -> str:
    """Получает транскрипцию с заданным размером чанков."""
    import tritonclient.grpc.aio as grpcclient
    
    client = grpcclient.InferenceServerClient(url=server_url)
    sequence_id = int(uuid.uuid4().int & 0xFFFFFFFF)
    chunk_size_samples = int((chunk_size_ms / 1000.0) * sample_rate)
    
    is_first = True
    last_transcription = ""
    
    for i in range(0, len(audio_data), chunk_size_samples):
        chunk = audio_data[i:i + chunk_size_samples].astype(np.float32)
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
    
    return last_transcription


async def run_wer_test(server_url: str, audio_file: str, chunk_sizes: list[int]):
    """Запуск теста WER."""
    
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
    
    # Получаем транскрипции
    print("📝 Получение транскрипций...")
    results = {}
    
    for chunk_ms in chunk_sizes:
        transcription = await get_transcription(server_url, audio_data, sample_rate, chunk_ms)
        results[chunk_ms] = transcription
        print(f"   {chunk_ms:>5}ms: {transcription}")
        await asyncio.sleep(0.3)
    
    print()
    print("=" * 80)
    print("📊 WER МЕЖДУ ТРАНСКРИПЦИЯМИ")
    print("=" * 80)
    print()
    
    # Матрица WER
    print("Матрица WER (%):")
    print()
    
    # Заголовок
    header = "        " + "".join(f"{cs:>8}ms" for cs in chunk_sizes)
    print(header)
    print("-" * len(header))
    
    wer_values = []
    
    for cs1 in chunk_sizes:
        row = f"{cs1:>5}ms |"
        for cs2 in chunk_sizes:
            if cs1 == cs2:
                row += "      -  "
            else:
                wer, s, d, ins, ref_len = calculate_wer(results[cs1], results[cs2])
                wer_values.append(wer)
                row += f"  {wer*100:5.1f}% "
        print(row)
    
    print()
    
    # Статистика
    if wer_values:
        avg_wer = np.mean(wer_values)
        max_wer = np.max(wer_values)
        min_wer = np.min(wer_values)
        
        print(f"📈 Статистика WER между парами:")
        print(f"   Минимум: {min_wer*100:.1f}%")
        print(f"   Среднее: {avg_wer*100:.1f}%")
        print(f"   Максимум: {max_wer*100:.1f}%")
    
    # Сравнение с базовой (самый большой чанк)
    print()
    print(f"📊 WER относительно эталона ({max(chunk_sizes)}ms чанки):")
    print()
    
    reference = results[max(chunk_sizes)]
    
    for cs in chunk_sizes:
        if cs == max(chunk_sizes):
            print(f"   {cs:>5}ms: ЭТАЛОН")
        else:
            wer, s, d, ins, ref_len = calculate_wer(reference, results[cs])
            words = len(results[cs].split())
            print(f"   {cs:>5}ms: WER={wer*100:5.1f}% (S={s}, D={d}, I={ins}, слов={words})")
    
    print()
    print("📝 Легенда: S=замены, D=удаления, I=вставки")


def main():
    parser = argparse.ArgumentParser(description="Тест WER вариативности")
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
    
    asyncio.run(run_wer_test(
        server_url=args.server,
        audio_file=args.audio,
        chunk_sizes=args.chunk_sizes
    ))


if __name__ == "__main__":
    main()


