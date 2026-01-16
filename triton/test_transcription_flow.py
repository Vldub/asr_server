#!/usr/bin/env python3
"""
Показывает как транскрипция меняется в процессе streaming.
"""

import argparse
import asyncio
import time
import uuid

import numpy as np
import soundfile as sf


async def show_transcription_flow(
    server_url: str,
    audio_file: str,
    chunk_size_ms: int = 200
):
    """Показывает поток транскрипции."""
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
    print("=" * 80)
    print("🎤 ПОТОК ТРАНСКРИПЦИИ")
    print("=" * 80)
    print()
    
    client = grpcclient.InferenceServerClient(url=server_url)
    
    if not await client.is_server_live():
        print("❌ Сервер недоступен!")
        return
    
    sequence_id = int(uuid.uuid4().int & 0xFFFFFFFF)
    
    start_time = time.perf_counter()
    is_first = True
    chunk_num = 0
    last_transcription = ""
    changes = []
    
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
        
        # Показываем только если транскрипция изменилась
        if transcription != last_transcription:
            # Находим что добавилось
            if transcription.startswith(last_transcription):
                added = transcription[len(last_transcription):].strip()
                if added:
                    changes.append({
                        "audio_time": audio_time,
                        "elapsed": elapsed,
                        "added": added,
                        "full": transcription
                    })
                    print(f"⏱️ {audio_time:5.2f}s | +{elapsed:.2f}s | 📝 ...{added}")
            else:
                # Транскрипция изменилась полностью
                changes.append({
                    "audio_time": audio_time,
                    "elapsed": elapsed,
                    "added": f"[изменено] {transcription}",
                    "full": transcription
                })
                print(f"⏱️ {audio_time:5.2f}s | +{elapsed:.2f}s | 🔄 {transcription}")
            
            last_transcription = transcription
        
        await asyncio.sleep(0.001)
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    print()
    print("=" * 80)
    print("✅ ИТОГ")
    print("=" * 80)
    print()
    print(f"📝 Финальная транскрипция:")
    print(f"   \"{last_transcription}\"")
    print()
    print(f"⏱️  Время обработки: {total_time:.2f}s")
    print(f"📊 RTF: {total_time / audio_duration:.3f}")
    print(f"📈 Обновлений транскрипции: {len(changes)}")
    
    if changes:
        print()
        print("📋 Хронология изменений:")
        for i, c in enumerate(changes, 1):
            print(f"   {i}. [{c['audio_time']:.1f}s] {c['added']}")


def main():
    parser = argparse.ArgumentParser(description="Поток транскрипции")
    parser.add_argument("--server", default="localhost:8001", help="Triton gRPC URL")
    parser.add_argument("--audio", required=True, help="Путь к аудио файлу")
    parser.add_argument("--chunk-size", type=int, default=200, help="Размер чанка в мс")
    
    args = parser.parse_args()
    
    asyncio.run(show_transcription_flow(
        server_url=args.server,
        audio_file=args.audio,
        chunk_size_ms=args.chunk_size
    ))


if __name__ == "__main__":
    main()


