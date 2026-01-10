#!/usr/bin/env python3
"""
Нагрузочное тестирование Triton Streaming ASR.
Поддержка ModelStreamInfer для Decoupled Mode.

Использование:
    uv run --with tritonclient[grpc] --with soundfile --with numpy \
        python benchmark.py --server localhost:8001 --audio test.wav --concurrent 5 --iterations 3
"""

import argparse
import asyncio
import logging
import statistics
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
from queue import Queue

import numpy as np
import soundfile as sf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class RequestMetrics:
    """Метрики одного запроса."""
    request_id: int
    success: bool = False
    error: Optional[str] = None
    
    connection_time: float = 0.0
    time_to_first_transcription: float = 0.0
    total_processing_time: float = 0.0
    
    audio_duration: float = 0.0
    chunks_sent: int = 0
    transcriptions_received: int = 0
    final_transcription: str = ""
    
    @property
    def real_time_factor(self) -> float:
        if self.audio_duration > 0:
            return self.total_processing_time / self.audio_duration
        return 0.0


@dataclass 
class BenchmarkResults:
    """Результаты тестирования."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    metrics: List[RequestMetrics] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    
    @property
    def success_rate(self) -> float:
        if self.total_requests > 0:
            return self.successful_requests / self.total_requests * 100
        return 0.0
    
    @property
    def total_duration(self) -> float:
        return self.end_time - self.start_time
    
    @property
    def requests_per_second(self) -> float:
        if self.total_duration > 0:
            return self.successful_requests / self.total_duration
        return 0.0
    
    def get_successful_metrics(self) -> List[RequestMetrics]:
        return [m for m in self.metrics if m.success]
    
    def calculate_stats(self, values: List[float]) -> dict:
        if not values:
            return {"min": 0, "max": 0, "avg": 0, "median": 0, "p95": 0}
        sorted_values = sorted(values)
        return {
            "min": min(values),
            "max": max(values),
            "avg": statistics.mean(values),
            "median": statistics.median(values),
            "p95": sorted_values[int(len(sorted_values) * 0.95)] if len(sorted_values) > 1 else sorted_values[0],
        }


async def run_single_request_stream(
    server_url: str,
    audio_data: np.ndarray,
    sample_rate: int,
    chunk_size_ms: int,
    request_id: int,
    model_name: str = "streaming_asr"
) -> RequestMetrics:
    """Выполняет один запрос к Triton через ModelStreamInfer (для Decoupled Mode)."""
    import tritonclient.grpc.aio as grpcclient
    
    metrics = RequestMetrics(request_id=request_id)
    metrics.audio_duration = len(audio_data) / sample_rate
    
    chunk_size_samples = int((chunk_size_ms / 1000.0) * sample_rate)
    sequence_id = int(uuid.uuid4().int & 0xFFFFFFFF)
    
    start_time = time.perf_counter()
    first_transcription_time = None
    transcriptions = []
    
    try:
        # Подключение
        connect_start = time.perf_counter()
        client = grpcclient.InferenceServerClient(url=server_url)
        
        if not await client.is_server_live():
            metrics.error = "Сервер недоступен"
            return metrics
        
        metrics.connection_time = time.perf_counter() - connect_start
        
        # Генератор запросов для streaming
        async def request_generator():
            is_first = True
            for i in range(0, len(audio_data), chunk_size_samples):
                chunk = audio_data[i:i + chunk_size_samples].astype(np.float32)
                is_last = (i + chunk_size_samples >= len(audio_data))
                
                audio_input = grpcclient.InferInput("audio_signal", chunk.shape, "FP32")
                audio_input.set_data_from_numpy(chunk)
                
                yield {
                    "model_name": model_name,
                    "inputs": [audio_input],
                    "sequence_id": sequence_id,
                    "sequence_start": is_first,
                    "sequence_end": is_last,
                }
                
                is_first = False
                await asyncio.sleep(0.005)
        
        # Streaming infer
        chunks_sent = 0
        async for response in client.stream_infer(request_generator()):
            chunks_sent += 1
            result, error = response
            
            if error:
                metrics.error = str(error)
                break
            
            transcription = result.as_numpy("transcription")[0]
            if isinstance(transcription, bytes):
                transcription = transcription.decode("utf-8")
            
            if transcription:
                if first_transcription_time is None:
                    first_transcription_time = time.perf_counter()
                transcriptions.append(transcription)
        
        metrics.chunks_sent = chunks_sent
        metrics.transcriptions_received = len(transcriptions)
        metrics.final_transcription = transcriptions[-1] if transcriptions else ""
        
        if first_transcription_time:
            metrics.time_to_first_transcription = first_transcription_time - start_time
        
        if not metrics.error:
            metrics.success = True
        
    except Exception as e:
        metrics.error = str(e)
    
    metrics.total_processing_time = time.perf_counter() - start_time
    return metrics


async def run_single_request_simple(
    server_url: str,
    audio_data: np.ndarray,
    sample_rate: int,
    chunk_size_ms: int,
    request_id: int,
    model_name: str = "streaming_asr"
) -> RequestMetrics:
    """Выполняет один запрос к Triton через обычный ModelInfer."""
    import tritonclient.grpc.aio as grpcclient
    
    metrics = RequestMetrics(request_id=request_id)
    metrics.audio_duration = len(audio_data) / sample_rate
    
    chunk_size_samples = int((chunk_size_ms / 1000.0) * sample_rate)
    sequence_id = int(uuid.uuid4().int & 0xFFFFFFFF)
    
    start_time = time.perf_counter()
    first_transcription_time = None
    transcriptions = []
    
    try:
        # Подключение
        connect_start = time.perf_counter()
        client = grpcclient.InferenceServerClient(url=server_url)
        
        if not await client.is_server_live():
            metrics.error = "Сервер недоступен"
            return metrics
        
        metrics.connection_time = time.perf_counter() - connect_start
        
        # Отправка чанков
        chunks_sent = 0
        is_first = True
        
        for i in range(0, len(audio_data), chunk_size_samples):
            chunk = audio_data[i:i + chunk_size_samples].astype(np.float32)
            chunks_sent += 1
            
            is_last = (i + chunk_size_samples >= len(audio_data))
            
            audio_input = grpcclient.InferInput("audio_signal", chunk.shape, "FP32")
            audio_input.set_data_from_numpy(chunk)
            
            result = await client.infer(
                model_name=model_name,
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
                if first_transcription_time is None:
                    first_transcription_time = time.perf_counter()
                transcriptions.append(transcription)
            
            await asyncio.sleep(0.005)
        
        metrics.chunks_sent = chunks_sent
        metrics.transcriptions_received = len(transcriptions)
        metrics.final_transcription = transcriptions[-1] if transcriptions else ""
        
        if first_transcription_time:
            metrics.time_to_first_transcription = first_transcription_time - start_time
        
        metrics.success = True
        
    except Exception as e:
        metrics.error = str(e)
    
    metrics.total_processing_time = time.perf_counter() - start_time
    return metrics


async def run_benchmark(
    server_url: str,
    audio_file: str,
    concurrent: int,
    iterations: int,
    chunk_size_ms: int,
    model_name: str,
    use_stream: bool = False
) -> BenchmarkResults:
    """Запускает нагрузочное тестирование."""
    
    logger.info(f"Загрузка аудио: {audio_file}")
    audio_data, sample_rate = sf.read(audio_file)
    
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    if audio_data.dtype != np.float32:
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        else:
            audio_data = audio_data.astype(np.float32)
    
    audio_duration = len(audio_data) / sample_rate
    logger.info(f"Аудио: {audio_duration:.2f}s, sample_rate={sample_rate}")
    
    results = BenchmarkResults()
    results.total_requests = concurrent * iterations
    
    # Выбор функции запроса
    request_func = run_single_request_stream if use_stream else run_single_request_simple
    mode = "StreamInfer (Decoupled)" if use_stream else "Infer (Standard)"
    
    logger.info(f"\n{'='*60}")
    logger.info(f"🚀 НАГРУЗОЧНОЕ ТЕСТИРОВАНИЕ TRITON")
    logger.info(f"{'='*60}")
    logger.info(f"Сервер: {server_url}")
    logger.info(f"Модель: {model_name}")
    logger.info(f"Режим: {mode}")
    logger.info(f"Параллельных запросов: {concurrent}")
    logger.info(f"Итераций: {iterations}")
    logger.info(f"Всего запросов: {results.total_requests}")
    logger.info(f"{'='*60}\n")
    
    results.start_time = time.perf_counter()
    
    request_id = 0
    for iteration in range(iterations):
        logger.info(f"Итерация {iteration + 1}/{iterations}...")
        
        tasks = []
        for _ in range(concurrent):
            request_id += 1
            task = request_func(
                server_url=server_url,
                audio_data=audio_data,
                sample_rate=sample_rate,
                chunk_size_ms=chunk_size_ms,
                request_id=request_id,
                model_name=model_name
            )
            tasks.append(task)
        
        iteration_results = await asyncio.gather(*tasks)
        
        for metrics in iteration_results:
            results.metrics.append(metrics)
            if metrics.success:
                results.successful_requests += 1
            else:
                results.failed_requests += 1
                logger.warning(f"  Запрос #{metrics.request_id} FAILED: {metrics.error}")
        
        success_count = sum(1 for m in iteration_results if m.success)
        logger.info(f"  Успешно: {success_count}/{concurrent}")
    
    results.end_time = time.perf_counter()
    return results


def print_results(results: BenchmarkResults):
    """Выводит результаты."""
    successful = results.get_successful_metrics()
    
    print(f"\n{'='*60}")
    print(f"📊 РЕЗУЛЬТАТЫ TRITON BENCHMARK")
    print(f"{'='*60}")
    
    print(f"\n📈 ОБЩАЯ СТАТИСТИКА:")
    print(f"  Всего запросов:     {results.total_requests}")
    print(f"  Успешных:           {results.successful_requests}")
    print(f"  Неудачных:          {results.failed_requests}")
    print(f"  Успешность:         {results.success_rate:.1f}%")
    print(f"  Общее время:        {results.total_duration:.2f}s")
    print(f"  Запросов/сек:       {results.requests_per_second:.2f}")
    
    if successful:
        connection_times = [m.connection_time for m in successful]
        stats = results.calculate_stats(connection_times)
        print(f"\n⏱️  ВРЕМЯ ПОДКЛЮЧЕНИЯ (сек):")
        print(f"  Min: {stats['min']:.3f} | Max: {stats['max']:.3f} | Avg: {stats['avg']:.3f} | P95: {stats['p95']:.3f}")
        
        ttft = [m.time_to_first_transcription for m in successful if m.time_to_first_transcription > 0]
        if ttft:
            stats = results.calculate_stats(ttft)
            print(f"\n⚡ ВРЕМЯ ДО ПЕРВОЙ ТРАНСКРИПЦИИ (сек):")
            print(f"  Min: {stats['min']:.3f} | Max: {stats['max']:.3f} | Avg: {stats['avg']:.3f} | P95: {stats['p95']:.3f}")
        
        total_times = [m.total_processing_time for m in successful]
        stats = results.calculate_stats(total_times)
        print(f"\n🕐 ОБЩЕЕ ВРЕМЯ ОБРАБОТКИ (сек):")
        print(f"  Min: {stats['min']:.3f} | Max: {stats['max']:.3f} | Avg: {stats['avg']:.3f} | P95: {stats['p95']:.3f}")
        
        rtf_values = [m.real_time_factor for m in successful]
        stats = results.calculate_stats(rtf_values)
        print(f"\n📉 REAL-TIME FACTOR (меньше 1 = быстрее реального времени):")
        print(f"  Min: {stats['min']:.3f} | Max: {stats['max']:.3f} | Avg: {stats['avg']:.3f} | P95: {stats['p95']:.3f}")
        
        transcription_counts = [m.transcriptions_received for m in successful]
        stats = results.calculate_stats(transcription_counts)
        print(f"\n📝 КОЛИЧЕСТВО ТРАНСКРИПЦИЙ НА ЗАПРОС:")
        print(f"  Min: {int(stats['min'])} | Max: {int(stats['max'])} | Avg: {stats['avg']:.1f}")
        
        if successful[0].final_transcription:
            print(f"\n💬 ПРИМЕР ТРАНСКРИПЦИИ:")
            text = successful[0].final_transcription[:100]
            print(f"  \"{text}{'...' if len(successful[0].final_transcription) > 100 else ''}\"")
    
    print(f"\n{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Triton ASR Benchmark")
    parser.add_argument("--server", default="localhost:8001", help="Triton gRPC URL")
    parser.add_argument("--model", default="streaming_asr", help="Имя модели")
    parser.add_argument("--audio", required=True, help="Путь к аудио файлу")
    parser.add_argument("--concurrent", type=int, default=5, help="Параллельных запросов")
    parser.add_argument("--iterations", type=int, default=3, help="Итераций")
    parser.add_argument("--chunk-size", type=int, default=100, help="Размер чанка в мс")
    parser.add_argument("--stream", action="store_true", help="Использовать StreamInfer (для Decoupled Mode)")
    
    args = parser.parse_args()
    
    if not Path(args.audio).exists():
        logger.error(f"Файл не найден: {args.audio}")
        return
    
    results = asyncio.run(run_benchmark(
        server_url=args.server,
        audio_file=args.audio,
        concurrent=args.concurrent,
        iterations=args.iterations,
        chunk_size_ms=args.chunk_size,
        model_name=args.model,
        use_stream=args.stream
    ))
    
    print_results(results)


if __name__ == "__main__":
    main()
