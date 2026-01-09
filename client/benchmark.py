#!/usr/bin/env python3
"""
Скрипт для нагрузочного тестирования Streaming ASR сервера.

Использование:
    uv run benchmark.py --server ws://localhost:8765/ws/transcribe --audio test.wav --concurrent 5 --iterations 10
"""

import argparse
import asyncio
import json
import logging
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf
import websockets
from websockets.exceptions import ConnectionClosed

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class RequestMetrics:
    """Метрики одного запроса."""
    request_id: int
    success: bool = False
    error: Optional[str] = None
    
    # Временные метрики (в секундах)
    connection_time: float = 0.0
    session_create_time: float = 0.0
    time_to_first_transcription: float = 0.0
    total_processing_time: float = 0.0
    
    # Результаты
    audio_duration: float = 0.0
    chunks_sent: int = 0
    transcriptions_received: int = 0
    final_transcription: str = ""
    
    @property
    def real_time_factor(self) -> float:
        """RTF = время обработки / длительность аудио. Меньше 1 = быстрее реального времени."""
        if self.audio_duration > 0:
            return self.total_processing_time / self.audio_duration
        return 0.0


@dataclass 
class BenchmarkResults:
    """Результаты нагрузочного тестирования."""
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
            return {"min": 0, "max": 0, "avg": 0, "median": 0, "p95": 0, "p99": 0}
        
        sorted_values = sorted(values)
        return {
            "min": min(values),
            "max": max(values),
            "avg": statistics.mean(values),
            "median": statistics.median(values),
            "p95": sorted_values[int(len(sorted_values) * 0.95)] if len(sorted_values) > 1 else sorted_values[0],
            "p99": sorted_values[int(len(sorted_values) * 0.99)] if len(sorted_values) > 1 else sorted_values[0],
        }


async def run_single_request(
    server_url: str,
    audio_data: np.ndarray,
    sample_rate: int,
    chunk_size_ms: int,
    request_id: int
) -> RequestMetrics:
    """Выполняет один запрос к серверу."""
    metrics = RequestMetrics(request_id=request_id)
    metrics.audio_duration = len(audio_data) / sample_rate
    
    chunk_size_samples = int((chunk_size_ms / 1000.0) * sample_rate)
    
    start_time = time.perf_counter()
    
    try:
        # Подключение
        connect_start = time.perf_counter()
        async with websockets.connect(server_url, close_timeout=10) as websocket:
            metrics.connection_time = time.perf_counter() - connect_start
            
            session_id = f"bench_{request_id}_{int(time.time() * 1000)}"
            
            # Создание сессии
            session_start = time.perf_counter()
            await websocket.send(json.dumps({
                "action": "start_session",
                "session_id": session_id,
                "sample_rate": sample_rate
            }))
            
            response = await asyncio.wait_for(websocket.recv(), timeout=10)
            data = json.loads(response)
            metrics.session_create_time = time.perf_counter() - session_start
            
            if data.get("status") != "session_created" and data.get("type") != "status":
                metrics.error = f"Ошибка создания сессии: {data}"
                return metrics
            
            first_transcription_time = None
            transcriptions = []
            
            # Задача для получения транскрипций
            async def receive_transcriptions():
                nonlocal first_transcription_time, transcriptions
                try:
                    async for message in websocket:
                        try:
                            data = json.loads(message)
                            if data.get("type") == "transcription":
                                if first_transcription_time is None:
                                    first_transcription_time = time.perf_counter()
                                transcriptions.append(data.get("text", ""))
                            elif data.get("type") == "status" and data.get("status") == "session_closed":
                                final = data.get("final_transcription", "")
                                if final:
                                    transcriptions.append(final)
                                break
                            elif data.get("type") == "error":
                                metrics.error = data.get("error")
                        except json.JSONDecodeError:
                            pass
                except ConnectionClosed:
                    pass
            
            receive_task = asyncio.create_task(receive_transcriptions())
            
            # Отправка аудио чанков
            chunks_sent = 0
            for i in range(0, len(audio_data), chunk_size_samples):
                chunk = audio_data[i:i + chunk_size_samples]
                await websocket.send(chunk.astype(np.float32).tobytes())
                chunks_sent += 1
                await asyncio.sleep(0.005)  # Небольшая задержка
            
            metrics.chunks_sent = chunks_sent
            
            # Завершение сессии
            await websocket.send(json.dumps({"action": "end_session"}))
            
            # Ждём получения всех транскрипций
            try:
                await asyncio.wait_for(receive_task, timeout=15)
            except asyncio.TimeoutError:
                receive_task.cancel()
            
            metrics.transcriptions_received = len(transcriptions)
            metrics.final_transcription = transcriptions[-1] if transcriptions else ""
            
            if first_transcription_time:
                metrics.time_to_first_transcription = first_transcription_time - start_time
            
            metrics.success = True
            
    except asyncio.TimeoutError:
        metrics.error = "Timeout"
    except ConnectionClosed as e:
        metrics.error = f"Connection closed: {e}"
    except Exception as e:
        metrics.error = str(e)
    
    metrics.total_processing_time = time.perf_counter() - start_time
    return metrics


async def run_benchmark(
    server_url: str,
    audio_file: str,
    concurrent: int,
    iterations: int,
    chunk_size_ms: int
) -> BenchmarkResults:
    """Запускает нагрузочное тестирование."""
    
    # Загрузка аудио
    logger.info(f"Загрузка аудио файла: {audio_file}")
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
    
    logger.info(f"\n{'='*60}")
    logger.info(f"🚀 НАГРУЗОЧНОЕ ТЕСТИРОВАНИЕ")
    logger.info(f"{'='*60}")
    logger.info(f"Сервер: {server_url}")
    logger.info(f"Параллельных запросов: {concurrent}")
    logger.info(f"Итераций: {iterations}")
    logger.info(f"Всего запросов: {results.total_requests}")
    logger.info(f"{'='*60}\n")
    
    results.start_time = time.perf_counter()
    
    request_id = 0
    for iteration in range(iterations):
        logger.info(f"Итерация {iteration + 1}/{iterations}...")
        
        # Создаём concurrent задач
        tasks = []
        for _ in range(concurrent):
            request_id += 1
            task = run_single_request(
                server_url=server_url,
                audio_data=audio_data,
                sample_rate=sample_rate,
                chunk_size_ms=chunk_size_ms,
                request_id=request_id
            )
            tasks.append(task)
        
        # Выполняем параллельно
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
    """Выводит результаты тестирования."""
    successful = results.get_successful_metrics()
    
    print(f"\n{'='*60}")
    print(f"📊 РЕЗУЛЬТАТЫ НАГРУЗОЧНОГО ТЕСТИРОВАНИЯ")
    print(f"{'='*60}")
    
    print(f"\n📈 ОБЩАЯ СТАТИСТИКА:")
    print(f"  Всего запросов:     {results.total_requests}")
    print(f"  Успешных:           {results.successful_requests}")
    print(f"  Неудачных:          {results.failed_requests}")
    print(f"  Успешность:         {results.success_rate:.1f}%")
    print(f"  Общее время:        {results.total_duration:.2f}s")
    print(f"  Запросов/сек:       {results.requests_per_second:.2f}")
    
    if successful:
        # Время подключения
        connection_times = [m.connection_time for m in successful]
        stats = results.calculate_stats(connection_times)
        print(f"\n⏱️  ВРЕМЯ ПОДКЛЮЧЕНИЯ (сек):")
        print(f"  Min: {stats['min']:.3f} | Max: {stats['max']:.3f} | Avg: {stats['avg']:.3f} | P95: {stats['p95']:.3f}")
        
        # Время до первой транскрипции
        ttft = [m.time_to_first_transcription for m in successful if m.time_to_first_transcription > 0]
        if ttft:
            stats = results.calculate_stats(ttft)
            print(f"\n⚡ ВРЕМЯ ДО ПЕРВОЙ ТРАНСКРИПЦИИ (сек):")
            print(f"  Min: {stats['min']:.3f} | Max: {stats['max']:.3f} | Avg: {stats['avg']:.3f} | P95: {stats['p95']:.3f}")
        
        # Общее время обработки
        total_times = [m.total_processing_time for m in successful]
        stats = results.calculate_stats(total_times)
        print(f"\n🕐 ОБЩЕЕ ВРЕМЯ ОБРАБОТКИ (сек):")
        print(f"  Min: {stats['min']:.3f} | Max: {stats['max']:.3f} | Avg: {stats['avg']:.3f} | P95: {stats['p95']:.3f}")
        
        # Real-Time Factor
        rtf_values = [m.real_time_factor for m in successful]
        stats = results.calculate_stats(rtf_values)
        print(f"\n📉 REAL-TIME FACTOR (меньше 1 = быстрее реального времени):")
        print(f"  Min: {stats['min']:.3f} | Max: {stats['max']:.3f} | Avg: {stats['avg']:.3f} | P95: {stats['p95']:.3f}")
        
        # Транскрипции
        transcription_counts = [m.transcriptions_received for m in successful]
        stats = results.calculate_stats(transcription_counts)
        print(f"\n📝 КОЛИЧЕСТВО ТРАНСКРИПЦИЙ НА ЗАПРОС:")
        print(f"  Min: {int(stats['min'])} | Max: {int(stats['max'])} | Avg: {stats['avg']:.1f}")
        
        # Пример транскрипции
        if successful[0].final_transcription:
            print(f"\n💬 ПРИМЕР ТРАНСКРИПЦИИ:")
            print(f"  \"{successful[0].final_transcription[:100]}{'...' if len(successful[0].final_transcription) > 100 else ''}\"")
    
    print(f"\n{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Нагрузочное тестирование ASR сервера")
    parser.add_argument("--server", required=True, help="WebSocket URL сервера")
    parser.add_argument("--audio", required=True, help="Путь к аудио файлу")
    parser.add_argument("--concurrent", type=int, default=5, help="Количество параллельных запросов")
    parser.add_argument("--iterations", type=int, default=10, help="Количество итераций")
    parser.add_argument("--chunk-size", type=int, default=100, help="Размер чанка в мс")
    
    args = parser.parse_args()
    
    if not args.server.startswith("ws://") and not args.server.startswith("wss://"):
        args.server = f"ws://{args.server}"
    
    if not args.server.endswith("/ws/transcribe"):
        args.server = f"{args.server}/ws/transcribe"
    
    if not Path(args.audio).exists():
        logger.error(f"Файл не найден: {args.audio}")
        return
    
    results = asyncio.run(run_benchmark(
        server_url=args.server,
        audio_file=args.audio,
        concurrent=args.concurrent,
        iterations=args.iterations,
        chunk_size_ms=args.chunk_size
    ))
    
    print_results(results)


if __name__ == "__main__":
    main()

