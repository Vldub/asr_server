"""
ASR Engine для онлайн стриминговой транскрипции с поддержкой множественных сессий.

Основан на NeMo CacheAwareStreamingAudioBuffer и официальном примере:
https://github.com/NVIDIA-NeMo/NeMo/blob/main/examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py
"""

import asyncio
import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from queue import Queue
from typing import Dict, Optional, Callable

import numpy as np
import torch
from omegaconf import OmegaConf

from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
from nemo.collections.asr.parts.utils.transcribe_utils import setup_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SessionConfig:
    """Конфигурация для сессии стриминга."""
    session_id: str
    sample_rate: int = 16000
    online_normalization: bool = False
    pad_and_drop_preencoded: bool = False
    chunk_size: int = -1
    shift_size: int = -1
    left_chunks: int = 2
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    max_idle_time: float = 300.0  # 5 минут
    min_buffer_duration: float = 1.0  # Минимальная длительность буфера для транскрипции (секунды)
    accumulated_samples: int = 0  # Накопленные samples
    audio_buffer: deque = field(default_factory=deque)  # Буфер для накопления аудио
    cache_state: dict = field(default_factory=dict)  # Состояние кэша для стриминга


class StreamingASREngine:
    """
    ASR движок для онлайн стриминговой транскрипции с поддержкой множественных сессий.
    """
    
    def __init__(
        self,
        model_path: str,
        device: Optional[torch.device] = None,
        compute_dtype: torch.dtype = torch.float32,
        online_normalization: bool = False,
        pad_and_drop_preencoded: bool = False,
    ):
        """
        Инициализация ASR движка.
        
        Args:
            model_path: Путь к .nemo модели
            device: Устройство для вычислений
            compute_dtype: Тип данных для вычислений
            online_normalization: Включить нормализацию на лету
            pad_and_drop_preencoded: Параметр кэширования
        """
        logger.info(f"Загрузка модели из {model_path}")
        
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else:
                device = torch.device("cpu")
        
        self.asr_model, _ = setup_model(
            cfg=OmegaConf.create({"model_path": model_path}),
            map_location=device
        )
        
        self.asr_model = self.asr_model.to(device=device, dtype=compute_dtype)
        self.asr_model.eval()
        
        self.device = device
        self.compute_dtype = compute_dtype
        self.online_normalization = online_normalization
        self.pad_and_drop_preencoded = pad_and_drop_preencoded
        
        self.sessions: Dict[str, tuple] = {}
        self.lock = threading.Lock()
        
        logger.info(f"Модель загружена на {device}, готов к обработке сессий")
    
    def create_session(
        self,
        session_id: str,
        sample_rate: int = 16000,
        callback: Optional[Callable] = None
    ) -> SessionConfig:
        """Создает новую сессию для стриминга."""
        with self.lock:
            if session_id in self.sessions:
                logger.warning(f"Сессия {session_id} уже существует, пересоздаю")
                self.close_session(session_id)
            
            streaming_buffer = CacheAwareStreamingAudioBuffer(
                model=self.asr_model,
                online_normalization=self.online_normalization,
                pad_and_drop_preencoded=self.pad_and_drop_preencoded,
            )
            
            config = SessionConfig(
                session_id=session_id,
                sample_rate=sample_rate,
                min_buffer_duration=1.0  # Минимум 1 секунда для транскрипции
            )
            
            # Инициализируем состояние кэша для стриминга
            batch_size = 1
            cache_last_channel, cache_last_time, cache_last_channel_len = self.asr_model.encoder.get_initial_cache_state(batch_size=batch_size)
            
            def move_to_device(x, dev):
                if isinstance(x, torch.Tensor):
                    return x.to(dev)
                elif isinstance(x, (list, tuple)):
                    return type(x)([move_to_device(item, dev) for item in x])
                return x
            
            config.cache_state = {
                "cache_last_channel": move_to_device(cache_last_channel, self.device),
                "cache_last_time": move_to_device(cache_last_time, self.device),
                "cache_last_channel_len": cache_last_channel_len,
                "previous_hypotheses": None,
                "pred_out_stream": None
            }
            
            self.sessions[session_id] = (streaming_buffer, config, callback)
            logger.info(f"Создана сессия {session_id} с sample_rate={sample_rate}")
            
            return config
    
    def process_audio_chunk(
        self,
        session_id: str,
        audio_chunk: np.ndarray,
        sample_rate: int = 16000,
        return_transcription: bool = False
    ) -> Optional[str]:
        """Обрабатывает чанк аудио для указанной сессии."""
        with self.lock:
            if session_id not in self.sessions:
                raise ValueError(f"Сессия {session_id} не найдена")
            
            streaming_buffer, config, callback = self.sessions[session_id]
            config.last_activity = time.time()
            
            if config.sample_rate != sample_rate:
                logger.warning(f"Несоответствие sample_rate: сессия={config.sample_rate}, чанк={sample_rate}")
        
        try:
            chunk_duration = len(audio_chunk) / sample_rate
            logger.info(f"[{session_id}] Добавление чанка в буфер: {len(audio_chunk)} samples ({chunk_duration:.3f}s)")
            
            # Добавляем в наш собственный буфер для накопления
            with self.lock:
                config.audio_buffer.append(audio_chunk)
                config.accumulated_samples += len(audio_chunk)
                accumulated_duration = config.accumulated_samples / sample_rate
            
            # Также добавляем в streaming_buffer для совместимости
            streaming_buffer.append_audio(
                audio=audio_chunk,
                stream_id=-1
            )
            
            # Проверяем состояние буфера (streams_length может быть Tensor)
            if hasattr(streaming_buffer, 'streams_length') and streaming_buffer.streams_length is not None:
                try:
                    streams_length = streaming_buffer.streams_length
                    if isinstance(streams_length, torch.Tensor):
                        total_samples = streams_length.sum().item()
                    else:
                        total_samples = sum(streams_length) if streams_length else 0
                    total_duration = total_samples / sample_rate
                    logger.info(f"[{session_id}] Буфер NeMo: {total_samples} samples ({total_duration:.3f}s), накоплено: {config.accumulated_samples} samples ({accumulated_duration:.3f}s)")
                except Exception as e:
                    logger.debug(f"[{session_id}] Не удалось получить размер буфера: {e}")
            
            # Если требуется транскрипция и накопилось достаточно данных, обрабатываем буфер
            if return_transcription and accumulated_duration >= config.min_buffer_duration:
                logger.info(f"[{session_id}] Накоплено достаточно данных ({accumulated_duration:.3f}s >= {config.min_buffer_duration}s), запускаем транскрипцию")
                transcription = self._process_accumulated_audio(session_id, config)
                # Сбрасываем счетчик и буфер после обработки
                with self.lock:
                    config.accumulated_samples = 0
                    config.audio_buffer.clear()
                return transcription
            elif return_transcription:
                logger.debug(f"[{session_id}] Недостаточно данных для транскрипции: {accumulated_duration:.3f}s < {config.min_buffer_duration}s")
                return None
            
            return None
            
        except Exception as e:
            logger.error(f"Ошибка обработки чанка для сессии {session_id}: {e}")
            raise
    
    def _process_accumulated_audio(self, session_id: str, config: SessionConfig) -> str:
        """Обрабатывает накопленное аудио напрямую через модель."""
        from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
        
        # Получаем копию буфера для обработки (чтобы избежать проблем с блокировкой)
        with self.lock:
            if session_id not in self.sessions:
                logger.warning(f"[{session_id}] Сессия не найдена при обработке")
                return ""
            audio_buffer_copy = list(config.audio_buffer)
            accumulated_samples = config.accumulated_samples
        
        logger.info(f"[{session_id}] Обработка накопленного аудио: {len(audio_buffer_copy)} чанков, {accumulated_samples} samples")
        
        if not audio_buffer_copy:
            logger.warning(f"[{session_id}] Буфер аудио пуст")
            return ""
        
        # Объединяем все чанки в один массив
        accumulated_audio = np.concatenate(audio_buffer_copy)
        logger.info(f"[{session_id}] Объединенное аудио: {len(accumulated_audio)} samples ({len(accumulated_audio)/config.sample_rate:.3f}s)")
        
        # Конвертируем в torch tensor и обрабатываем через preprocessor модели
        # Формат: (batch, time) -> preprocessor -> (batch, features, time)
        audio_tensor = torch.from_numpy(accumulated_audio).unsqueeze(0).to(device=self.device, dtype=self.compute_dtype)
        audio_lengths = torch.tensor([len(accumulated_audio)], device=self.device)
        
        # Обрабатываем через preprocessor модели для получения features
        with torch.inference_mode():
            processed_signal, processed_signal_length = self.asr_model.preprocessor(
                input_signal=audio_tensor,
                length=audio_lengths
            )
        
        logger.info(f"[{session_id}] Обработанный сигнал: shape={processed_signal.shape}, length={processed_signal_length}")
        
        # Получаем состояние кэша
        cache_state = config.cache_state
        cache_last_channel = cache_state["cache_last_channel"]
        cache_last_time = cache_state["cache_last_time"]
        cache_last_channel_len = cache_state["cache_last_channel_len"]
        previous_hypotheses = cache_state["previous_hypotheses"]
        pred_out_stream = cache_state["pred_out_stream"]
        
        try:
            with torch.inference_mode():
                with torch.no_grad():
                    # Используем streaming step для обработки
                    (
                        pred_out_stream,
                        transcribed_texts,
                        cache_last_channel,
                        cache_last_time,
                        cache_last_channel_len,
                        previous_hypotheses,
                    ) = self.asr_model.conformer_stream_step(
                        processed_signal=processed_signal,
                        processed_signal_length=processed_signal_length,
                        cache_last_channel=cache_last_channel,
                        cache_last_time=cache_last_time,
                        cache_last_channel_len=cache_last_channel_len,
                        keep_all_outputs=True,  # Получаем все выходы
                        previous_hypotheses=previous_hypotheses,
                        previous_pred_out=pred_out_stream,
                        drop_extra_pre_encoded=0,
                        return_transcription=True,
                    )
            
            # Обновляем состояние кэша (без блокировки, так как это может вызывать deadlock)
            # Получаем доступ к конфигу через lock
            with self.lock:
                if session_id in self.sessions:
                    _, config, _ = self.sessions[session_id]
                    config.cache_state["cache_last_channel"] = cache_last_channel
                    config.cache_state["cache_last_time"] = cache_last_time
                    config.cache_state["cache_last_channel_len"] = cache_last_channel_len
                    config.cache_state["previous_hypotheses"] = previous_hypotheses
                    config.cache_state["pred_out_stream"] = pred_out_stream
            
            if transcribed_texts:
                if isinstance(transcribed_texts[0], Hypothesis):
                    result = transcribed_texts[0].text
                else:
                    result = str(transcribed_texts[0]) if transcribed_texts else ""
                logger.info(f"[{session_id}] Транскрипция получена: \"{result}\"")
                return result
            else:
                logger.warning(f"[{session_id}] Транскрипция не получена, transcribed_texts={transcribed_texts}")
                return ""
                
        except Exception as e:
            logger.error(f"[{session_id}] Ошибка обработки накопленного аудио: {e}")
            import traceback
            logger.error(f"[{session_id}] Traceback: {traceback.format_exc()}")
            return ""
    
    def _process_streaming_buffer(self, session_id: str, streaming_buffer: CacheAwareStreamingAudioBuffer) -> str:
        """Внутренний метод для обработки накопленных данных в буфере."""
        from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
        
        logger.info(f"[{session_id}] Начало обработки буфера для транскрипции")
        
        def extract_transcriptions(hyps):
            if not hyps:
                return [""]
            if isinstance(hyps[0], Hypothesis):
                return [hyp.text for hyp in hyps]
            return [str(hyp) for hyp in hyps]
        
        def calc_drop_extra_pre_encoded(asr_model, step_num, pad_and_drop_preencoded):
            if step_num == 0 and not pad_and_drop_preencoded:
                return 0
            elif hasattr(asr_model.encoder, "streaming_cfg"):
                return asr_model.encoder.streaming_cfg.drop_extra_pre_encoded
            return 0
        
        def move_to_device(x, dev):
            if isinstance(x, torch.Tensor):
                return x.to(dev)
            elif isinstance(x, (list, tuple)):
                return type(x)([move_to_device(item, dev) for item in x])
            return x
        
        with self.lock:
            if session_id not in self.sessions:
                logger.warning(f"[{session_id}] Сессия не найдена")
                return ""
            streaming_buffer, config, callback = self.sessions[session_id]
        
        # Получаем состояние кэша из конфигурации
        cache_state = config.cache_state
        cache_last_channel = move_to_device(cache_state["cache_last_channel"], self.device)
        cache_last_time = move_to_device(cache_state["cache_last_time"], self.device)
        cache_last_channel_len = cache_state["cache_last_channel_len"]
        previous_hypotheses = cache_state["previous_hypotheses"]
        pred_out_stream = cache_state["pred_out_stream"]
        
        transcribed_texts = None
        
        try:
            # Проверяем, не пуст ли буфер (is_buffer_empty может вернуть Tensor)
            buffer_empty = streaming_buffer.is_buffer_empty()
            if isinstance(buffer_empty, torch.Tensor):
                buffer_empty = buffer_empty.item() if buffer_empty.numel() == 1 else bool(buffer_empty.any())
            
            if buffer_empty:
                logger.warning(f"[{session_id}] Буфер пуст, транскрипция невозможна")
                return ""
            
            # Проверяем размер буфера перед обработкой
            if hasattr(streaming_buffer, 'streams_length') and streaming_buffer.streams_length is not None:
                try:
                    streams_length = streaming_buffer.streams_length
                    if isinstance(streams_length, torch.Tensor):
                        total_samples = streams_length.sum().item()
                    else:
                        total_samples = sum(streams_length) if streams_length else 0
                    logger.info(f"[{session_id}] Размер буфера перед обработкой: {total_samples} samples")
                except Exception as e:
                    logger.debug(f"[{session_id}] Не удалось получить размер буфера: {e}")
            
            # Пробуем получить данные из буфера через итератор
            step_count = 0
            try:
                streaming_buffer_iter = iter(streaming_buffer)
                logger.info(f"[{session_id}] Итератор буфера создан, начинаем обработку чанков")
                
                for step_num, (chunk_audio, chunk_lengths) in enumerate(streaming_buffer_iter):
                    step_count += 1
                    chunk_size = chunk_audio.shape[-1] if len(chunk_audio.shape) > 0 else 0
                    logger.info(f"[{session_id}] Обработка шага {step_num + 1}: {chunk_size} samples, lengths={chunk_lengths}")
                    
                    with torch.inference_mode():
                        chunk_audio = chunk_audio.to(device=self.device, dtype=self.compute_dtype)
                        chunk_lengths = chunk_lengths.to(device=self.device)
                        
                        with torch.no_grad():
                            (
                                pred_out_stream,
                                transcribed_texts,
                                cache_last_channel,
                                cache_last_time,
                                cache_last_channel_len,
                                previous_hypotheses,
                            ) = self.asr_model.conformer_stream_step(
                                processed_signal=chunk_audio,
                                processed_signal_length=chunk_lengths,
                                cache_last_channel=cache_last_channel,
                                cache_last_time=cache_last_time,
                                cache_last_channel_len=cache_last_channel_len,
                                keep_all_outputs=streaming_buffer.is_buffer_empty(),
                                previous_hypotheses=previous_hypotheses,
                                previous_pred_out=pred_out_stream,
                                drop_extra_pre_encoded=calc_drop_extra_pre_encoded(
                                    self.asr_model, step_num, self.pad_and_drop_preencoded
                                ),
                                return_transcription=True,
                            )
                    
                    # Обновляем состояние кэша
                    with self.lock:
                        if session_id in self.sessions:
                            config.cache_state["cache_last_channel"] = cache_last_channel
                            config.cache_state["cache_last_time"] = cache_last_time
                            config.cache_state["cache_last_channel_len"] = cache_last_channel_len
                            config.cache_state["previous_hypotheses"] = previous_hypotheses
                            config.cache_state["pred_out_stream"] = pred_out_stream
                
            except StopIteration:
                logger.info(f"[{session_id}] Итератор завершен, обработано {step_count} шагов")
            except Exception as iter_error:
                logger.error(f"[{session_id}] Ошибка при итерации буфера: {iter_error}")
                import traceback
                logger.error(f"[{session_id}] Traceback: {traceback.format_exc()}")
                # Пробуем получить транскрипцию из последнего состояния
                if previous_hypotheses:
                    transcribed_texts = previous_hypotheses
        
        except Exception as e:
            logger.error(f"[{session_id}] Общая ошибка при обработке буфера: {e}")
            import traceback
            logger.error(f"[{session_id}] Traceback: {traceback.format_exc()}")
            return ""
        
        if transcribed_texts:
            final_transcriptions = extract_transcriptions(transcribed_texts)
            result = final_transcriptions[0] if final_transcriptions else ""
            logger.info(f"[{session_id}] Транскрипция завершена ({step_count} шагов): \"{result}\"")
            if not result:
                logger.warning(f"[{session_id}] Транскрипция пустая, transcribed_texts={transcribed_texts}")
            return result
        
        logger.warning(f"[{session_id}] Транскрипция не получена после {step_count} шагов, transcribed_texts={transcribed_texts}")
        return ""
    
    def close_session(self, session_id: str) -> Optional[str]:
        """Закрывает сессию и возвращает финальную транскрипцию."""
        with self.lock:
            if session_id not in self.sessions:
                logger.warning(f"Сессия {session_id} не найдена")
                return None
            
            streaming_buffer, config, callback = self.sessions[session_id]
            session_duration = time.time() - config.created_at
            accumulated_duration = config.accumulated_samples / config.sample_rate
            logger.info(f"[{session_id}] Закрытие сессии (длительность: {session_duration:.2f}s, накоплено: {accumulated_duration:.3f}s)")
            
            # Сохраняем данные для обработки вне блокировки
            has_audio_buffer = len(config.audio_buffer) > 0
            audio_buffer_copy = list(config.audio_buffer) if has_audio_buffer else []
        
        # Обрабатываем данные вне блокировки, чтобы избежать deadlock
        try:
            if has_audio_buffer:
                # Временно восстанавливаем буфер для обработки
                with self.lock:
                    if session_id in self.sessions:
                        config.audio_buffer = deque(audio_buffer_copy)
                final_transcription = self._process_accumulated_audio(session_id, config)
            else:
                final_transcription = self._process_streaming_buffer(session_id, streaming_buffer)
        except Exception as e:
            logger.error(f"[{session_id}] Ошибка при обработке финальных данных: {e}")
            import traceback
            logger.error(f"[{session_id}] Traceback: {traceback.format_exc()}")
            final_transcription = ""
        
        # Очищаем сессию
        with self.lock:
            if session_id in self.sessions:
                streaming_buffer.reset_buffer()
                config.accumulated_samples = 0
                config.audio_buffer.clear()
                del self.sessions[session_id]
        
        logger.info(f"[{session_id}] Сессия закрыта, финальная транскрипция: \"{final_transcription}\"")
        return final_transcription
    
    def get_session_count(self) -> int:
        """Возвращает количество активных сессий."""
        with self.lock:
            return len(self.sessions)



