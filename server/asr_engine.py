"""
ASR Engine для онлайн стриминговой транскрипции с поддержкой множественных сессий.

Основан на NeMo streaming ASR и подходе pipecat-ai/nemotron.
Использует cache_state для инкрементальной обработки аудио.
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional, Callable

import numpy as np
import torch
from scipy import signal as scipy_signal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Целевой sample rate для модели
TARGET_SAMPLE_RATE = 16000


@dataclass
class SessionConfig:
    """Конфигурация для сессии стриминга."""
    session_id: str
    sample_rate: int = 16000
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    max_idle_time: float = 300.0  # 5 минут
    
    # Буфер для накопления новых чанков (только непрочитанные)
    pending_chunks: deque = field(default_factory=deque)
    pending_samples: int = 0
    
    # Минимальное количество samples для запуска транскрипции
    # NeMo рекомендует chunk_size_in_secs=0.08 (80мс) для FastConformer
    # Но для лучшего качества используем 0.5 сек (8000 samples)
    min_chunk_samples: int = 8000  # 0.5 секунды при 16kHz
    
    # Состояние кэша для стриминга (сохраняет контекст)
    cache_state: dict = field(default_factory=dict)
    
    # Полная транскрипция сессии
    full_transcription: str = ""


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Ресемплинг аудио."""
    if orig_sr == target_sr:
        return audio
    
    num_samples = int(len(audio) * target_sr / orig_sr)
    resampled = scipy_signal.resample(audio, num_samples)
    return resampled.astype(np.float32)


class StreamingASREngine:
    """
    ASR движок с правильной поддержкой стриминга через cache_state.
    """
    
    def __init__(
        self,
        model_path: str,
        device: Optional[torch.device] = None,
        compute_dtype: torch.dtype = torch.float32,
    ):
        from omegaconf import OmegaConf
        from nemo.collections.asr.parts.utils.transcribe_utils import setup_model
        
        logger.info(f"Загрузка модели из {model_path}")
        
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.asr_model, _ = setup_model(
            cfg=OmegaConf.create({"model_path": model_path}),
            map_location=device
        )
        
        self.asr_model = self.asr_model.to(device=device, dtype=compute_dtype)
        self.asr_model.eval()
        
        self.device = device
        self.compute_dtype = compute_dtype
        
        # Получаем sample rate модели
        self.model_sample_rate = TARGET_SAMPLE_RATE
        if hasattr(self.asr_model, 'preprocessor') and hasattr(self.asr_model.preprocessor, 'featurizer'):
            if hasattr(self.asr_model.preprocessor.featurizer, 'sample_rate'):
                self.model_sample_rate = self.asr_model.preprocessor.featurizer.sample_rate
        
        logger.info(f"Модель использует sample_rate: {self.model_sample_rate}")
        
        self.sessions: Dict[str, tuple] = {}
        self.lock = threading.Lock()
        
        logger.info(f"Модель загружена на {device}, готов к обработке сессий")
    
    def _get_initial_cache_state(self) -> dict:
        """Получает начальное состояние кэша для стриминга."""
        batch_size = 1
        cache_last_channel, cache_last_time, cache_last_channel_len = \
            self.asr_model.encoder.get_initial_cache_state(batch_size=batch_size)
        
        return {
            "cache_last_channel": self._move_to_device(cache_last_channel),
            "cache_last_time": self._move_to_device(cache_last_time),
            "cache_last_channel_len": cache_last_channel_len,
            "previous_hypotheses": None,
            "pred_out_stream": None
        }
    
    def _move_to_device(self, x):
        """Перемещает tensor или коллекцию на устройство."""
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        elif isinstance(x, (list, tuple)):
            return type(x)([self._move_to_device(item) for item in x])
        return x
    
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
                self._close_session_internal(session_id)
            
            config = SessionConfig(
                session_id=session_id,
                sample_rate=sample_rate,
            )
            config.cache_state = self._get_initial_cache_state()
            
            self.sessions[session_id] = (config, callback)
            logger.info(f"Создана сессия {session_id} с sample_rate={sample_rate}")
            
            return config
    
    def add_audio_chunk(
        self,
        session_id: str,
        audio_chunk: np.ndarray,
        sample_rate: int = 16000
    ) -> bool:
        """
        Добавляет аудио чанк в буфер сессии.
        Возвращает True если добавлено успешно.
        """
        with self.lock:
            if session_id not in self.sessions:
                return False
            
            config, _ = self.sessions[session_id]
            config.last_activity = time.time()
            
            # Ресемплинг если нужно
            if sample_rate != self.model_sample_rate:
                audio_chunk = resample_audio(audio_chunk, sample_rate, self.model_sample_rate)
            
            config.pending_chunks.append(audio_chunk)
            config.pending_samples += len(audio_chunk)
            
            return True
    
    def has_pending_audio(self, session_id: str) -> bool:
        """Проверяет есть ли достаточно данных для транскрипции."""
        with self.lock:
            if session_id not in self.sessions:
                return False
            config, _ = self.sessions[session_id]
            return config.pending_samples >= config.min_chunk_samples
    
    def transcribe_pending(self, session_id: str) -> Optional[str]:
        """
        Транскрибирует накопленные чанки используя cache_state.
        Возвращает ПОЛНУЮ транскрипцию сессии.
        """
        from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
        
        # Получаем и очищаем pending чанки под lock
        with self.lock:
            if session_id not in self.sessions:
                return None
            
            config, _ = self.sessions[session_id]
            
            if config.pending_samples < config.min_chunk_samples:
                return None
            
            # Забираем все pending чанки
            chunks = list(config.pending_chunks)
            config.pending_chunks.clear()
            config.pending_samples = 0
            
            # Копируем cache_state для обработки
            cache_state = config.cache_state
        
        if not chunks:
            return None
        
        # Объединяем чанки
        audio = np.concatenate(chunks)
        logger.debug(f"[{session_id}] Обработка {len(audio)} samples ({len(audio)/self.model_sample_rate:.2f}s)")
        
        # Конвертируем в tensor
        audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(
            device=self.device, dtype=self.compute_dtype
        )
        audio_lengths = torch.tensor([len(audio)], device=self.device)
        
        try:
            with torch.inference_mode():
                # Preprocessor
                processed_signal, processed_signal_length = self.asr_model.preprocessor(
                    input_signal=audio_tensor,
                    length=audio_lengths
                )
                
                # Streaming step с cache_state
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
                    cache_last_channel=cache_state["cache_last_channel"],
                    cache_last_time=cache_state["cache_last_time"],
                    cache_last_channel_len=cache_state["cache_last_channel_len"],
                    keep_all_outputs=True,
                    previous_hypotheses=cache_state["previous_hypotheses"],
                    previous_pred_out=cache_state["pred_out_stream"],
                    drop_extra_pre_encoded=0,
                    return_transcription=True,
                )
            
            # Обновляем cache_state
            with self.lock:
                if session_id in self.sessions:
                    config, _ = self.sessions[session_id]
                    config.cache_state["cache_last_channel"] = cache_last_channel
                    config.cache_state["cache_last_time"] = cache_last_time
                    config.cache_state["cache_last_channel_len"] = cache_last_channel_len
                    config.cache_state["previous_hypotheses"] = previous_hypotheses
                    config.cache_state["pred_out_stream"] = pred_out_stream
                    
                    # Получаем текст
                    if transcribed_texts:
                        if isinstance(transcribed_texts[0], Hypothesis):
                            text = transcribed_texts[0].text
                        else:
                            text = str(transcribed_texts[0]) if transcribed_texts else ""
                        
                        config.full_transcription = text
                        logger.info(f"[{session_id}] Транскрипция: \"{text}\"")
                        return text
            
            return None
            
        except Exception as e:
            logger.error(f"[{session_id}] Ошибка транскрипции: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def get_transcription(self, session_id: str) -> str:
        """Возвращает текущую полную транскрипцию сессии."""
        with self.lock:
            if session_id not in self.sessions:
                return ""
            config, _ = self.sessions[session_id]
            return config.full_transcription
    
    def soft_reset(self, session_id: str):
        """
        Soft reset - сбрасывает hypotheses но сохраняет encoder cache.
        Используется после паузы в речи для начала нового предложения.
        """
        with self.lock:
            if session_id not in self.sessions:
                return
            config, _ = self.sessions[session_id]
            config.cache_state["previous_hypotheses"] = None
            config.cache_state["pred_out_stream"] = None
            config.full_transcription = ""
            logger.info(f"[{session_id}] Soft reset выполнен")
    
    def hard_reset(self, session_id: str):
        """
        Hard reset - полностью сбрасывает состояние.
        Используется для начала новой сессии без создания новой.
        """
        with self.lock:
            if session_id not in self.sessions:
                return
            config, _ = self.sessions[session_id]
            config.cache_state = self._get_initial_cache_state()
            config.pending_chunks.clear()
            config.pending_samples = 0
            config.full_transcription = ""
            logger.info(f"[{session_id}] Hard reset выполнен")
    
    def _close_session_internal(self, session_id: str) -> str:
        """Внутренний метод закрытия сессии (под lock)."""
        if session_id not in self.sessions:
            return ""
        config, _ = self.sessions[session_id]
        transcription = config.full_transcription
        del self.sessions[session_id]
        return transcription
    
    def transcribe_all_pending(self, session_id: str) -> Optional[str]:
        """
        Транскрибирует ВСЕ накопленные данные без проверки минимума.
        Используется при закрытии сессии.
        """
        from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
        
        with self.lock:
            if session_id not in self.sessions:
                return None
            
            config, _ = self.sessions[session_id]
            
            if config.pending_samples == 0:
                return config.full_transcription
            
            # Забираем все pending чанки
            chunks = list(config.pending_chunks)
            config.pending_chunks.clear()
            config.pending_samples = 0
            cache_state = config.cache_state
        
        if not chunks:
            return None
        
        # Объединяем чанки
        audio = np.concatenate(chunks)
        logger.debug(f"[{session_id}] Финальная обработка {len(audio)} samples ({len(audio)/self.model_sample_rate:.2f}s)")
        
        # Конвертируем в tensor
        audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(
            device=self.device, dtype=self.compute_dtype
        )
        audio_lengths = torch.tensor([len(audio)], device=self.device)
        
        try:
            with torch.inference_mode():
                processed_signal, processed_signal_length = self.asr_model.preprocessor(
                    input_signal=audio_tensor,
                    length=audio_lengths
                )
                
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
                    cache_last_channel=cache_state["cache_last_channel"],
                    cache_last_time=cache_state["cache_last_time"],
                    cache_last_channel_len=cache_state["cache_last_channel_len"],
                    keep_all_outputs=True,
                    previous_hypotheses=cache_state["previous_hypotheses"],
                    previous_pred_out=cache_state["pred_out_stream"],
                    drop_extra_pre_encoded=0,
                    return_transcription=True,
                )
            
            with self.lock:
                if session_id in self.sessions:
                    config, _ = self.sessions[session_id]
                    if transcribed_texts:
                        if isinstance(transcribed_texts[0], Hypothesis):
                            text = transcribed_texts[0].text
                        else:
                            text = str(transcribed_texts[0]) if transcribed_texts else ""
                        config.full_transcription = text
                        return text
            
            return None
            
        except Exception as e:
            logger.error(f"[{session_id}] Ошибка финальной транскрипции: {e}")
            return None
    
    def close_session(self, session_id: str) -> str:
        """Закрывает сессию и возвращает финальную транскрипцию."""
        # Сначала обрабатываем ВСЕ оставшиеся данные (без проверки минимума)
        final = self.transcribe_all_pending(session_id) or ""
        
        with self.lock:
            if session_id not in self.sessions:
                return final
            
            config, _ = self.sessions[session_id]
            duration = time.time() - config.created_at
            
            # Получаем финальную транскрипцию
            if not final:
                final = config.full_transcription
            
            logger.info(f"[{session_id}] Закрытие сессии (длительность: {duration:.2f}s)")
            self._close_session_internal(session_id)
        
        logger.info(f"[{session_id}] Финальная транскрипция: \"{final}\"")
        return final
    
    def get_session_count(self) -> int:
        """Возвращает количество активных сессий."""
        with self.lock:
            return len(self.sessions)
    
    # Backward compatibility methods
    def process_audio_chunk(
        self,
        session_id: str,
        audio_chunk: np.ndarray,
        sample_rate: int = 16000,
        return_transcription: bool = False
    ) -> Optional[str]:
        """Совместимость со старым API."""
        self.add_audio_chunk(session_id, audio_chunk, sample_rate)
        return None  # Транскрипция теперь через transcribe_pending
    
    def transcribe_buffer(self, session_id: str) -> str:
        """Совместимость со старым API."""
        return self.transcribe_pending(session_id) or ""
