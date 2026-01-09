"""
Triton Python Backend для Streaming ASR с NeMo.

Этот backend обрабатывает streaming аудио с сохранением состояния кэша
между запросами одной последовательности (сессии).
"""

import json
import logging
import numpy as np
import torch
from typing import Dict, Optional, List

import triton_python_backend_utils as pb_utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TritonPythonModel:
    """Triton Python Backend для Streaming ASR."""
    
    def initialize(self, args: Dict[str, str]):
        """Инициализация модели при загрузке."""
        self.model_config = json.loads(args['model_config'])
        self.model_instance_device_id = int(args.get('model_instance_device_id', 0))
        
        # Определяем устройство
        if torch.cuda.is_available() and self.model_instance_device_id >= 0:
            self.device = torch.device(f'cuda:{self.model_instance_device_id}')
        else:
            self.device = torch.device('cpu')
        
        logger.info(f"Инициализация Streaming ASR на устройстве: {self.device}")
        
        # Загружаем NeMo модель
        model_path = self._get_model_path()
        self._load_model(model_path)
        
        # Кэш состояний для активных последовательностей
        self.sequence_states: Dict[int, dict] = {}
        
        # Параметры модели
        self.sample_rate = 16000
        self.min_chunk_samples = 8000  # 0.5 сек
        
        logger.info("Streaming ASR инициализирован")
    
    def _get_model_path(self) -> str:
        """Получает путь к модели из конфигурации."""
        # Модель должна быть в /models/streaming_asr/model.nemo
        # или указана в параметрах
        parameters = self.model_config.get('parameters', {})
        model_path = parameters.get('model_path', {}).get('string_value', '/models/model.nemo')
        return model_path
    
    def _load_model(self, model_path: str):
        """Загружает NeMo модель."""
        from omegaconf import OmegaConf
        from nemo.collections.asr.parts.utils.transcribe_utils import setup_model
        
        logger.info(f"Загрузка модели: {model_path}")
        
        self.asr_model, _ = setup_model(
            cfg=OmegaConf.create({"model_path": model_path}),
            map_location=self.device
        )
        self.asr_model = self.asr_model.to(self.device)
        self.asr_model.eval()
        
        # Получаем sample rate
        if hasattr(self.asr_model, 'preprocessor') and hasattr(self.asr_model.preprocessor, 'featurizer'):
            if hasattr(self.asr_model.preprocessor.featurizer, 'sample_rate'):
                self.sample_rate = self.asr_model.preprocessor.featurizer.sample_rate
        
        logger.info(f"Модель загружена, sample_rate={self.sample_rate}")
    
    def _get_initial_cache_state(self) -> dict:
        """Создаёт начальное состояние кэша."""
        batch_size = 1
        cache_last_channel, cache_last_time, cache_last_channel_len = \
            self.asr_model.encoder.get_initial_cache_state(batch_size=batch_size)
        
        return {
            "cache_last_channel": cache_last_channel.to(self.device),
            "cache_last_time": cache_last_time.to(self.device),
            "cache_last_channel_len": cache_last_channel_len,
            "previous_hypotheses": None,
            "pred_out_stream": None,
            "audio_buffer": np.array([], dtype=np.float32),
            "full_transcription": "",
        }
    
    def _transcribe_chunk(self, audio: np.ndarray, state: dict) -> tuple:
        """Транскрибирует аудио чанк с использованием кэша."""
        from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
        
        # Конвертируем в tensor
        audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(
            device=self.device, dtype=torch.float32
        )
        audio_lengths = torch.tensor([len(audio)], device=self.device)
        
        with torch.inference_mode():
            # Preprocessor
            processed_signal, processed_signal_length = self.asr_model.preprocessor(
                input_signal=audio_tensor,
                length=audio_lengths
            )
            
            # Streaming step
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
                cache_last_channel=state["cache_last_channel"],
                cache_last_time=state["cache_last_time"],
                cache_last_channel_len=state["cache_last_channel_len"],
                keep_all_outputs=True,
                previous_hypotheses=state["previous_hypotheses"],
                previous_pred_out=state["pred_out_stream"],
                drop_extra_pre_encoded=0,
                return_transcription=True,
            )
        
        # Обновляем состояние
        state["cache_last_channel"] = cache_last_channel
        state["cache_last_time"] = cache_last_time
        state["cache_last_channel_len"] = cache_last_channel_len
        state["previous_hypotheses"] = previous_hypotheses
        state["pred_out_stream"] = pred_out_stream
        
        # Получаем текст
        text = ""
        if transcribed_texts:
            if isinstance(transcribed_texts[0], Hypothesis):
                text = transcribed_texts[0].text
            else:
                text = str(transcribed_texts[0])
        
        state["full_transcription"] = text
        
        return text, state
    
    def execute(self, requests: List) -> List:
        """Обрабатывает batch запросов."""
        responses = []
        
        for request in requests:
            try:
                response = self._process_request(request)
                responses.append(response)
            except Exception as e:
                logger.error(f"Ошибка обработки запроса: {e}")
                error_response = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(str(e))
                )
                responses.append(error_response)
        
        return responses
    
    def _process_request(self, request) -> pb_utils.InferenceResponse:
        """Обрабатывает один запрос."""
        # Получаем входные данные
        audio_tensor = pb_utils.get_input_tensor_by_name(request, "audio_signal")
        audio_data = audio_tensor.as_numpy().flatten().astype(np.float32)
        
        # Получаем sequence ID (для streaming)
        sequence_id = request.correlation_id()
        sequence_start = request.flags() & pb_utils.TRITONSERVER_REQUEST_FLAG_SEQUENCE_START
        sequence_end = request.flags() & pb_utils.TRITONSERVER_REQUEST_FLAG_SEQUENCE_END
        
        logger.debug(f"Request: seq_id={sequence_id}, start={sequence_start}, end={sequence_end}")
        
        # Управление состоянием последовательности
        if sequence_start or sequence_id not in self.sequence_states:
            # Новая последовательность
            self.sequence_states[sequence_id] = self._get_initial_cache_state()
            logger.info(f"Новая последовательность: {sequence_id}")
        
        state = self.sequence_states[sequence_id]
        
        # Добавляем аудио в буфер
        state["audio_buffer"] = np.concatenate([state["audio_buffer"], audio_data])
        
        transcription = ""
        
        # Транскрибируем если достаточно данных
        if len(state["audio_buffer"]) >= self.min_chunk_samples or sequence_end:
            if len(state["audio_buffer"]) > 0:
                transcription, state = self._transcribe_chunk(state["audio_buffer"], state)
                state["audio_buffer"] = np.array([], dtype=np.float32)
        
        # Завершение последовательности
        if sequence_end:
            # Финальная транскрипция
            transcription = state["full_transcription"]
            # Очищаем состояние
            del self.sequence_states[sequence_id]
            logger.info(f"Последовательность завершена: {sequence_id}")
        
        # Создаём ответ
        transcription_tensor = pb_utils.Tensor(
            "transcription",
            np.array([transcription], dtype=object)
        )
        
        response = pb_utils.InferenceResponse(
            output_tensors=[transcription_tensor]
        )
        
        return response
    
    def finalize(self):
        """Очистка при выгрузке модели."""
        logger.info("Выгрузка Streaming ASR")
        self.sequence_states.clear()
        if hasattr(self, 'asr_model'):
            del self.asr_model
        torch.cuda.empty_cache()

