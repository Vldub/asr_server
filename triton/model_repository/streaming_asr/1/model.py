"""
Triton Python Backend для Streaming ASR с NeMo.
Поддержка Decoupled Mode для отправки промежуточных результатов.
"""

import json
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ленивый импорт для совместимости
pb_utils = None
torch = None


def lazy_import():
    global pb_utils, torch
    if pb_utils is None:
        import triton_python_backend_utils as _pb_utils
        pb_utils = _pb_utils
    if torch is None:
        import torch as _torch
        torch = _torch


class TritonPythonModel:
    """Triton Python Backend для Streaming ASR с Decoupled Mode."""
    
    def initialize(self, args):
        """Инициализация модели при загрузке."""
        lazy_import()
        
        self.model_config = json.loads(args["model_config"])
        
        # Проверяем decoupled mode
        self.decoupled = self.model_config.get("model_transaction_policy", {}).get("decoupled", False)
        logger.info(f"Decoupled mode: {self.decoupled}")
        
        # Определяем устройство
        device_id = args.get("model_instance_device_id", "0")
        if torch.cuda.is_available() and int(device_id) >= 0:
            self.device = torch.device(f"cuda:{device_id}")
        else:
            self.device = torch.device("cpu")
        
        logger.info(f"Инициализация Streaming ASR на устройстве: {self.device}")
        
        # Загружаем NeMo модель
        self._load_model()
        
        # Кэш состояний для активных последовательностей
        self.sequence_states = {}
        
        # Оптимальный размер чанка для модели
        # 1040ms при 16kHz = 16640 samples
        # Можно переопределить через параметры модели
        self.sample_rate = int(self.model_config.get("parameters", {}).get(
            "sample_rate", {"string_value": "16000"}
        ).get("string_value", "16000"))
        
        optimal_chunk_ms = int(self.model_config.get("parameters", {}).get(
            "optimal_chunk_ms", {"string_value": "1040"}
        ).get("string_value", "1040"))
        
        self.min_chunk_samples = int(optimal_chunk_ms * self.sample_rate / 1000)
        
        logger.info(f"Оптимальный размер чанка: {optimal_chunk_ms}ms = {self.min_chunk_samples} samples")
        
        # Статистика для мониторинга
        self.batch_stats = {
            "total_batches": 0,
            "total_requests": 0,
            "max_batch_size": 0,
        }
        
        logger.info("Streaming ASR с Decoupled Mode инициализирован")
    
    def _load_model(self):
        """Загружает NeMo модель."""
        from omegaconf import OmegaConf
        from nemo.collections.asr.parts.utils.transcribe_utils import setup_model
        
        model_path = "/models/model.nemo"
        logger.info(f"Загрузка модели: {model_path}")
        
        self.asr_model, _ = setup_model(
            cfg=OmegaConf.create({"model_path": model_path}),
            map_location=self.device
        )
        self.asr_model = self.asr_model.to(self.device)
        self.asr_model.eval()
        
        logger.info("Модель NeMo загружена")
    
    def _get_initial_cache_state(self):
        """Создаёт начальное состояние кэша для новой sequence."""
        batch_size = 1
        cache_last_channel, cache_last_time, cache_last_channel_len = \
            self.asr_model.encoder.get_initial_cache_state(batch_size=batch_size)
        
        return {
            "cache_last_channel": cache_last_channel.to(self.device),
            "cache_last_time": cache_last_time.to(self.device),
            "cache_last_channel_len": cache_last_channel_len,
            "previous_hypotheses": None,
            "pred_out_stream": None,
            "audio_buffer": np.array([], dtype=np.float32),  # Буфер для streaming
            "full_audio": np.array([], dtype=np.float32),    # Всё аудио для batch финализации
            "full_transcription": "",
        }
    
    def _transcribe_chunk(self, audio, state):
        """Транскрибирует аудио чанк для одной sequence (streaming mode)."""
        from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
        
        audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(
            device=self.device, dtype=torch.float32
        )
        audio_lengths = torch.tensor([len(audio)], device=self.device)
        
        with torch.inference_mode():
            processed_signal, processed_signal_length = self.asr_model.preprocessor(
                input_signal=audio_tensor,
                length=audio_lengths
            )
            
            # Используем оптимальные параметры из streaming_cfg модели
            # drop_extra_pre_encoded=2 рекомендуется для этой архитектуры
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
                drop_extra_pre_encoded=None,  # Использовать значение из streaming_cfg модели
                return_transcription=True,
            )
        
        state["cache_last_channel"] = cache_last_channel
        state["cache_last_time"] = cache_last_time
        state["cache_last_channel_len"] = cache_last_channel_len
        state["previous_hypotheses"] = previous_hypotheses
        state["pred_out_stream"] = pred_out_stream
        
        text = ""
        if transcribed_texts:
            if isinstance(transcribed_texts[0], Hypothesis):
                text = transcribed_texts[0].text
            else:
                text = str(transcribed_texts[0])
        
        state["full_transcription"] = text
        return text, state
    
    def _transcribe_batch(self, audio):
        """Транскрибирует аудио целиком (batch mode) - лучшее качество."""
        audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(
            device=self.device, dtype=torch.float32
        )
        audio_lengths = torch.tensor([len(audio)], device=self.device)
        
        with torch.inference_mode():
            # Используем полный transcribe для лучшего качества
            hypotheses = self.asr_model.transcribe(
                audio=[audio],
                batch_size=1,
                return_hypotheses=True,
                verbose=False
            )
        
        if hypotheses and len(hypotheses) > 0:
            if hasattr(hypotheses[0], 'text'):
                return hypotheses[0].text
            elif isinstance(hypotheses[0], str):
                return hypotheses[0]
        
        return ""
    
    def _process_single_request(self, request):
        """Обрабатывает один запрос."""
        # Получаем аудио
        audio_tensor = pb_utils.get_input_tensor_by_name(request, "audio_signal")
        audio_data = audio_tensor.as_numpy().flatten().astype(np.float32)
        
        # Sequence ID и флаги
        sequence_id = request.correlation_id()
        flags = request.flags()
        sequence_start = bool(flags & 1)  # SEQUENCE_START
        sequence_end = bool(flags & 2)    # SEQUENCE_END
        
        # Управление состоянием
        if sequence_start or sequence_id not in self.sequence_states:
            self.sequence_states[sequence_id] = self._get_initial_cache_state()
            logger.info(f"Новая сессия: {sequence_id}")
        
        state = self.sequence_states[sequence_id]
        
        # Накапливаем аудио в оба буфера
        state["audio_buffer"] = np.concatenate([state["audio_buffer"], audio_data])
        state["full_audio"] = np.concatenate([state["full_audio"], audio_data])
        
        buffer_samples = len(state["audio_buffer"])
        buffer_ms = buffer_samples * 1000 / self.sample_rate
        
        # Промежуточная транскрипция через streaming mode
        if buffer_samples >= self.min_chunk_samples and not sequence_end:
            if buffer_samples > 0:
                logger.debug(f"Streaming транскрипция: {buffer_ms:.0f}ms ({buffer_samples} samples)")
                transcription, state = self._transcribe_chunk(state["audio_buffer"], state)
                state["audio_buffer"] = np.array([], dtype=np.float32)
        elif not sequence_end:
            # Буфер ещё накапливается - возвращаем последнюю транскрипцию
            transcription = state["full_transcription"]
            logger.debug(f"Накопление буфера: {buffer_ms:.0f}ms / {self.min_chunk_samples * 1000 / self.sample_rate:.0f}ms")
        else:
            transcription = state["full_transcription"]
        
        # Завершение sequence - используем batch mode для лучшего качества
        is_final = False
        if sequence_end:
            full_audio = state["full_audio"]
            full_duration = len(full_audio) / self.sample_rate
            logger.info(f"Финальная batch транскрипция: {full_duration:.2f}s ({len(full_audio)} samples)")
            
            # Batch транскрипция всего аудио для лучшего качества
            transcription = self._transcribe_batch(full_audio)
            
            del self.sequence_states[sequence_id]
            logger.info(f"Сессия завершена: {sequence_id}")
            is_final = True
        
        return transcription, is_final
    
    def execute(self, requests):
        """
        Обрабатывает батч запросов в Decoupled Mode.
        
        В Decoupled Mode ответы отправляются через response_sender,
        что позволяет отправлять промежуточные результаты по мере готовности.
        """
        lazy_import()
        
        batch_size = len(requests)
        self.batch_stats["total_batches"] += 1
        self.batch_stats["total_requests"] += batch_size
        self.batch_stats["max_batch_size"] = max(
            self.batch_stats["max_batch_size"], batch_size
        )
        
        if batch_size > 1:
            logger.debug(f"Обработка батча: {batch_size} запросов")
        
        responses = []
        
        for request in requests:
            try:
                # В Decoupled mode используем response_sender
                if self.decoupled:
                    response_sender = request.get_response_sender()
                
                transcription, is_final = self._process_single_request(request)
                
                # Создаём ответ
                out_tensor = pb_utils.Tensor(
                    "transcription",
                    np.array([transcription], dtype=object)
                )
                response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
                
                if self.decoupled:
                    # В Decoupled mode отправляем через response_sender
                    # flags указывает является ли это финальным ответом
                    flags = pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL if is_final else 0
                    response_sender.send(response, flags=flags)
                    
                    # Если не финальный - закрываем sender после отправки
                    if is_final:
                        responses.append(None)  # Placeholder
                    else:
                        responses.append(None)
                else:
                    responses.append(response)
                
            except Exception as e:
                logger.error(f"Ошибка обработки запроса: {e}")
                import traceback
                traceback.print_exc()
                
                if self.decoupled:
                    response_sender = request.get_response_sender()
                    error = pb_utils.TritonError(str(e))
                    error_response = pb_utils.InferenceResponse(error=error)
                    response_sender.send(
                        error_response, 
                        flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                    )
                    responses.append(None)
                else:
                    error = pb_utils.TritonError(str(e))
                    response = pb_utils.InferenceResponse(error=error)
                    responses.append(response)
        
        # В Decoupled mode возвращаем None (ответы уже отправлены)
        if self.decoupled:
            return None
        return responses
    
    def finalize(self):
        """Очистка при выгрузке модели."""
        logger.info("Выгрузка Streaming ASR")
        logger.info(f"Статистика батчинга: {self.batch_stats}")
        self.sequence_states.clear()
