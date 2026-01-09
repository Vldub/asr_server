"""
Triton Python Backend для Streaming ASR с NeMo.
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
    """Triton Python Backend для Streaming ASR."""
    
    def initialize(self, args):
        """Инициализация модели при загрузке."""
        lazy_import()
        
        self.model_config = json.loads(args['model_config'])
        
        # Определяем устройство
        device_id = args.get('model_instance_device_id', '0')
        if torch.cuda.is_available() and int(device_id) >= 0:
            self.device = torch.device(f'cuda:{device_id}')
        else:
            self.device = torch.device('cpu')
        
        logger.info(f"Инициализация Streaming ASR на устройстве: {self.device}")
        
        # Загружаем NeMo модель
        self._load_model()
        
        # Кэш состояний для активных последовательностей
        self.sequence_states = {}
        self.min_chunk_samples = 8000  # 0.5 сек
        
        logger.info("Streaming ASR инициализирован")
    
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
    
    def _transcribe_chunk(self, audio, state):
        """Транскрибирует аудио чанк."""
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
    
    def execute(self, requests):
        """Обрабатывает batch запросов."""
        lazy_import()
        responses = []
        
        for request in requests:
            try:
                # Получаем аудио
                audio_tensor = pb_utils.get_input_tensor_by_name(request, "audio_signal")
                audio_data = audio_tensor.as_numpy().flatten().astype(np.float32)
                
                # Sequence ID
                sequence_id = request.correlation_id()
                flags = request.flags()
                sequence_start = bool(flags & 1)  # SEQUENCE_START
                sequence_end = bool(flags & 2)    # SEQUENCE_END
                
                # Управление состоянием
                if sequence_start or sequence_id not in self.sequence_states:
                    self.sequence_states[sequence_id] = self._get_initial_cache_state()
                    logger.info(f"Новая сессия: {sequence_id}")
                
                state = self.sequence_states[sequence_id]
                state["audio_buffer"] = np.concatenate([state["audio_buffer"], audio_data])
                
                transcription = ""
                
                # Транскрибируем
                if len(state["audio_buffer"]) >= self.min_chunk_samples or sequence_end:
                    if len(state["audio_buffer"]) > 0:
                        transcription, state = self._transcribe_chunk(state["audio_buffer"], state)
                        state["audio_buffer"] = np.array([], dtype=np.float32)
                
                if sequence_end:
                    transcription = state["full_transcription"]
                    del self.sequence_states[sequence_id]
                    logger.info(f"Сессия завершена: {sequence_id}")
                
                # Ответ
                out_tensor = pb_utils.Tensor(
                    "transcription",
                    np.array([transcription], dtype=object)
                )
                response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
                responses.append(response)
                
            except Exception as e:
                logger.error(f"Ошибка: {e}")
                import traceback
                traceback.print_exc()
                error = pb_utils.TritonError(str(e))
                response = pb_utils.InferenceResponse(error=error)
                responses.append(response)
        
        return responses
    
    def finalize(self):
        """Очистка."""
        logger.info("Выгрузка Streaming ASR")
        self.sequence_states.clear()
