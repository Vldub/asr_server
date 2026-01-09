#!/usr/bin/env python3
"""
Скрипт для экспорта NeMo ASR модели в формат ONNX для Triton Inference Server.

Экспортирует компоненты модели отдельно:
- Preprocessor (feature extraction)
- Encoder (acoustic model)
- Decoder (joint network для RNNT)

Использование:
    python export_model.py --model /path/to/model.nemo --output-dir ../model_repository
"""

import argparse
import logging
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import onnx
from onnxruntime import InferenceSession

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def export_encoder_onnx(model, output_path: str, batch_size: int = 1):
    """Экспортирует encoder в ONNX с поддержкой streaming (cache_state)."""
    logger.info("Экспорт Encoder в ONNX...")
    
    encoder = model.encoder
    encoder.eval()
    
    # Получаем начальное состояние кэша
    cache_last_channel, cache_last_time, cache_last_channel_len = \
        encoder.get_initial_cache_state(batch_size=batch_size)
    
    # Размеры для примера входных данных
    # После preprocessor: [batch, features, time]
    # Типичные значения: features=80 (mel), time=50 (0.5 сек)
    n_features = 80
    chunk_time = 50  # ~0.5 сек после feature extraction
    
    # Создаём примеры входных данных
    dummy_audio_signal = torch.randn(batch_size, n_features, chunk_time)
    dummy_length = torch.tensor([chunk_time] * batch_size)
    
    # Перемещаем на то же устройство что и модель
    device = next(encoder.parameters()).device
    dummy_audio_signal = dummy_audio_signal.to(device)
    dummy_length = dummy_length.to(device)
    
    # Wrapper для encoder с cache
    class EncoderWrapper(torch.nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder
        
        def forward(self, audio_signal, length, cache_last_channel, cache_last_time):
            # Streaming forward
            encoded, encoded_len, cache_last_channel_out, cache_last_time_out = \
                self.encoder.forward_for_export(
                    audio_signal=audio_signal,
                    length=length,
                    cache_last_channel=cache_last_channel,
                    cache_last_time=cache_last_time,
                )
            return encoded, encoded_len, cache_last_channel_out, cache_last_time_out
    
    wrapper = EncoderWrapper(encoder)
    wrapper.eval()
    
    try:
        # Экспорт в ONNX
        torch.onnx.export(
            wrapper,
            (dummy_audio_signal, dummy_length, cache_last_channel, cache_last_time),
            output_path,
            input_names=['audio_signal', 'length', 'cache_last_channel', 'cache_last_time'],
            output_names=['encoded', 'encoded_len', 'cache_last_channel_out', 'cache_last_time_out'],
            dynamic_axes={
                'audio_signal': {0: 'batch', 2: 'time'},
                'length': {0: 'batch'},
                'encoded': {0: 'batch', 1: 'time'},
                'encoded_len': {0: 'batch'},
            },
            opset_version=17,
            do_constant_folding=True,
        )
        logger.info(f"✓ Encoder экспортирован: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Ошибка экспорта encoder: {e}")
        # Попробуем альтернативный метод через NeMo
        logger.info("Попытка экспорта через NeMo API...")
        try:
            model.export(output_path.replace('encoder', 'full_model'), check_trace=False)
            logger.info(f"✓ Полная модель экспортирована через NeMo API")
            return True
        except Exception as e2:
            logger.error(f"Ошибка экспорта через NeMo: {e2}")
            return False


def export_decoder_joint_onnx(model, output_path: str, batch_size: int = 1):
    """Экспортирует decoder joint network в ONNX."""
    logger.info("Экспорт Decoder Joint в ONNX...")
    
    if not hasattr(model, 'decoder') or not hasattr(model.decoder, 'joint'):
        logger.warning("Decoder joint не найден в модели")
        return False
    
    joint = model.decoder.joint
    joint.eval()
    
    # Размеры для примера
    hidden_size = model.decoder.joint.encoder_hidden  # обычно 512 или 1024
    pred_hidden = model.decoder.joint.pred_hidden
    
    dummy_encoder_output = torch.randn(batch_size, 1, hidden_size)
    dummy_decoder_output = torch.randn(batch_size, 1, pred_hidden)
    
    device = next(joint.parameters()).device
    dummy_encoder_output = dummy_encoder_output.to(device)
    dummy_decoder_output = dummy_decoder_output.to(device)
    
    try:
        torch.onnx.export(
            joint,
            (dummy_encoder_output, dummy_decoder_output),
            output_path,
            input_names=['encoder_output', 'decoder_output'],
            output_names=['joint_output'],
            dynamic_axes={
                'encoder_output': {0: 'batch', 1: 'time'},
                'decoder_output': {0: 'batch', 1: 'time'},
                'joint_output': {0: 'batch', 1: 'time'},
            },
            opset_version=17,
            do_constant_folding=True,
        )
        logger.info(f"✓ Decoder Joint экспортирован: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Ошибка экспорта decoder joint: {e}")
        return False


def export_full_model_onnx(model, output_dir: str):
    """Экспортирует полную модель через NeMo API."""
    logger.info("Экспорт полной модели через NeMo API...")
    
    output_path = os.path.join(output_dir, "model.onnx")
    
    try:
        # NeMo модели имеют встроенный метод export
        model.export(output_path, check_trace=False)
        logger.info(f"✓ Модель экспортирована: {output_path}")
        
        # Проверяем ONNX
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        logger.info("✓ ONNX модель валидна")
        
        return output_path
    except Exception as e:
        logger.error(f"Ошибка экспорта: {e}")
        return None


def create_triton_config(model_name: str, output_dir: str, max_batch_size: int = 8):
    """Создаёт конфигурационный файл для Triton."""
    config_template = f'''name: "{model_name}"
platform: "onnxruntime_onnx"
max_batch_size: {max_batch_size}

input [
  {{
    name: "audio_signal"
    data_type: TYPE_FP32
    dims: [ -1 ]  # variable length audio
  }}
]

output [
  {{
    name: "transcription"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }}
]

# Sequence batching для streaming
sequence_batching {{
  max_sequence_idle_microseconds: 10000000  # 10 секунд
  oldest {{
    max_candidate_sequences: {max_batch_size}
    preferred_batch_size: [ 4, {max_batch_size} ]
    max_queue_delay_microseconds: 100000  # 100ms
  }}
  control_input [
    {{
      name: "START"
      control [
        {{
          kind: CONTROL_SEQUENCE_START
          fp32_false_true: [ 0, 1 ]
        }}
      ]
    }},
    {{
      name: "END" 
      control [
        {{
          kind: CONTROL_SEQUENCE_END
          fp32_false_true: [ 0, 1 ]
        }}
      ]
    }},
    {{
      name: "CORRID"
      control [
        {{
          kind: CONTROL_SEQUENCE_CORRID
          data_type: TYPE_UINT64
        }}
      ]
    }}
  ]
  state [
    {{
      input_name: "cache_state_in"
      output_name: "cache_state_out"
      data_type: TYPE_FP32
      dims: [ -1 ]  # variable cache size
    }}
  ]
}}

instance_group [
  {{
    count: 1
    kind: KIND_GPU
    gpus: [ 0 ]
  }}
]

# Оптимизации
optimization {{
  execution_accelerators {{
    gpu_execution_accelerator: [
      {{
        name: "tensorrt"
        parameters {{
          key: "precision_mode"
          value: "FP16"
        }}
        parameters {{
          key: "max_workspace_size_bytes"
          value: "1073741824"  # 1GB
        }}
      }}
    ]
  }}
}}
'''
    
    config_path = os.path.join(output_dir, "config.pbtxt")
    with open(config_path, 'w') as f:
        f.write(config_template)
    
    logger.info(f"✓ Triton config создан: {config_path}")
    return config_path


def verify_onnx_model(onnx_path: str):
    """Проверяет ONNX модель."""
    logger.info(f"Проверка ONNX модели: {onnx_path}")
    
    try:
        # Загружаем и проверяем
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        
        # Выводим информацию о модели
        logger.info(f"  IR version: {model.ir_version}")
        logger.info(f"  Opset version: {model.opset_import[0].version}")
        logger.info(f"  Inputs: {[inp.name for inp in model.graph.input]}")
        logger.info(f"  Outputs: {[out.name for out in model.graph.output]}")
        
        # Тестируем инференс
        session = InferenceSession(onnx_path)
        logger.info("✓ ONNX Runtime сессия создана успешно")
        
        return True
    except Exception as e:
        logger.error(f"Ошибка проверки: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Экспорт NeMo модели для Triton")
    parser.add_argument("--model", required=True, help="Путь к .nemo модели")
    parser.add_argument("--output-dir", default="../model_repository", help="Директория для моделей")
    parser.add_argument("--max-batch-size", type=int, default=8, help="Максимальный размер батча")
    parser.add_argument("--device", type=int, default=0, help="CUDA устройство (-1 для CPU)")
    
    args = parser.parse_args()
    
    # Импорты NeMo (тяжёлые, поэтому здесь)
    from omegaconf import OmegaConf
    from nemo.collections.asr.models import EncDecRNNTBPEModel
    
    # Устройство
    if args.device >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cpu")
    
    logger.info(f"Устройство: {device}")
    
    # Загрузка модели
    logger.info(f"Загрузка модели: {args.model}")
    model = EncDecRNNTBPEModel.restore_from(args.model, map_location=device)
    model.eval()
    
    # Создаём директории
    output_dir = Path(args.output_dir)
    encoder_dir = output_dir / "encoder" / "1"
    decoder_dir = output_dir / "decoder" / "1"
    full_model_dir = output_dir / "asr_model" / "1"
    
    for d in [encoder_dir, decoder_dir, full_model_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Экспорт полной модели (самый надёжный способ)
    logger.info("\n" + "="*60)
    logger.info("ЭКСПОРТ ПОЛНОЙ МОДЕЛИ")
    logger.info("="*60)
    
    full_model_path = export_full_model_onnx(model, str(full_model_dir))
    
    if full_model_path:
        verify_onnx_model(full_model_path)
        
        # Создаём Triton config
        create_triton_config("asr_model", str(output_dir / "asr_model"), args.max_batch_size)
    
    # Попытка экспорта компонентов отдельно (для более гибкой настройки)
    logger.info("\n" + "="*60)
    logger.info("ЭКСПОРТ КОМПОНЕНТОВ (опционально)")
    logger.info("="*60)
    
    # encoder_path = str(encoder_dir / "model.onnx")
    # export_encoder_onnx(model, encoder_path)
    
    # decoder_path = str(decoder_dir / "model.onnx")
    # export_decoder_joint_onnx(model, decoder_path)
    
    logger.info("\n" + "="*60)
    logger.info("ЭКСПОРТ ЗАВЕРШЁН")
    logger.info("="*60)
    logger.info(f"Модели сохранены в: {output_dir}")
    logger.info("\nСледующие шаги:")
    logger.info("1. Запустите Triton: docker compose -f triton-compose.yml up")
    logger.info("2. Проверьте: curl localhost:8000/v2/health/ready")


if __name__ == "__main__":
    main()

