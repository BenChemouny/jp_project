from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from server.asr_interface import ASR
from server.asr_types import AudioData


@dataclass
class Qwen3AsrConfig:
    model: str
    device: str
    dtype: str
    max_inference_batch_size: int
    max_new_tokens: int
    language: str
    attn_implementation: str
    context: str


class Qwen3AsrTransformers(ASR):
    def __init__(self, config: Qwen3AsrConfig) -> None:
        self._config = config
        self._model = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        import torch
        from qwen_asr import Qwen3ASRModel

        dtype = _resolve_dtype(self._config.dtype, torch)
        kwargs = {
            "device_map": self._config.device,
            "max_inference_batch_size": self._config.max_inference_batch_size,
            "max_new_tokens": self._config.max_new_tokens,
            "attn_implementation": self._config.attn_implementation,
        }
        if dtype is not None:
            kwargs["dtype"] = dtype

        self._model = Qwen3ASRModel.from_pretrained(self._config.model, **kwargs)

    def transcribe(self, audio: AudioData) -> str:
        self._load_model()
        if self._model is None:
            return ""
        results = self._model.transcribe(
            audio=audio.wav_path,
            context=self._config.context,
            language=self._config.language,
        )
        if not results:
            return ""
        return results[0].text

    def warmup(self) -> None:
        self._load_model()


def _resolve_dtype(dtype_name: str, torch) -> Optional[object]:
    name = dtype_name.strip().lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"f16", "float16", "fp16"}:
        return torch.float16
    if name in {"f32", "float32", "fp32"}:
        return torch.float32
    return None
