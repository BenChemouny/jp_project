from __future__ import annotations

import os
import tempfile
import wave

from server.asr_types import AudioData
from server.config import ServerConfig
from server.qwen3_transformers import Qwen3AsrConfig, Qwen3AsrTransformers


class Qwen3AsrAdapter:
    def __init__(self, config: ServerConfig) -> None:
        asr_config = Qwen3AsrConfig(
            model=config.asr_model,
            device=config.device,
            dtype=config.dtype,
            max_inference_batch_size=config.max_inference_batch_size,
            max_new_tokens=config.max_new_tokens,
            language=config.language,
            attn_implementation=config.attn_implementation,
            context=config.qwen3_asr_context,
        )
        self._asr = Qwen3AsrTransformers(asr_config)

    def warmup(self) -> None:
        self._asr.warmup()

    def transcribe_pcm_s16le(
        self,
        buffer: bytes,
        sample_rate: int,
        channels: int,
    ) -> str:
        audio = pcm_s16le_to_asr_input(buffer, sample_rate, channels)
        try:
            return self._asr.transcribe(audio).strip()
        finally:
            try:
                os.unlink(audio.wav_path)
            except FileNotFoundError:
                pass


def pcm_s16le_to_asr_input(
    buffer: bytes,
    sample_rate: int,
    channels: int,
) -> AudioData:
    if len(buffer) % 2 != 0:
        buffer = buffer[:-1]
    frames = len(buffer) // (2 * channels)

    tmp = tempfile.NamedTemporaryFile(
        prefix="asr_",
        suffix=".wav",
        delete=False,
    )
    tmp.close()
    with wave.open(tmp.name, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(buffer)

    return AudioData(
        wav_path=tmp.name,
        sample_rate=sample_rate,
        channels=channels,
        frames=frames,
    )
