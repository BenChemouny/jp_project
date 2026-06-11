from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import wave

from server.config import ServerConfig


REPO_ROOT = Path(__file__).resolve().parents[1]


def _install_jp_talk_aliases() -> None:
    """Let the checked-in example keep using its original jp_talk imports."""
    import jp_code_example.core as core
    import jp_code_example.core.types as core_types
    import jp_code_example.impl as impl

    sys.modules.setdefault("jp_talk", sys.modules["jp_code_example"])
    sys.modules.setdefault("jp_talk.core", core)
    sys.modules.setdefault("jp_talk.core.types", core_types)
    sys.modules.setdefault("jp_talk.impl", impl)

    import jp_code_example.core.interfaces as interfaces
    import jp_code_example.core.interfaces.asr as asr_interface
    import jp_code_example.impl.asr as impl_asr

    sys.modules.setdefault("jp_talk.core.interfaces", interfaces)
    sys.modules.setdefault("jp_talk.core.interfaces.asr", asr_interface)
    sys.modules.setdefault("jp_talk.impl.asr", impl_asr)


class Qwen3AsrAdapter:
    def __init__(self, config: ServerConfig) -> None:
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        _install_jp_talk_aliases()

        from jp_code_example.impl.asr.qwen3_transformers import (
            Qwen3AsrConfig,
            Qwen3AsrTransformers,
        )

        asr_config = Qwen3AsrConfig(
            model=config.qwen3_asr_model_path,
            device=config.device,
            dtype=config.dtype,
            max_inference_batch_size=config.max_inference_batch_size,
            max_new_tokens=config.max_new_tokens,
            language=config.language,
            attn_implementation=config.attn_implementation,
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


def pcm_s16le_to_asr_input(buffer: bytes, sample_rate: int, channels: int):
    from jp_code_example.core.types import AudioData

    if len(buffer) % 2 != 0:
        buffer = buffer[:-1]
    frames = len(buffer) // (2 * channels)

    tmp = tempfile.NamedTemporaryFile(
        prefix="qwen3_asr_",
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
