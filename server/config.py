from __future__ import annotations

import argparse
from dataclasses import dataclass
import os


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ServerConfig:
    host: str
    port: int
    asr_interval_ms: int
    sample_rate: int
    channels: int
    audio_format: str
    qwen3_asr_model_path: str
    device: str
    max_segment_seconds: float
    send_partials_to_client: bool
    dtype: str
    max_inference_batch_size: int
    max_new_tokens: int
    language: str
    attn_implementation: str

    @property
    def bytes_per_second(self) -> int:
        return self.sample_rate * self.channels * 2

    @property
    def max_segment_bytes(self) -> int:
        return int(self.max_segment_seconds * self.bytes_per_second)


def load_config(argv: list[str] | None = None) -> ServerConfig:
    parser = argparse.ArgumentParser(
        description="Receive speech-only PCM audio over WebSocket and stream partial Qwen3-ASR results."
    )
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    parser.add_argument(
        "--asr-interval-ms",
        type=int,
        default=int(os.getenv("ASR_INTERVAL_MS", "500")),
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=int(os.getenv("SAMPLE_RATE", "16000")),
    )
    parser.add_argument("--channels", type=int, default=int(os.getenv("CHANNELS", "1")))
    parser.add_argument("--audio-format", default=os.getenv("AUDIO_FORMAT", "pcm_s16le"))
    parser.add_argument(
        "--qwen3-asr-model-path",
        default=os.getenv("QWEN3_ASR_MODEL_PATH", "Qwen/Qwen3-ASR-Flash"),
    )
    parser.add_argument("--device", default=os.getenv("DEVICE", "cuda"))
    parser.add_argument(
        "--max-segment-seconds",
        type=float,
        default=float(os.getenv("MAX_SEGMENT_SECONDS", "30")),
    )
    parser.add_argument(
        "--send-partials-to-client",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("SEND_PARTIALS_TO_CLIENT", True),
    )
    parser.add_argument("--dtype", default=os.getenv("QWEN3_ASR_DTYPE", "bfloat16"))
    parser.add_argument(
        "--max-inference-batch-size",
        type=int,
        default=int(os.getenv("QWEN3_ASR_MAX_INFERENCE_BATCH_SIZE", "1")),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=int(os.getenv("QWEN3_ASR_MAX_NEW_TOKENS", "256")),
    )
    parser.add_argument("--language", default=os.getenv("QWEN3_ASR_LANGUAGE", "auto"))
    parser.add_argument(
        "--attn-implementation",
        default=os.getenv("QWEN3_ASR_ATTN_IMPLEMENTATION", "sdpa"),
    )

    args = parser.parse_args(argv)
    return ServerConfig(
        host=args.host,
        port=args.port,
        asr_interval_ms=args.asr_interval_ms,
        sample_rate=args.sample_rate,
        channels=args.channels,
        audio_format=args.audio_format,
        qwen3_asr_model_path=args.qwen3_asr_model_path,
        device=args.device,
        max_segment_seconds=args.max_segment_seconds,
        send_partials_to_client=args.send_partials_to_client,
        dtype=args.dtype,
        max_inference_batch_size=args.max_inference_batch_size,
        max_new_tokens=args.max_new_tokens,
        language=args.language,
        attn_implementation=args.attn_implementation,
    )
