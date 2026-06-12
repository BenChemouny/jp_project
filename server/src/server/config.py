from __future__ import annotations

import argparse
from dataclasses import dataclass
import os


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_first(names: tuple[str, ...], default: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip():
            return value
    return default


def _int_env_first(names: tuple[str, ...], default: int) -> int:
    for name in names:
        value = os.getenv(name)
        if value is None or not value.strip():
            continue
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _float_env_first(names: tuple[str, ...], default: float) -> float:
    for name in names:
        value = os.getenv(name)
        if value is None or not value.strip():
            continue
        try:
            return float(value)
        except ValueError:
            return default
    return default


DEFAULT_ASR_MODEL = "Qwen/Qwen3-ASR-1.7B"
DEFAULT_ASR_DEVICE = "cuda:0"
DEFAULT_ASR_DTYPE = "bfloat16"
DEFAULT_ASR_MAX_BATCH = 32
DEFAULT_ASR_MAX_NEW_TOKENS = 512
DEFAULT_ASR_LANGUAGE = "Japanese"
DEFAULT_ASR_ATTN_IMPLEMENTATION = "sdpa"
DEFAULT_QWEN3_ASR_CONTEXT = (
    "日本語音声を正確に書き起こしてください。"
    "一般的な語彙、固有名詞、専門用語は文脈に応じて自然な漢字表記を優先し、"
    "不必要なひらがな表記を避けてください。翻訳や要約はせず、発話内容のみを出力してください。"
)


@dataclass(frozen=True)
class ServerConfig:
    host: str
    port: int
    asr_interval_ms: int
    sample_rate: int
    channels: int
    audio_format: str
    asr_model: str
    device: str
    max_segment_seconds: float
    send_partials_to_client: bool
    dtype: str
    max_inference_batch_size: int
    max_new_tokens: int
    language: str
    attn_implementation: str
    qwen3_asr_context: str
    min_asr_segment_ms: int
    min_asr_rms_dbfs: float
    max_asr_silence_ratio: float
    max_asr_low_level_ratio: float
    suppress_short_fillers: bool

    @property
    def bytes_per_second(self) -> int:
        return self.sample_rate * self.channels * 2

    @property
    def max_segment_bytes(self) -> int:
        return int(self.max_segment_seconds * self.bytes_per_second)


def load_config(argv: list[str] | None = None) -> ServerConfig:
    parser = argparse.ArgumentParser(
        description="Receive speech-only PCM audio over WebSocket and stream partial ASR results."
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
        "--asr-model",
        "--qwen3-asr-model-path",
        dest="asr_model",
        default=_env_first(
            ("QWEN3_ASR_MODEL_PATH", "QWEN3_ASR_MODEL"),
            DEFAULT_ASR_MODEL,
        ),
    )
    parser.add_argument(
        "--device",
        default=_env_first(
            ("DEVICE", "QWEN3_ASR_DEVICE"),
            DEFAULT_ASR_DEVICE,
        ),
    )
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
    parser.add_argument(
        "--dtype",
        default=_env_first(
            ("QWEN3_ASR_DTYPE",),
            DEFAULT_ASR_DTYPE,
        ),
    )
    parser.add_argument(
        "--max-inference-batch-size",
        type=int,
        default=_int_env_first(
            (
                "QWEN3_ASR_MAX_INFERENCE_BATCH_SIZE",
                "QWEN3_ASR_MAX_BATCH",
            ),
            DEFAULT_ASR_MAX_BATCH,
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=_int_env_first(
            ("QWEN3_ASR_MAX_NEW_TOKENS",),
            DEFAULT_ASR_MAX_NEW_TOKENS,
        ),
    )
    parser.add_argument(
        "--language",
        default=_env_first(
            ("QWEN3_ASR_LANGUAGE",),
            DEFAULT_ASR_LANGUAGE,
        ),
    )
    parser.add_argument(
        "--attn-implementation",
        default=_env_first(
            ("QWEN3_ASR_ATTN_IMPLEMENTATION",),
            DEFAULT_ASR_ATTN_IMPLEMENTATION,
        ),
    )
    parser.add_argument(
        "--qwen3-asr-context",
        default=_env_first(("QWEN3_ASR_CONTEXT",), DEFAULT_QWEN3_ASR_CONTEXT),
        help="Prompt context passed to Qwen3-ASR. Use an empty string to disable.",
    )
    parser.add_argument(
        "--min-asr-segment-ms",
        type=int,
        default=_int_env_first(("MIN_ASR_SEGMENT_MS",), 650),
    )
    parser.add_argument(
        "--min-asr-rms-dbfs",
        type=float,
        default=_float_env_first(("MIN_ASR_RMS_DBFS",), -55.0),
    )
    parser.add_argument(
        "--max-asr-silence-ratio",
        type=float,
        default=_float_env_first(("MAX_ASR_SILENCE_RATIO",), 0.98),
    )
    parser.add_argument(
        "--max-asr-low-level-ratio",
        type=float,
        default=_float_env_first(("MAX_ASR_LOW_LEVEL_RATIO",), 0.995),
    )
    parser.add_argument(
        "--suppress-short-fillers",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("SUPPRESS_SHORT_FILLERS", True),
    )

    args = parser.parse_args(argv)
    return ServerConfig(
        host=args.host,
        port=args.port,
        asr_interval_ms=args.asr_interval_ms,
        sample_rate=args.sample_rate,
        channels=args.channels,
        audio_format=args.audio_format,
        asr_model=args.asr_model,
        device=args.device,
        max_segment_seconds=args.max_segment_seconds,
        send_partials_to_client=args.send_partials_to_client,
        dtype=args.dtype,
        max_inference_batch_size=args.max_inference_batch_size,
        max_new_tokens=args.max_new_tokens,
        language=args.language,
        attn_implementation=args.attn_implementation,
        qwen3_asr_context=args.qwen3_asr_context,
        min_asr_segment_ms=args.min_asr_segment_ms,
        min_asr_rms_dbfs=args.min_asr_rms_dbfs,
        max_asr_silence_ratio=args.max_asr_silence_ratio,
        max_asr_low_level_ratio=args.max_asr_low_level_ratio,
        suppress_short_fillers=args.suppress_short_fillers,
    )
