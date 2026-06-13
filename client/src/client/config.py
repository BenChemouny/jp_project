from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _default_silero_vad_onnx_path() -> str | None:
    env_path = os.getenv("SILERO_VAD_ONNX_PATH")
    if env_path:
        return env_path

    model_path = Path(__file__).resolve().parents[2] / "models" / "silero_vad.onnx"
    if model_path.exists():
        return str(model_path)
    return None


@dataclass(frozen=True)
class ClientConfig:
    server_ws_url: str
    sample_rate: int
    frame_ms: int
    pre_roll_ms: int
    hangover_ms: int
    min_speech_ms: int
    vad_start_threshold: float
    vad_continue_threshold: float
    dynamic_vad: bool
    vad_min_start_threshold: float
    vad_min_continue_threshold: float
    vad_noise_margin_db: float
    vad_speech_margin_db: float
    vad_energy_fallback: bool
    vad_energy_start_margin_db: float
    vad_energy_continue_margin_db: float
    enable_noise_reduction: bool
    high_pass_hz: float
    input_device: str | int | None
    vad_backend: str
    silero_vad_onnx_path: str | None
    reconnect_initial_delay_s: float
    reconnect_max_delay_s: float
    audio_queue_size: int

    @property
    def frame_samples(self) -> int:
        return int(self.sample_rate * self.frame_ms / 1000)

    @property
    def frame_bytes(self) -> int:
        return self.frame_samples * 2


def load_config(argv: list[str] | None = None) -> ClientConfig:
    parser = argparse.ArgumentParser(
        description="Capture microphone speech locally and stream PCM frames to the ASR WebSocket server."
    )
    parser.add_argument(
        "--server-ws-url",
        default=os.getenv("SERVER_WS_URL", "ws://localhost:8000/ws/audio"),
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=int(os.getenv("SAMPLE_RATE", "16000")),
    )
    parser.add_argument("--frame-ms", type=int, default=int(os.getenv("FRAME_MS", "30")))
    parser.add_argument(
        "--pre-roll-ms",
        type=int,
        default=int(os.getenv("PRE_ROLL_MS", "500")),
    )
    parser.add_argument(
        "--hangover-ms",
        type=int,
        default=int(os.getenv("HANGOVER_MS", "1000")),
    )
    parser.add_argument(
        "--min-speech-ms",
        type=int,
        default=int(os.getenv("MIN_SPEECH_MS", "300")),
    )
    parser.add_argument(
        "--vad-start-threshold",
        type=float,
        default=float(os.getenv("VAD_START_THRESHOLD", "0.65")),
    )
    parser.add_argument(
        "--vad-continue-threshold",
        type=float,
        default=float(os.getenv("VAD_CONTINUE_THRESHOLD", "0.45")),
    )
    parser.add_argument(
        "--dynamic-vad",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("DYNAMIC_VAD", True),
    )
    parser.add_argument(
        "--vad-min-start-threshold",
        type=float,
        default=float(os.getenv("VAD_MIN_START_THRESHOLD", "0.35")),
    )
    parser.add_argument(
        "--vad-min-continue-threshold",
        type=float,
        default=float(os.getenv("VAD_MIN_CONTINUE_THRESHOLD", "0.25")),
    )
    parser.add_argument(
        "--vad-noise-margin-db",
        type=float,
        default=float(os.getenv("VAD_NOISE_MARGIN_DB", "6.0")),
    )
    parser.add_argument(
        "--vad-speech-margin-db",
        type=float,
        default=float(os.getenv("VAD_SPEECH_MARGIN_DB", "12.0")),
    )
    parser.add_argument(
        "--vad-energy-fallback",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("VAD_ENERGY_FALLBACK", True),
    )
    parser.add_argument(
        "--vad-energy-start-margin-db",
        type=float,
        default=float(os.getenv("VAD_ENERGY_START_MARGIN_DB", "9.0")),
    )
    parser.add_argument(
        "--vad-energy-continue-margin-db",
        type=float,
        default=float(os.getenv("VAD_ENERGY_CONTINUE_MARGIN_DB", "13.0")),
    )
    parser.add_argument(
        "--enable-noise-reduction",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("ENABLE_NOISE_REDUCTION", True),
    )
    parser.add_argument(
        "--high-pass-hz",
        type=float,
        default=float(os.getenv("HIGH_PASS_HZ", "100")),
    )
    parser.add_argument("--input-device", default=os.getenv("INPUT_DEVICE"))
    parser.add_argument(
        "--vad-backend",
        choices=("auto", "silero", "webrtc", "energy"),
        default=os.getenv("VAD_BACKEND", "auto"),
    )
    parser.add_argument(
        "--silero-vad-onnx-path",
        default=_default_silero_vad_onnx_path(),
    )
    parser.add_argument(
        "--reconnect-initial-delay-s",
        type=float,
        default=float(os.getenv("RECONNECT_INITIAL_DELAY_S", "1.0")),
    )
    parser.add_argument(
        "--reconnect-max-delay-s",
        type=float,
        default=float(os.getenv("RECONNECT_MAX_DELAY_S", "15.0")),
    )
    parser.add_argument(
        "--audio-queue-size",
        type=int,
        default=int(os.getenv("AUDIO_QUEUE_SIZE", "200")),
    )
    args = parser.parse_args(argv)

    input_device: str | int | None = args.input_device
    if isinstance(input_device, str) and input_device.isdigit():
        input_device = int(input_device)

    return ClientConfig(
        server_ws_url=args.server_ws_url,
        sample_rate=args.sample_rate,
        frame_ms=args.frame_ms,
        pre_roll_ms=args.pre_roll_ms,
        hangover_ms=args.hangover_ms,
        min_speech_ms=args.min_speech_ms,
        vad_start_threshold=args.vad_start_threshold,
        vad_continue_threshold=args.vad_continue_threshold,
        dynamic_vad=args.dynamic_vad,
        vad_min_start_threshold=args.vad_min_start_threshold,
        vad_min_continue_threshold=args.vad_min_continue_threshold,
        vad_noise_margin_db=args.vad_noise_margin_db,
        vad_speech_margin_db=args.vad_speech_margin_db,
        vad_energy_fallback=args.vad_energy_fallback,
        vad_energy_start_margin_db=args.vad_energy_start_margin_db,
        vad_energy_continue_margin_db=args.vad_energy_continue_margin_db,
        enable_noise_reduction=args.enable_noise_reduction,
        high_pass_hz=args.high_pass_hz,
        input_device=input_device,
        vad_backend=args.vad_backend,
        silero_vad_onnx_path=args.silero_vad_onnx_path,
        reconnect_initial_delay_s=args.reconnect_initial_delay_s,
        reconnect_max_delay_s=args.reconnect_max_delay_s,
        audio_queue_size=args.audio_queue_size,
    )
