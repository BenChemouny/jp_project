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
class ClientConfig:
    server_ws_url: str
    sample_rate: int
    frame_ms: int
    pre_roll_ms: int
    hangover_ms: int
    min_speech_ms: int
    vad_start_threshold: float
    vad_continue_threshold: float
    webrtc_vad_aggressiveness: int
    max_client_segment_ms: int
    disconnected_reset_ms: int
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
        "--webrtc-vad-aggressiveness",
        type=int,
        choices=(0, 1, 2, 3),
        default=int(os.getenv("WEBRTC_VAD_AGGRESSIVENESS", "3")),
    )
    parser.add_argument(
        "--max-client-segment-ms",
        type=int,
        default=int(os.getenv("MAX_CLIENT_SEGMENT_MS", "10000")),
    )
    parser.add_argument(
        "--disconnected-reset-ms",
        type=int,
        default=int(os.getenv("DISCONNECTED_RESET_MS", "1500")),
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
        default=os.getenv("SILERO_VAD_ONNX_PATH"),
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
        webrtc_vad_aggressiveness=args.webrtc_vad_aggressiveness,
        max_client_segment_ms=args.max_client_segment_ms,
        disconnected_reset_ms=args.disconnected_reset_ms,
        enable_noise_reduction=args.enable_noise_reduction,
        high_pass_hz=args.high_pass_hz,
        input_device=input_device,
        vad_backend=args.vad_backend,
        silero_vad_onnx_path=args.silero_vad_onnx_path,
        reconnect_initial_delay_s=args.reconnect_initial_delay_s,
        reconnect_max_delay_s=args.reconnect_max_delay_s,
        audio_queue_size=args.audio_queue_size,
    )
