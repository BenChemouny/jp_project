from __future__ import annotations

import asyncio
from array import array
from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
import math
import signal
import sys
import time
from typing import Any
from uuid import uuid4

from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from server.config import ServerConfig, load_config
from server.qwen3_asr_adapter import Qwen3AsrAdapter


SUPPORTED_FORMATS = {"pcm_s16le"}
INT16_MAX = 32768.0


@dataclass
class PcmFrameMetrics:
    samples: int
    rms_dbfs: float
    peak_dbfs: float
    clip_count: int
    silence_samples: int
    low_level_samples: int


@dataclass
class SegmentAudioMetrics:
    frames: int = 0
    samples: int = 0
    rms_power_sum: float = 0.0
    rms_dbfs_sum: float = 0.0
    min_rms_dbfs: float = 120.0
    max_rms_dbfs: float = -120.0
    peak_dbfs: float = -120.0
    clip_count: int = 0
    silence_samples: int = 0
    low_level_samples: int = 0

    def add_frame(self, frame: bytes) -> None:
        metrics = measure_pcm_s16le(frame)
        if metrics.samples <= 0:
            return
        self.frames += 1
        self.samples += metrics.samples
        self.rms_power_sum += _db_to_power(metrics.rms_dbfs) * metrics.samples
        self.rms_dbfs_sum += metrics.rms_dbfs
        self.min_rms_dbfs = min(self.min_rms_dbfs, metrics.rms_dbfs)
        self.max_rms_dbfs = max(self.max_rms_dbfs, metrics.rms_dbfs)
        self.peak_dbfs = max(self.peak_dbfs, metrics.peak_dbfs)
        self.clip_count += metrics.clip_count
        self.silence_samples += metrics.silence_samples
        self.low_level_samples += metrics.low_level_samples

    @property
    def rms_dbfs(self) -> float:
        if self.samples <= 0:
            return -120.0
        power = self.rms_power_sum / self.samples
        if power <= 1e-16:
            return -120.0
        return 10.0 * math.log10(power)

    @property
    def mean_frame_rms_dbfs(self) -> float:
        if self.frames <= 0:
            return -120.0
        return self.rms_dbfs_sum / self.frames

    @property
    def clip_ratio(self) -> float:
        if self.samples <= 0:
            return 0.0
        return self.clip_count / self.samples

    @property
    def silence_ratio(self) -> float:
        if self.samples <= 0:
            return 0.0
        return self.silence_samples / self.samples

    @property
    def low_level_ratio(self) -> float:
        if self.samples <= 0:
            return 0.0
        return self.low_level_samples / self.samples


@dataclass(frozen=True)
class AsrRunResult:
    text: str
    latency_ms: int
    rtf: float


@dataclass
class AudioSession:
    websocket: ServerConnection
    session_id: str
    config: ServerConfig
    asr: Qwen3AsrAdapter
    is_streaming: bool = False
    audio_buffer: bytearray = field(default_factory=bytearray)
    sample_rate: int = 16000
    channels: int = 1
    format: str = "pcm_s16le"
    stream_started_at: float | None = None
    last_asr_at: float = 0.0
    last_partial_text: str = ""
    partial_task: asyncio.Task[None] | None = None
    stop_partials: asyncio.Event = field(default_factory=asyncio.Event)
    asr_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    audio_metrics: SegmentAudioMetrics = field(default_factory=SegmentAudioMetrics)

    @property
    def bytes_per_second(self) -> int:
        return self.sample_rate * self.channels * 2

    @property
    def buffer_ms(self) -> int:
        if self.bytes_per_second <= 0:
            return 0
        return int(len(self.audio_buffer) / self.bytes_per_second * 1000)

    @property
    def max_segment_bytes(self) -> int:
        return int(self.config.max_segment_seconds * self.bytes_per_second)

    async def start_stream(self, message: dict[str, Any]) -> None:
        if self.is_streaming:
            await self.finish_stream(reason="restart")

        sample_rate = int(message.get("sample_rate", self.config.sample_rate))
        channels = int(message.get("channels", self.config.channels))
        audio_format = str(message.get("format", self.config.audio_format))
        self._validate_audio_metadata(sample_rate, channels, audio_format)

        self.is_streaming = True
        self.audio_buffer = bytearray()
        self.audio_metrics = SegmentAudioMetrics()
        self.sample_rate = sample_rate
        self.channels = channels
        self.format = audio_format
        self.stream_started_at = time.monotonic()
        self.last_asr_at = 0.0
        self.last_partial_text = ""
        self.stop_partials = asyncio.Event()
        self.partial_task = asyncio.create_task(self._partial_loop())
        print(
            f"[{self.session_id}] speech_start "
            f"{self.sample_rate}Hz {self.channels}ch {self.format}",
            flush=True,
        )

    def append_audio(self, frame: bytes) -> bool:
        if not self.is_streaming:
            return False
        self.audio_buffer.extend(frame)
        self.audio_metrics.add_frame(frame)
        return len(self.audio_buffer) <= self.max_segment_bytes

    async def finish_stream(self, reason: str = "speech_end") -> None:
        if not self.is_streaming and not self.audio_buffer:
            return

        self.is_streaming = False
        if self.partial_task is not None:
            self.stop_partials.set()
            await self.partial_task
            self.partial_task = None

        snapshot = bytes(self.audio_buffer)
        buffer_ms = self.buffer_ms
        result = await self._run_asr(snapshot, final=True)
        if result.text:
            print(
                f"[final {buffer_ms}ms] {self._audio_metrics_log()} "
                f"asr={result.latency_ms}ms rtf={result.rtf:.2f} {result.text}",
                flush=True,
            )
        else:
            print(
                f"[final {buffer_ms}ms] {self._audio_metrics_log()} "
                f"asr={result.latency_ms}ms rtf={result.rtf:.2f} <empty>",
                flush=True,
            )
        await self._send_json(
            {
                "type": "final_transcript",
                "text": result.text,
                "buffer_ms": buffer_ms,
                "reason": reason,
                "asr_latency_ms": result.latency_ms,
                "asr_rtf": result.rtf,
            }
        )
        self.audio_buffer.clear()
        self.audio_metrics = SegmentAudioMetrics()
        self.last_partial_text = ""
        self.stream_started_at = None

    async def close(self) -> None:
        self.is_streaming = False
        if self.partial_task is not None:
            self.stop_partials.set()
            await self.partial_task
            self.partial_task = None
        self.audio_buffer.clear()

    async def _partial_loop(self) -> None:
        interval_s = self.config.asr_interval_ms / 1000.0
        while self.is_streaming:
            try:
                await asyncio.wait_for(self.stop_partials.wait(), timeout=interval_s)
                break
            except TimeoutError:
                pass
            if not self.is_streaming or not self.audio_buffer:
                continue
            if self.asr_lock.locked():
                continue

            snapshot = bytes(self.audio_buffer)
            buffer_ms = self.buffer_ms
            result = await self._run_asr(snapshot, final=False)
            self.last_asr_at = time.monotonic()
            if not self.is_streaming:
                break
            if not result.text or result.text == self.last_partial_text:
                continue

            self.last_partial_text = result.text
            print(
                f"[partial {buffer_ms}ms] {self._audio_metrics_log()} "
                f"asr={result.latency_ms}ms rtf={result.rtf:.2f} {result.text}",
                flush=True,
            )
            if self.config.send_partials_to_client:
                await self._send_json(
                    {
                        "type": "partial_transcript",
                        "text": result.text,
                        "buffer_ms": buffer_ms,
                        "asr_latency_ms": result.latency_ms,
                        "asr_rtf": result.rtf,
                    }
                )

    async def _run_asr(self, snapshot: bytes, final: bool) -> AsrRunResult:
        if not snapshot:
            return AsrRunResult(text="", latency_ms=0, rtf=0.0)
        async with self.asr_lock:
            started_at = time.monotonic()
            try:
                text = await asyncio.to_thread(
                    self.asr.transcribe_pcm_s16le,
                    snapshot,
                    self.sample_rate,
                    self.channels,
                )
                latency_ms = int((time.monotonic() - started_at) * 1000)
                audio_seconds = len(snapshot) / self.bytes_per_second
                rtf = latency_ms / 1000.0 / audio_seconds if audio_seconds > 0 else 0.0
                return AsrRunResult(text=text, latency_ms=latency_ms, rtf=rtf)
            except Exception as exc:
                latency_ms = int((time.monotonic() - started_at) * 1000)
                phase = "final" if final else "partial"
                print(f"[{self.session_id}] {phase} ASR error: {exc}", flush=True)
                await self._send_json(
                    {
                        "type": "error",
                        "message": f"{phase} ASR failed: {exc}",
                    }
                )
                return AsrRunResult(text="", latency_ms=latency_ms, rtf=0.0)

    async def _send_json(self, payload: dict[str, Any]) -> None:
        payload.setdefault("timestamp", datetime.now(UTC).isoformat())
        try:
            await self.websocket.send(json.dumps(payload, ensure_ascii=False))
        except ConnectionClosed:
            pass

    def _validate_audio_metadata(
        self,
        sample_rate: int,
        channels: int,
        audio_format: str,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if channels != self.config.channels:
            raise ValueError(f"unsupported channels: {channels}")
        if audio_format not in SUPPORTED_FORMATS:
            raise ValueError(f"unsupported audio format: {audio_format}")

    def _audio_metrics_log(self) -> str:
        metrics = self.audio_metrics
        min_rms = metrics.min_rms_dbfs if metrics.frames else -120.0
        max_rms = metrics.max_rms_dbfs if metrics.frames else -120.0
        return (
            f"frames={metrics.frames} "
            f"rms={metrics.rms_dbfs:6.1f}dBFS "
            f"frame_rms={metrics.mean_frame_rms_dbfs:6.1f}dBFS "
            f"rms_min={min_rms:6.1f}dBFS "
            f"rms_max={max_rms:6.1f}dBFS "
            f"peak={metrics.peak_dbfs:6.1f}dBFS "
            f"clip={metrics.clip_ratio:.2%} "
            f"silence={metrics.silence_ratio:.0%} "
            f"low={metrics.low_level_ratio:.0%}"
        )


class AudioWebSocketServer:
    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self.asr = Qwen3AsrAdapter(config)

    async def run(self) -> None:
        print(
            f"Loading Qwen3-ASR model '{self.config.asr_model}' "
            f"on {self.config.device}...",
            flush=True,
        )
        await asyncio.to_thread(self.asr.warmup)
        print("Qwen3-ASR model ready.", flush=True)

        async with serve(self.handle_connection, self.config.host, self.config.port):
            print(
                f"Listening on ws://{self.config.host}:{self.config.port}/ws/audio",
                flush=True,
            )
            stop = asyncio.Future()
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, stop.set_result, None)
            await stop

    async def handle_connection(self, websocket: ServerConnection) -> None:
        if websocket.request.path != "/ws/audio":
            await websocket.close(code=1008, reason="Use /ws/audio")
            return

        session = AudioSession(
            websocket=websocket,
            session_id=uuid4().hex[:8],
            config=self.config,
            asr=self.asr,
            sample_rate=self.config.sample_rate,
            channels=self.config.channels,
            format=self.config.audio_format,
        )
        print(f"[{session.session_id}] connected", flush=True)
        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    await self._handle_audio_frame(session, message)
                else:
                    await self._handle_control_message(session, message)
        except ConnectionClosed:
            pass
        finally:
            await session.close()
            print(f"[{session.session_id}] disconnected", flush=True)

    async def _handle_audio_frame(self, session: AudioSession, frame: bytes) -> None:
        if not session.is_streaming:
            await session._send_json(
                {"type": "error", "message": "binary audio received before speech_start"}
            )
            return
        within_limit = session.append_audio(frame)
        if not within_limit:
            await session._send_json(
                {
                    "type": "error",
                    "message": "max segment duration exceeded; finalizing current segment",
                    "buffer_ms": session.buffer_ms,
                }
            )
            await session.finish_stream(reason="max_segment_exceeded")

    async def _handle_control_message(self, session: AudioSession, message: str) -> None:
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            await session._send_json({"type": "error", "message": "invalid JSON control message"})
            return

        message_type = payload.get("type")
        try:
            if message_type == "speech_start":
                await session.start_stream(payload)
            elif message_type == "speech_end":
                await session.finish_stream()
            else:
                await session._send_json(
                    {"type": "error", "message": f"unknown control type: {message_type}"}
                )
        except ValueError as exc:
            await session._send_json({"type": "error", "message": str(exc)})


def main() -> None:
    config = load_config()
    asyncio.run(AudioWebSocketServer(config).run())


def measure_pcm_s16le(frame: bytes) -> PcmFrameMetrics:
    if len(frame) % 2:
        frame = frame[:-1]
    values = array("h")
    values.frombytes(frame)
    if sys.byteorder != "little":
        values.byteswap()
    if not values:
        return PcmFrameMetrics(
            samples=0,
            rms_dbfs=-120.0,
            peak_dbfs=-120.0,
            clip_count=0,
            silence_samples=0,
            low_level_samples=0,
        )

    samples = [value / INT16_MAX for value in values]
    rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples))
    peak = max(abs(sample) for sample in samples)
    rms_dbfs = _amplitude_to_dbfs(rms)
    peak_dbfs = _amplitude_to_dbfs(peak)
    return PcmFrameMetrics(
        samples=len(samples),
        rms_dbfs=rms_dbfs,
        peak_dbfs=peak_dbfs,
        clip_count=sum(1 for value in values if value <= -32768 or value >= 32767),
        silence_samples=sum(1 for sample in samples if abs(sample) < _dbfs_to_amplitude(-60.0)),
        low_level_samples=sum(1 for sample in samples if abs(sample) < _dbfs_to_amplitude(-45.0)),
    )


def _amplitude_to_dbfs(amplitude: float) -> float:
    if amplitude <= 1e-8:
        return -120.0
    return 20.0 * math.log10(amplitude)


def _dbfs_to_amplitude(dbfs: float) -> float:
    return 10.0 ** (dbfs / 20.0)


def _db_to_power(dbfs: float) -> float:
    if dbfs <= -120.0:
        return 0.0
    return 10.0 ** (dbfs / 10.0)


if __name__ == "__main__":
    main()
