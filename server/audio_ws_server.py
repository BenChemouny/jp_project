from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
import signal
import time
from typing import Any
from uuid import uuid4

from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from server.config import ServerConfig, load_config
from server.qwen3_asr_adapter import Qwen3AsrAdapter


SUPPORTED_FORMATS = {"pcm_s16le"}


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
        text = await self._run_asr(snapshot, final=True)
        if text:
            print(f"[final {buffer_ms}ms] {text}", flush=True)
        else:
            print(f"[final {buffer_ms}ms] <empty>", flush=True)
        await self._send_json(
            {
                "type": "final_transcript",
                "text": text,
                "buffer_ms": buffer_ms,
                "reason": reason,
            }
        )
        self.audio_buffer.clear()
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
            text = await self._run_asr(snapshot, final=False)
            self.last_asr_at = time.monotonic()
            if not self.is_streaming:
                break
            if not text or text == self.last_partial_text:
                continue

            self.last_partial_text = text
            print(f"[partial {buffer_ms}ms] {text}", flush=True)
            if self.config.send_partials_to_client:
                await self._send_json(
                    {
                        "type": "partial_transcript",
                        "text": text,
                        "buffer_ms": buffer_ms,
                    }
                )

    async def _run_asr(self, snapshot: bytes, final: bool) -> str:
        if not snapshot:
            return ""
        async with self.asr_lock:
            try:
                return await asyncio.to_thread(
                    self.asr.transcribe_pcm_s16le,
                    snapshot,
                    self.sample_rate,
                    self.channels,
                )
            except Exception as exc:
                phase = "final" if final else "partial"
                print(f"[{self.session_id}] {phase} ASR error: {exc}", flush=True)
                await self._send_json(
                    {
                        "type": "error",
                        "message": f"{phase} ASR failed: {exc}",
                    }
                )
                return ""

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


class AudioWebSocketServer:
    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self.asr = Qwen3AsrAdapter(config)

    async def run(self) -> None:
        print(
            f"Loading Qwen3-ASR model '{self.config.qwen3_asr_model_path}' "
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


if __name__ == "__main__":
    main()
