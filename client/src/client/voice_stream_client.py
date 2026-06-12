from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import math
import signal
import time
from typing import Any

from client.audio_processing import AudioPreprocessor, ProcessedFrame
from client.config import ClientConfig, load_config
from client.vad import VadBackend, create_vad


STATE_IDLE = "idle"
STATE_MAYBE_SPEECH = "maybe_speech"
STATE_STREAMING = "streaming"
STATE_MAYBE_SILENCE = "maybe_silence"


@dataclass(frozen=True)
class FrameAnalysis:
    frame: ProcessedFrame
    vad_probability: float
    received_at: float


@dataclass
class ClientMetricsWindow:
    frames: int = 0
    raw_power_sum: float = 0.0
    filtered_power_sum: float = 0.0
    output_power_sum: float = 0.0
    nr_gain_sum: float = 0.0
    vad_probability_sum: float = 0.0
    vad_positive_frames: int = 0
    clip_count: int = 0
    peak_dbfs: float = -120.0
    min_output_rms_dbfs: float = 120.0
    max_output_rms_dbfs: float = -120.0
    latest_noise_floor_dbfs: float = -120.0

    def add(self, analysis: FrameAnalysis, vad_threshold: float) -> None:
        metrics = analysis.frame.metrics
        self.frames += 1
        self.raw_power_sum += _db_to_power(metrics.raw_rms_dbfs)
        self.filtered_power_sum += _db_to_power(metrics.filtered_rms_dbfs)
        self.output_power_sum += _db_to_power(metrics.output_rms_dbfs)
        self.nr_gain_sum += metrics.nr_gain_db
        self.vad_probability_sum += analysis.vad_probability
        if analysis.vad_probability >= vad_threshold:
            self.vad_positive_frames += 1
        self.clip_count += metrics.clip_count
        self.peak_dbfs = max(self.peak_dbfs, metrics.peak_dbfs)
        self.min_output_rms_dbfs = min(
            self.min_output_rms_dbfs,
            metrics.output_rms_dbfs,
        )
        self.max_output_rms_dbfs = max(
            self.max_output_rms_dbfs,
            metrics.output_rms_dbfs,
        )
        self.latest_noise_floor_dbfs = metrics.noise_floor_dbfs

    def reset(self) -> None:
        self.frames = 0
        self.raw_power_sum = 0.0
        self.filtered_power_sum = 0.0
        self.output_power_sum = 0.0
        self.nr_gain_sum = 0.0
        self.vad_probability_sum = 0.0
        self.vad_positive_frames = 0
        self.clip_count = 0
        self.peak_dbfs = -120.0
        self.min_output_rms_dbfs = 120.0
        self.max_output_rms_dbfs = -120.0
        self.latest_noise_floor_dbfs = -120.0

    def summary(self) -> dict[str, float | int]:
        if self.frames <= 0:
            return {
                "frames": 0,
                "raw_rms_dbfs": -120.0,
                "filtered_rms_dbfs": -120.0,
                "output_rms_dbfs": -120.0,
                "nr_gain_db": 0.0,
                "vad_probability": 0.0,
                "vad_positive_ratio": 0.0,
                "clip_count": 0,
                "peak_dbfs": -120.0,
                "min_output_rms_dbfs": -120.0,
                "max_output_rms_dbfs": -120.0,
                "noise_floor_dbfs": -120.0,
            }
        return {
            "frames": self.frames,
            "raw_rms_dbfs": _power_to_db(self.raw_power_sum / self.frames),
            "filtered_rms_dbfs": _power_to_db(self.filtered_power_sum / self.frames),
            "output_rms_dbfs": _power_to_db(self.output_power_sum / self.frames),
            "nr_gain_db": self.nr_gain_sum / self.frames,
            "vad_probability": self.vad_probability_sum / self.frames,
            "vad_positive_ratio": self.vad_positive_frames / self.frames,
            "clip_count": self.clip_count,
            "peak_dbfs": self.peak_dbfs,
            "min_output_rms_dbfs": self.min_output_rms_dbfs,
            "max_output_rms_dbfs": self.max_output_rms_dbfs,
            "noise_floor_dbfs": self.latest_noise_floor_dbfs,
        }


def _db_to_power(dbfs: float) -> float:
    if dbfs <= -120.0:
        return 0.0
    return 10.0 ** (dbfs / 10.0)


def _power_to_db(power: float) -> float:
    if power <= 1e-16:
        return -120.0
    return 10.0 * math.log10(power)


class AudioCapture:
    def __init__(self, config: ClientConfig, queue: asyncio.Queue[bytes]) -> None:
        self.config = config
        self.queue = queue
        self.stream = None

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        import sounddevice as sd

        def callback(indata: bytes, frames: int, time_info: Any, status: Any) -> None:
            if status:
                print(f"[audio] {status}", flush=True)
            payload = bytes(indata)
            loop.call_soon_threadsafe(self._enqueue, payload)

        self.stream = sd.RawInputStream(
            samplerate=self.config.sample_rate,
            blocksize=self.config.frame_samples,
            channels=1,
            dtype="int16",
            device=self.config.input_device,
            callback=callback,
        )
        self.stream.start()

    def stop(self) -> None:
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def _enqueue(self, payload: bytes) -> None:
        try:
            self.queue.put_nowait(payload)
        except asyncio.QueueFull:
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                self.queue.put_nowait(payload)
            except asyncio.QueueFull:
                pass


class WebSocketClient:
    def __init__(self, config: ClientConfig) -> None:
        self.config = config
        self.websocket = None
        self.connected = asyncio.Event()
        self.generation = 0
        self._stop = asyncio.Event()
        self._send_lock = asyncio.Lock()

    async def run(self) -> None:
        delay = self.config.reconnect_initial_delay_s
        while not self._stop.is_set():
            try:
                from websockets.asyncio.client import connect

                print(f"[ws] connecting {self.config.server_ws_url}", flush=True)
                async with connect(self.config.server_ws_url) as websocket:
                    self.websocket = websocket
                    self.generation += 1
                    self.connected.set()
                    delay = self.config.reconnect_initial_delay_s
                    print("[ws] connected", flush=True)
                    await self._receive_loop(websocket)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(f"[ws] disconnected: {exc}", flush=True)
            finally:
                self.websocket = None
                self.connected.clear()

            if not self._stop.is_set():
                await asyncio.sleep(delay)
                delay = min(delay * 2.0, self.config.reconnect_max_delay_s)

    async def stop(self) -> None:
        self._stop.set()
        if self.websocket is not None:
            await self.websocket.close()

    async def send_json(self, payload: dict[str, Any]) -> bool:
        payload.setdefault("timestamp", datetime.now(UTC).isoformat())
        return await self._send(json.dumps(payload, ensure_ascii=False))

    async def send_bytes(self, payload: bytes) -> bool:
        return await self._send(payload)

    async def _send(self, payload: str | bytes) -> bool:
        if self.websocket is None or not self.connected.is_set():
            return False
        async with self._send_lock:
            try:
                await self.websocket.send(payload)
                return True
            except Exception as exc:
                print(f"[ws] send failed: {exc}", flush=True)
                self.connected.clear()
                return False

    async def _receive_loop(self, websocket: Any) -> None:
        async for message in websocket:
            if isinstance(message, bytes):
                print(f"[ws] binary response {len(message)} bytes", flush=True)
            else:
                print(f"[ws] {message}", flush=True)


class VoiceStreamer:
    def __init__(
        self,
        config: ClientConfig,
        websocket: WebSocketClient,
        preprocessor: AudioPreprocessor,
        vad: VadBackend,
    ) -> None:
        self.config = config
        self.websocket = websocket
        self.preprocessor = preprocessor
        self.vad = vad
        self.state = STATE_IDLE
        self.pre_roll: deque[bytes] = deque(maxlen=max(1, config.pre_roll_ms // config.frame_ms))
        self.positive_ms = 0
        self.segment_audio_ms = 0
        self.last_voice_at: float | None = None
        self.stream_open = False
        self.stream_generation = 0
        self.last_log_at = 0.0
        self.metrics_window = ClientMetricsWindow()

    async def handle_raw_frame(self, pcm_s16le: bytes) -> None:
        processed = self.preprocessor.process(pcm_s16le)
        vad_probability = self.vad.probability(
            processed.pcm,
            processed.samples,
            processed.rms_db,
        )
        analysis = FrameAnalysis(
            frame=processed,
            vad_probability=vad_probability,
            received_at=time.monotonic(),
        )
        await self._advance_state(analysis)
        self._log_status(analysis)

    async def finish_active_stream(self) -> None:
        if self.stream_open:
            await self.websocket.send_json({"type": "speech_end"})
        self._reset_segment()

    async def _advance_state(self, analysis: FrameAnalysis) -> None:
        is_start_positive = analysis.vad_probability >= self.config.vad_start_threshold
        is_continue_positive = (
            analysis.vad_probability >= self.config.vad_continue_threshold
        )

        if self.state in {STATE_IDLE, STATE_MAYBE_SPEECH}:
            self.pre_roll.append(analysis.frame.pcm)

        if self.state == STATE_IDLE:
            if is_start_positive:
                self.state = STATE_MAYBE_SPEECH
                self.positive_ms = self.config.frame_ms
            return

        if self.state == STATE_MAYBE_SPEECH:
            if is_start_positive:
                self.positive_ms += self.config.frame_ms
                if self.positive_ms >= self.config.min_speech_ms:
                    await self._start_streaming(analysis)
            else:
                self.state = STATE_IDLE
                self.positive_ms = 0
            return

        if self.state in {STATE_STREAMING, STATE_MAYBE_SILENCE}:
            if self.websocket.connected.is_set() and not self.stream_open:
                await self._open_stream()

            if self.stream_open:
                sent = await self.websocket.send_bytes(analysis.frame.pcm)
                if not sent:
                    self.stream_open = False

            self.segment_audio_ms += self.config.frame_ms
            if is_continue_positive:
                self.state = STATE_STREAMING
                self.last_voice_at = analysis.received_at
                return

            if self.state == STATE_STREAMING:
                self.state = STATE_MAYBE_SILENCE
                if self.last_voice_at is None:
                    self.last_voice_at = analysis.received_at

            silence_ms = int((analysis.received_at - (self.last_voice_at or analysis.received_at)) * 1000)
            if silence_ms >= self.config.hangover_ms:
                if self.stream_open:
                    await self.websocket.send_json({"type": "speech_end"})
                print(
                    f"[segment] ended duration_ms={self.segment_audio_ms} "
                    f"sent={self.stream_open}",
                    flush=True,
                )
                self._reset_segment()

    async def _start_streaming(self, analysis: FrameAnalysis) -> None:
        self.state = STATE_STREAMING
        self.last_voice_at = analysis.received_at
        self.segment_audio_ms = len(self.pre_roll) * self.config.frame_ms
        await self._open_stream()
        if self.stream_open:
            for frame in self.pre_roll:
                sent = await self.websocket.send_bytes(frame)
                if not sent:
                    self.stream_open = False
                    break
        print(
            f"[segment] started pre_roll_ms={self.segment_audio_ms} "
            f"sent={self.stream_open}",
            flush=True,
        )

    async def _open_stream(self) -> None:
        if self.stream_open and self.stream_generation == self.websocket.generation:
            return
        if not self.websocket.connected.is_set():
            self.stream_open = False
            return

        ok = await self.websocket.send_json(
            {
                "type": "speech_start",
                "sample_rate": self.config.sample_rate,
                "channels": 1,
                "format": "pcm_s16le",
                "frame_ms": self.config.frame_ms,
            }
        )
        self.stream_open = ok
        if ok:
            self.stream_generation = self.websocket.generation

    def _reset_segment(self) -> None:
        self.state = STATE_IDLE
        self.positive_ms = 0
        self.segment_audio_ms = 0
        self.last_voice_at = None
        self.stream_open = False
        self.stream_generation = 0
        self.pre_roll.clear()

    def _log_status(self, analysis: FrameAnalysis) -> None:
        self.metrics_window.add(analysis, self.config.vad_continue_threshold)
        now = time.monotonic()
        if now - self.last_log_at < 1.0:
            return
        self.last_log_at = now
        metrics = self.metrics_window.summary()
        print(
            f"[audio] frames={metrics['frames']} "
            f"raw={metrics['raw_rms_dbfs']:6.1f}dBFS "
            f"filtered={metrics['filtered_rms_dbfs']:6.1f}dBFS "
            f"out={metrics['output_rms_dbfs']:6.1f}dBFS "
            f"out_min={metrics['min_output_rms_dbfs']:6.1f}dBFS "
            f"out_max={metrics['max_output_rms_dbfs']:6.1f}dBFS "
            f"peak={metrics['peak_dbfs']:6.1f}dBFS "
            f"clip={metrics['clip_count']} "
            f"floor={metrics['noise_floor_dbfs']:6.1f}dBFS "
            f"nr={metrics['nr_gain_db']:5.1f}dB "
            f"vad={metrics['vad_probability']:.2f} "
            f"vad_pos={metrics['vad_positive_ratio']:.0%} "
            f"state={self.state} "
            f"streaming={self.stream_open} "
            f"segment_ms={self.segment_audio_ms} "
            f"ws={'connected' if self.websocket.connected.is_set() else 'disconnected'}",
            flush=True,
        )
        self.metrics_window.reset()


async def run_client(config: ClientConfig) -> None:
    audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=config.audio_queue_size)
    preprocessor = AudioPreprocessor(
        sample_rate=config.sample_rate,
        high_pass_hz=config.high_pass_hz,
        enable_noise_reduction=config.enable_noise_reduction,
    )
    vad_choice = create_vad(config)
    if vad_choice.warning:
        print(f"[vad] {vad_choice.warning}", flush=True)
    print(f"[vad] backend={vad_choice.backend.name}", flush=True)

    websocket = WebSocketClient(config)
    streamer = VoiceStreamer(config, websocket, preprocessor, vad_choice.backend)
    capture = AudioCapture(config, audio_queue)
    ws_task = asyncio.create_task(websocket.run())

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)

    capture.start(loop)
    print(
        f"[audio] capturing {config.sample_rate}Hz mono pcm_s16le "
        f"frame_ms={config.frame_ms}",
        flush=True,
    )
    try:
        while not stop_event.is_set():
            try:
                frame = await asyncio.wait_for(audio_queue.get(), timeout=0.2)
            except asyncio.TimeoutError:
                continue
            await streamer.handle_raw_frame(frame)
    finally:
        capture.stop()
        await streamer.finish_active_stream()
        await websocket.stop()
        ws_task.cancel()
        try:
            await ws_task
        except asyncio.CancelledError:
            pass
        print("[client] stopped", flush=True)


def main() -> None:
    asyncio.run(run_client(load_config()))


if __name__ == "__main__":
    main()
