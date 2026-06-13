from __future__ import annotations

import asyncio
from collections import Counter, deque
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import math
from pathlib import Path
import signal
import time
from typing import Any, Callable
import wave

from client.audio_processing import AudioPreprocessor, ProcessedFrame
from client.config import ClientConfig, load_config
from client.japanese_text import analyze_japanese_text_dicts
from client.vad import VadBackend, create_vad


STATE_IDLE = "idle"
STATE_CALIBRATING = "calibrating"
STATE_MAYBE_SPEECH = "maybe_speech"
STATE_STREAMING = "streaming"
STATE_MAYBE_SILENCE = "maybe_silence"
ClientEventSink = Callable[[dict[str, Any]], None]


@dataclass
class FrameAnalysis:
    frame: ProcessedFrame
    vad_probability: float
    received_at: float
    vad_start_threshold: float
    vad_continue_threshold: float
    vad_noise_floor_dbfs: float
    vad_snr_db: float
    vad_start_positive: bool
    vad_continue_positive: bool
    vad_reason: str


@dataclass(frozen=True)
class VadDecision:
    is_start_positive: bool
    is_continue_positive: bool
    start_threshold: float
    continue_threshold: float
    noise_floor_dbfs: float
    snr_db: float
    reason: str


class AdaptiveVadGate:
    def __init__(self, config: ClientConfig) -> None:
        self.config = config
        self.noise_floor_dbfs = -65.0
        self.probability_floor = 0.02
        self.noise_floor_observations = 0
        floor_frames = max(10, config.vad_floor_window_ms // config.frame_ms)
        self.noise_floor_samples: deque[float] = deque(maxlen=floor_frames)
        self.probability_floor_samples: deque[float] = deque(maxlen=floor_frames)

    def calibrate(self, probability: float, rms_dbfs: float) -> VadDecision:
        self._learn_floor(probability, rms_dbfs, force=True)
        return VadDecision(
            is_start_positive=False,
            is_continue_positive=False,
            start_threshold=self.config.vad_start_threshold,
            continue_threshold=self.config.vad_continue_threshold,
            noise_floor_dbfs=self.noise_floor_dbfs,
            snr_db=rms_dbfs - self.noise_floor_dbfs,
            reason="calibrating",
        )

    def decide(
        self,
        probability: float,
        rms_dbfs: float,
        in_speech: bool,
    ) -> VadDecision:
        if not self.config.dynamic_vad:
            self._update_floors(
                probability=probability,
                rms_dbfs=rms_dbfs,
                in_speech=in_speech,
                continue_threshold=self.config.vad_continue_threshold,
            )
            return VadDecision(
                is_start_positive=probability >= self.config.vad_start_threshold,
                is_continue_positive=probability >= self.config.vad_continue_threshold,
                start_threshold=self.config.vad_start_threshold,
                continue_threshold=self.config.vad_continue_threshold,
                noise_floor_dbfs=self.noise_floor_dbfs,
                snr_db=rms_dbfs - self.noise_floor_dbfs,
                reason="model" if probability >= self.config.vad_continue_threshold else "none",
            )

        if not in_speech and probability < self.config.vad_start_threshold:
            self._update_floors(
                probability=probability,
                rms_dbfs=rms_dbfs,
                in_speech=False,
                continue_threshold=self.config.vad_start_threshold,
            )

        start_threshold, continue_threshold = self._thresholds(rms_dbfs)
        snr_db = rms_dbfs - self.noise_floor_dbfs

        margin_ok = snr_db >= self.config.vad_noise_margin_db
        confident_start = probability >= self.config.vad_start_threshold
        confident_continue = probability >= self.config.vad_continue_threshold
        is_start_positive = probability >= start_threshold and (margin_ok or confident_start)
        is_continue_positive = probability >= continue_threshold and (
            margin_ok or confident_continue
        )
        reason = "model" if is_start_positive or is_continue_positive else "none"
        if self.config.vad_energy_fallback:
            if (
                not in_speech
                and not is_start_positive
                and snr_db >= self.config.vad_energy_start_margin_db
            ):
                is_start_positive = True
                reason = "energy_start"
            if (
                in_speech
                and not is_continue_positive
                and snr_db >= self.config.vad_energy_continue_margin_db
            ):
                is_continue_positive = True
                reason = "energy_continue"

        self._update_floors(
            probability=probability,
            rms_dbfs=rms_dbfs,
            in_speech=in_speech or is_start_positive or is_continue_positive,
            continue_threshold=continue_threshold,
        )
        return VadDecision(
            is_start_positive=is_start_positive,
            is_continue_positive=is_continue_positive,
            start_threshold=start_threshold,
            continue_threshold=continue_threshold,
            noise_floor_dbfs=self.noise_floor_dbfs,
            snr_db=rms_dbfs - self.noise_floor_dbfs,
            reason=reason,
        )

    def _thresholds(self, rms_dbfs: float) -> tuple[float, float]:
        if not self.config.dynamic_vad:
            return self.config.vad_start_threshold, self.config.vad_continue_threshold

        span_db = max(
            1.0,
            self.config.vad_speech_margin_db - self.config.vad_noise_margin_db,
        )
        snr_ratio = min(
            1.0,
            max(0.0, (rms_dbfs - self.noise_floor_dbfs - self.config.vad_noise_margin_db) / span_db),
        )
        start = _lerp(
            self.config.vad_start_threshold,
            self.config.vad_min_start_threshold,
            snr_ratio,
        )
        continue_ = _lerp(
            self.config.vad_continue_threshold,
            self.config.vad_min_continue_threshold,
            snr_ratio,
        )
        start = max(start, self.probability_floor + 0.15)
        continue_ = max(continue_, self.probability_floor + 0.08)
        return (
            min(0.95, max(0.05, start)),
            min(0.90, max(0.03, continue_)),
        )

    def _update_floors(
        self,
        probability: float,
        rms_dbfs: float,
        in_speech: bool,
        continue_threshold: float,
    ) -> None:
        if rms_dbfs <= -119.0:
            self.noise_floor_dbfs = max(
                -120.0,
                0.98 * self.noise_floor_dbfs + 0.02 * rms_dbfs,
            )
            return

        if not in_speech and probability < continue_threshold:
            self._learn_floor(probability, rms_dbfs)
            if len(self.noise_floor_samples) >= 5:
                return
            if rms_dbfs < self.noise_floor_dbfs:
                alpha = 0.08
            elif rms_dbfs < self.noise_floor_dbfs + 25.0:
                alpha = 0.12 if self.noise_floor_observations < 50 else 0.01
            elif self.noise_floor_observations < 50 and rms_dbfs < -12.0:
                alpha = 0.12
            else:
                alpha = 0.0
            if alpha:
                self.noise_floor_dbfs = (1.0 - alpha) * self.noise_floor_dbfs + alpha * rms_dbfs
                self.noise_floor_observations += 1

            prob_alpha = 0.04 if probability > self.probability_floor else 0.08
            self.probability_floor = (
                (1.0 - prob_alpha) * self.probability_floor + prob_alpha * probability
            )
            self.probability_floor = min(0.75, max(0.0, self.probability_floor))

    def _learn_floor(self, probability: float, rms_dbfs: float, force: bool = False) -> None:
        if rms_dbfs <= -119.0:
            return
        if not force and probability >= self.config.vad_continue_threshold:
            return
        if rms_dbfs > -6.0:
            return
        self.noise_floor_samples.append(rms_dbfs)
        self.probability_floor_samples.append(probability)
        self.noise_floor_dbfs = _percentile(list(self.noise_floor_samples), 30.0)
        self.probability_floor = min(
            0.75,
            max(0.0, _percentile(list(self.probability_floor_samples), 70.0)),
        )


def _lerp(start: float, end: float, ratio: float) -> float:
    return start + (end - start) * ratio


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return -65.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * percentile / 100.0))
    return ordered[min(len(ordered) - 1, max(0, index))]


@dataclass
class ClientMetricsWindow:
    frames: int = 0
    raw_power_sum: float = 0.0
    filtered_power_sum: float = 0.0
    output_power_sum: float = 0.0
    nr_gain_sum: float = 0.0
    vad_probability_sum: float = 0.0
    vad_start_threshold_sum: float = 0.0
    vad_continue_threshold_sum: float = 0.0
    vad_snr_sum: float = 0.0
    vad_positive_frames: int = 0
    vad_reason_counts: Counter[str] | None = None
    latest_vad_reason: str = "none"
    vad_gain_sum: float = 0.0
    clip_count: int = 0
    peak_dbfs: float = -120.0
    min_output_rms_dbfs: float = 120.0
    max_output_rms_dbfs: float = -120.0
    latest_noise_floor_dbfs: float = -120.0

    def add(self, analysis: FrameAnalysis) -> None:
        metrics = analysis.frame.metrics
        self.frames += 1
        self.raw_power_sum += _db_to_power(metrics.raw_rms_dbfs)
        self.filtered_power_sum += _db_to_power(metrics.filtered_rms_dbfs)
        self.output_power_sum += _db_to_power(metrics.output_rms_dbfs)
        self.nr_gain_sum += metrics.nr_gain_db
        self.vad_probability_sum += analysis.vad_probability
        self.vad_start_threshold_sum += analysis.vad_start_threshold
        self.vad_continue_threshold_sum += analysis.vad_continue_threshold
        self.vad_snr_sum += analysis.vad_snr_db
        self.vad_gain_sum += metrics.vad_gain_db
        if analysis.vad_continue_positive:
            self.vad_positive_frames += 1
        if self.vad_reason_counts is None:
            self.vad_reason_counts = Counter()
        self.vad_reason_counts[analysis.vad_reason] += 1
        self.latest_vad_reason = analysis.vad_reason
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
        self.latest_noise_floor_dbfs = analysis.vad_noise_floor_dbfs

    def reset(self) -> None:
        self.frames = 0
        self.raw_power_sum = 0.0
        self.filtered_power_sum = 0.0
        self.output_power_sum = 0.0
        self.nr_gain_sum = 0.0
        self.vad_probability_sum = 0.0
        self.vad_start_threshold_sum = 0.0
        self.vad_continue_threshold_sum = 0.0
        self.vad_snr_sum = 0.0
        self.vad_positive_frames = 0
        self.vad_reason_counts = None
        self.latest_vad_reason = "none"
        self.vad_gain_sum = 0.0
        self.clip_count = 0
        self.peak_dbfs = -120.0
        self.min_output_rms_dbfs = 120.0
        self.max_output_rms_dbfs = -120.0
        self.latest_noise_floor_dbfs = -120.0

    def summary(self) -> dict[str, Any]:
        if self.frames <= 0:
            return {
                "frames": 0,
                "raw_rms_dbfs": -120.0,
                "filtered_rms_dbfs": -120.0,
                "output_rms_dbfs": -120.0,
                "nr_gain_db": 0.0,
                "vad_probability": 0.0,
                "vad_start_threshold": 0.0,
                "vad_continue_threshold": 0.0,
                "vad_snr_db": 0.0,
                "vad_reason": "none",
                "vad_gain_db": 0.0,
                "vad_positive_ratio": 0.0,
                "clip_count": 0,
                "peak_dbfs": -120.0,
                "min_output_rms_dbfs": -120.0,
                "max_output_rms_dbfs": -120.0,
                "noise_floor_dbfs": -120.0,
            }
        reason = self.latest_vad_reason
        if self.vad_reason_counts:
            reason = self.vad_reason_counts.most_common(1)[0][0]
        return {
            "frames": self.frames,
            "raw_rms_dbfs": _power_to_db(self.raw_power_sum / self.frames),
            "filtered_rms_dbfs": _power_to_db(self.filtered_power_sum / self.frames),
            "output_rms_dbfs": _power_to_db(self.output_power_sum / self.frames),
            "nr_gain_db": self.nr_gain_sum / self.frames,
            "vad_probability": self.vad_probability_sum / self.frames,
            "vad_start_threshold": self.vad_start_threshold_sum / self.frames,
            "vad_continue_threshold": self.vad_continue_threshold_sum / self.frames,
            "vad_snr_db": self.vad_snr_sum / self.frames,
            "vad_reason": reason,
            "vad_gain_db": self.vad_gain_sum / self.frames,
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
    def __init__(
        self,
        config: ClientConfig,
        event_sink: ClientEventSink | None = None,
    ) -> None:
        self.config = config
        self.event_sink = event_sink
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
                    self._publish({"type": "connection", "status": "connected"})
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
                self._publish({"type": "connection", "status": "disconnected"})

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
                self._handle_text_message(message)

    def _handle_text_message(self, message: str) -> None:
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            return
        message_type = payload.get("type")
        if message_type == "partial_transcript":
            text = str(payload.get("text", ""))
            self._publish(
                {
                    "type": "transcript",
                    "phase": "partial",
                    "text": text,
                    "tokens": _analyze_transcript(text),
                }
            )
        elif message_type == "final_transcript":
            text = str(payload.get("text", ""))
            self._publish(
                {
                    "type": "transcript",
                    "phase": "final",
                    "text": text,
                    "tokens": _analyze_transcript(text),
                }
            )

    def _publish(self, event: dict[str, Any]) -> None:
        if self.event_sink is not None:
            self.event_sink(event)


def _analyze_transcript(text: str) -> list[dict[str, str]]:
    try:
        return analyze_japanese_text_dicts(text)
    except RuntimeError as exc:
        print(f"[text] {exc}", flush=True)
        return []


class DebugWavRecorder:
    def __init__(self, config: ClientConfig) -> None:
        self.config = config
        self.enabled = bool(config.vad_debug_wav_dir)
        max_frames = max(1, int(config.vad_debug_wav_seconds * 1000 / config.frame_ms))
        self.raw_frames: deque[bytes] = deque(maxlen=max_frames)
        self.vad_frames: deque[bytes] = deque(maxlen=max_frames)
        self.output_frames: deque[bytes] = deque(maxlen=max_frames)

    def add(self, raw_pcm: bytes, processed: ProcessedFrame) -> None:
        if not self.enabled:
            return
        self.raw_frames.append(raw_pcm)
        self.vad_frames.append(processed.vad_pcm)
        self.output_frames.append(processed.pcm)

    def flush(self, label: str) -> None:
        if not self.enabled or not self.raw_frames:
            return
        output_dir = Path(str(self.config.vad_debug_wav_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
        self._write(output_dir / f"{stamp}_{label}_raw.wav", self.raw_frames)
        self._write(output_dir / f"{stamp}_{label}_vad.wav", self.vad_frames)
        self._write(output_dir / f"{stamp}_{label}_out.wav", self.output_frames)

    def _write(self, path: Path, frames: deque[bytes]) -> None:
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.config.sample_rate)
            wav_file.writeframes(b"".join(frames))


class VoiceStreamer:
    def __init__(
        self,
        config: ClientConfig,
        websocket: WebSocketClient,
        preprocessor: AudioPreprocessor,
        vad: VadBackend,
        event_sink: ClientEventSink | None = None,
    ) -> None:
        self.config = config
        self.websocket = websocket
        self.preprocessor = preprocessor
        self.vad = vad
        self.event_sink = event_sink
        self.state = STATE_CALIBRATING if config.calibration_ms > 0 else STATE_IDLE
        self.calibration_remaining_ms = max(0, config.calibration_ms)
        self.pre_roll: deque[bytes] = deque(maxlen=max(1, config.pre_roll_ms // config.frame_ms))
        self.positive_ms = 0
        self.segment_audio_ms = 0
        self.last_voice_at: float | None = None
        self.stream_open = False
        self.stream_generation = 0
        self.last_log_at = 0.0
        self.metrics_window = ClientMetricsWindow()
        self.vad_gate = AdaptiveVadGate(config)
        self.energy_continue_ms = 0
        self.idle_ms = 0
        self.debug_wav = DebugWavRecorder(config)

    async def handle_raw_frame(self, pcm_s16le: bytes) -> None:
        processed = self.preprocessor.process(pcm_s16le)
        self.debug_wav.add(pcm_s16le, processed)
        vad_probability = self.vad.probability(
            processed.vad_pcm,
            processed.vad_samples,
            processed.vad_rms_db,
        )
        if self.state == STATE_CALIBRATING:
            vad_decision = self.vad_gate.calibrate(vad_probability, processed.vad_rms_db)
        else:
            vad_decision = self.vad_gate.decide(
                vad_probability,
                processed.vad_rms_db,
                self.state in {STATE_STREAMING, STATE_MAYBE_SILENCE},
            )
        analysis = FrameAnalysis(
            frame=processed,
            vad_probability=vad_probability,
            received_at=time.monotonic(),
            vad_start_threshold=vad_decision.start_threshold,
            vad_continue_threshold=vad_decision.continue_threshold,
            vad_noise_floor_dbfs=vad_decision.noise_floor_dbfs,
            vad_snr_db=vad_decision.snr_db,
            vad_start_positive=vad_decision.is_start_positive,
            vad_continue_positive=vad_decision.is_continue_positive,
            vad_reason=vad_decision.reason,
        )
        if self.state == STATE_CALIBRATING:
            self.calibration_remaining_ms -= self.config.frame_ms
            if self.calibration_remaining_ms <= 0:
                self.state = STATE_IDLE
                self.vad.reset()
                print(
                    f"[vad] calibration complete floor={vad_decision.noise_floor_dbfs:.1f}dBFS",
                    flush=True,
                )
            self._log_status(analysis)
            return

        await self._advance_state(analysis, vad_decision)
        self._log_status(analysis)

    async def finish_active_stream(self) -> None:
        if self.stream_open:
            await self.websocket.send_json({"type": "speech_end"})
        self.debug_wav.flush("segment")
        self.vad.reset()
        self._reset_segment()

    def start_calibration(self) -> None:
        self.vad.reset()
        self._reset_segment()
        self.state = STATE_CALIBRATING if self.config.calibration_ms > 0 else STATE_IDLE
        self.calibration_remaining_ms = max(0, self.config.calibration_ms)

    async def _advance_state(self, analysis: FrameAnalysis, vad_decision: VadDecision) -> None:
        is_start_positive = vad_decision.is_start_positive
        is_continue_positive = vad_decision.is_continue_positive
        if vad_decision.reason == "energy_continue":
            self.energy_continue_ms += self.config.frame_ms
            if self.energy_continue_ms > self.config.vad_energy_max_continue_ms:
                is_continue_positive = False
                analysis.vad_continue_positive = False
                analysis.vad_reason = "energy_limited"
        else:
            self.energy_continue_ms = 0

        if self.state in {STATE_IDLE, STATE_MAYBE_SPEECH}:
            self.pre_roll.append(analysis.frame.pcm)

        if self.state == STATE_IDLE:
            if is_start_positive:
                self.state = STATE_MAYBE_SPEECH
                self.positive_ms = self.config.frame_ms
                self.idle_ms = 0
            else:
                self.idle_ms += self.config.frame_ms
                if self.config.vad_idle_reset_ms > 0 and self.idle_ms >= self.config.vad_idle_reset_ms:
                    self.vad.reset()
                    self.idle_ms = 0
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
            if self.config.max_segment_ms > 0 and self.segment_audio_ms >= self.config.max_segment_ms:
                if self.stream_open:
                    await self.websocket.send_json({"type": "speech_end"})
                print(
                    f"[segment] max duration reached duration_ms={self.segment_audio_ms}",
                    flush=True,
                )
                self.debug_wav.flush("max_segment")
                self.vad.reset()
                self._reset_segment()
                return
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
                self.debug_wav.flush("segment")
                self.vad.reset()
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
        self.energy_continue_ms = 0
        self.idle_ms = 0
        self.pre_roll.clear()

    def _log_status(self, analysis: FrameAnalysis) -> None:
        self.metrics_window.add(analysis)
        now = time.monotonic()
        if now - self.last_log_at < 1.0:
            return
        self.last_log_at = now
        metrics = self.metrics_window.summary()
        self._publish_metrics(metrics)
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
            f"snr={metrics['vad_snr_db']:5.1f}dB "
            f"nr={metrics['nr_gain_db']:5.1f}dB "
            f"vad_gain={metrics['vad_gain_db']:5.1f}dB "
            f"vad={metrics['vad_probability']:.2f} "
            f"thr={metrics['vad_start_threshold']:.2f}/{metrics['vad_continue_threshold']:.2f} "
            f"vad_pos={metrics['vad_positive_ratio']:.0%} "
            f"reason={metrics['vad_reason']} "
            f"state={self.state} "
            f"streaming={self.stream_open} "
            f"segment_ms={self.segment_audio_ms} "
            f"ws={'connected' if self.websocket.connected.is_set() else 'disconnected'}",
            flush=True,
        )
        self.metrics_window.reset()

    def _publish_metrics(self, metrics: dict[str, Any]) -> None:
        if self.event_sink is None:
            return
        self.event_sink(
            {
                "type": "metrics",
                "metrics": metrics,
                "state": self.state,
                "streaming": self.stream_open,
                "segment_ms": self.segment_audio_ms,
                "ws": "connected" if self.websocket.connected.is_set() else "disconnected",
            }
        )


async def run_client(
    config: ClientConfig,
    event_sink: ClientEventSink | None = None,
    install_signal_handlers: bool = True,
    should_stop: Callable[[], bool] | None = None,
    is_paused: Callable[[], bool] | None = None,
) -> None:
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

    websocket = WebSocketClient(config, event_sink=event_sink)
    streamer = VoiceStreamer(
        config,
        websocket,
        preprocessor,
        vad_choice.backend,
        event_sink=event_sink,
    )
    capture = AudioCapture(config, audio_queue)
    ws_task = asyncio.create_task(websocket.run())

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()
    if install_signal_handlers:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop_event.set)

    capture.start(loop)
    print(
        f"[audio] capturing {config.sample_rate}Hz mono pcm_s16le "
        f"frame_ms={config.frame_ms}",
        flush=True,
    )
    try:
        was_paused = False
        while not stop_event.is_set() and not (should_stop and should_stop()):
            paused = bool(is_paused and is_paused())
            if paused:
                if not was_paused:
                    await streamer.finish_active_stream()
                    print("[audio] paused", flush=True)
                    if event_sink is not None:
                        event_sink({"type": "recording", "paused": True})
                was_paused = True
                _drain_audio_queue(audio_queue)
                await asyncio.sleep(0.05)
                continue
            if was_paused:
                streamer.start_calibration()
                print("[audio] resumed", flush=True)
                if event_sink is not None:
                    event_sink({"type": "recording", "paused": False})
                was_paused = False

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


def _drain_audio_queue(audio_queue: asyncio.Queue[bytes]) -> None:
    while True:
        try:
            audio_queue.get_nowait()
        except asyncio.QueueEmpty:
            return


def main() -> None:
    asyncio.run(run_client(load_config()))


if __name__ == "__main__":
    main()
