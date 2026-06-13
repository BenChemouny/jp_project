from __future__ import annotations

from array import array
from dataclasses import dataclass
import math
import sys


INT16_MAX = 32768.0


@dataclass(frozen=True)
class AudioMetrics:
    raw_rms_dbfs: float
    filtered_rms_dbfs: float
    output_rms_dbfs: float
    peak_dbfs: float
    clip_count: int
    noise_floor_dbfs: float
    nr_gain_db: float


@dataclass(frozen=True)
class ProcessedFrame:
    pcm: bytes
    samples: tuple[float, ...]
    rms_db: float
    vad_pcm: bytes
    vad_samples: tuple[float, ...]
    vad_rms_db: float
    metrics: AudioMetrics


class AudioPreprocessor:
    def __init__(
        self,
        sample_rate: int,
        high_pass_hz: float,
        enable_noise_reduction: bool,
    ) -> None:
        self.sample_rate = sample_rate
        self.high_pass_hz = high_pass_hz
        self.enable_noise_reduction = enable_noise_reduction
        self._prev_x = 0.0
        self._prev_y = 0.0
        self._noise_floor_db = -65.0
        self._noise_floor_observations = 0

    def process(self, pcm_s16le: bytes) -> ProcessedFrame:
        raw_samples = _pcm_to_float_samples(pcm_s16le)
        raw_rms_dbfs = _rms_db(raw_samples)
        filtered_samples = self._high_pass(raw_samples)
        filtered_rms_dbfs = _rms_db(filtered_samples)
        peak_dbfs = _peak_db(filtered_samples)
        clip_count = _clip_count(raw_samples)
        vad_samples = tuple(_clamp_float_samples(filtered_samples))
        vad_rms_dbfs = filtered_rms_dbfs
        samples = filtered_samples
        output_rms_dbfs = filtered_rms_dbfs
        nr_gain_db = 0.0
        if self.enable_noise_reduction:
            samples = self._mild_noise_reduction(samples, filtered_rms_dbfs)
            output_rms_dbfs = _rms_db(samples)
            nr_gain_db = output_rms_dbfs - filtered_rms_dbfs
        return ProcessedFrame(
            pcm=_float_samples_to_pcm(samples),
            samples=tuple(samples),
            rms_db=output_rms_dbfs,
            vad_pcm=_float_samples_to_pcm(list(vad_samples)),
            vad_samples=vad_samples,
            vad_rms_db=vad_rms_dbfs,
            metrics=AudioMetrics(
                raw_rms_dbfs=raw_rms_dbfs,
                filtered_rms_dbfs=filtered_rms_dbfs,
                output_rms_dbfs=output_rms_dbfs,
                peak_dbfs=peak_dbfs,
                clip_count=clip_count,
                noise_floor_dbfs=self._noise_floor_db,
                nr_gain_db=nr_gain_db,
            ),
        )

    def _high_pass(self, samples: list[float]) -> list[float]:
        if self.high_pass_hz <= 0:
            return samples

        rc = 1.0 / (2.0 * math.pi * self.high_pass_hz)
        dt = 1.0 / float(self.sample_rate)
        alpha = rc / (rc + dt)
        out: list[float] = []
        prev_x = self._prev_x
        prev_y = self._prev_y
        for x_value in samples:
            y_value = alpha * (prev_y + x_value - prev_x)
            out.append(y_value)
            prev_x = x_value
            prev_y = y_value
        self._prev_x = prev_x
        self._prev_y = prev_y
        return out

    def _mild_noise_reduction(self, samples: list[float], rms_db: float) -> list[float]:
        self._update_noise_floor(rms_db)
        margin_db = rms_db - self._noise_floor_db

        if margin_db < 3.0:
            return [sample * 0.45 for sample in samples]
        if margin_db < 8.0:
            return [sample * 0.65 for sample in samples]
        if margin_db < 12.0:
            return [sample * 0.85 for sample in samples]
        return samples

    def _update_noise_floor(self, rms_db: float) -> None:
        if rms_db <= -119.0:
            self._noise_floor_db = max(-120.0, 0.98 * self._noise_floor_db + 0.02 * rms_db)
            return

        if rms_db < self._noise_floor_db:
            alpha = 0.08
        elif rms_db < self._noise_floor_db + 25.0:
            alpha = 0.04 if self._noise_floor_observations < 50 else 0.004
        elif self._noise_floor_observations < 50 and rms_db < -12.0:
            alpha = 0.04
        else:
            return
        self._noise_floor_db = (1.0 - alpha) * self._noise_floor_db + alpha * rms_db
        self._noise_floor_observations += 1


def _pcm_to_float_samples(pcm_s16le: bytes) -> list[float]:
    if len(pcm_s16le) % 2:
        pcm_s16le = pcm_s16le[:-1]
    values = array("h")
    values.frombytes(pcm_s16le)
    if sys.byteorder != "little":
        values.byteswap()
    return [value / INT16_MAX for value in values]


def _float_samples_to_pcm(samples: list[float]) -> bytes:
    values = array(
        "h",
        (
            max(-32768, min(32767, int(round(sample * 32767.0))))
            for sample in samples
        ),
    )
    if sys.byteorder != "little":
        values.byteswap()
    return values.tobytes()


def _clamp_float_samples(samples: list[float]) -> list[float]:
    return [max(-1.0, min(1.0, sample)) for sample in samples]


def _rms_db(samples: list[float]) -> float:
    if not samples:
        return -120.0
    rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples))
    if rms <= 1e-8:
        return -120.0
    return 20.0 * math.log10(rms)


def _peak_db(samples: list[float]) -> float:
    if not samples:
        return -120.0
    peak = max(abs(sample) for sample in samples)
    if peak <= 1e-8:
        return -120.0
    return 20.0 * math.log10(peak)


def _clip_count(samples: list[float]) -> int:
    return sum(1 for sample in samples if sample <= -1.0 or sample >= 32767.0 / INT16_MAX)
