from __future__ import annotations

from array import array
from dataclasses import dataclass
import math
import sys


INT16_MAX = 32768.0


@dataclass(frozen=True)
class ProcessedFrame:
    pcm: bytes
    samples: tuple[float, ...]
    rms_db: float


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

    def process(self, pcm_s16le: bytes) -> ProcessedFrame:
        samples = _pcm_to_float_samples(pcm_s16le)
        samples = self._high_pass(samples)
        rms_db = _rms_db(samples)
        if self.enable_noise_reduction:
            samples = self._mild_noise_reduction(samples, rms_db)
            rms_db = _rms_db(samples)
        return ProcessedFrame(
            pcm=_float_samples_to_pcm(samples),
            samples=tuple(samples),
            rms_db=rms_db,
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
        if rms_db < self._noise_floor_db + 8.0:
            self._noise_floor_db = 0.98 * self._noise_floor_db + 0.02 * rms_db

        if rms_db < self._noise_floor_db + 6.0:
            return [sample * 0.65 for sample in samples]
        if rms_db < self._noise_floor_db + 10.0:
            return [sample * 0.85 for sample in samples]
        return samples


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


def _rms_db(samples: list[float]) -> float:
    if not samples:
        return -120.0
    rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples))
    if rms <= 1e-8:
        return -120.0
    return 20.0 * math.log10(rms)
