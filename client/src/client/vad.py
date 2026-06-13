from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from client.config import ClientConfig


class VadBackend:
    name = "base"

    def probability(
        self,
        pcm_s16le: bytes,
        samples: Sequence[float],
        rms_db: float,
    ) -> float:
        raise NotImplementedError


class EnergyVad(VadBackend):
    name = "energy"

    def __init__(self) -> None:
        self.noise_floor_db = -65.0

    def probability(
        self,
        pcm_s16le: bytes,
        samples: Sequence[float],
        rms_db: float,
    ) -> float:
        if rms_db < self.noise_floor_db + 6.0:
            self.noise_floor_db = 0.995 * self.noise_floor_db + 0.005 * rms_db
        margin_db = rms_db - self.noise_floor_db
        return 1.0 / (1.0 + math.exp(-(margin_db - 14.0) / 4.0))


class WebRtcVad(VadBackend):
    name = "webrtc"

    def __init__(self, sample_rate: int, frame_ms: int, aggressiveness: int) -> None:
        import webrtcvad

        if frame_ms not in {10, 20, 30}:
            raise ValueError("WebRTC VAD requires frame_ms to be 10, 20, or 30")
        self.sample_rate = sample_rate
        self.vad = webrtcvad.Vad(aggressiveness)

    def probability(
        self,
        pcm_s16le: bytes,
        samples: Sequence[float],
        rms_db: float,
    ) -> float:
        return 1.0 if self.vad.is_speech(pcm_s16le, self.sample_rate) else 0.0


class SileroOnnxVad(VadBackend):
    name = "silero"

    def __init__(self, model_path: str, sample_rate: int) -> None:
        import numpy as np
        import onnxruntime as ort

        if sample_rate not in {8000, 16000}:
            raise ValueError("Silero VAD ONNX supports 8000 Hz or 16000 Hz")
        self.sample_rate = sample_rate
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = [item.name for item in self.session.get_inputs()]
        self.output_names = [item.name for item in self.session.get_outputs()]
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.h = np.zeros((2, 1, 64), dtype=np.float32)
        self.c = np.zeros((2, 1, 64), dtype=np.float32)

    def probability(
        self,
        pcm_s16le: bytes,
        samples: Sequence[float],
        rms_db: float,
    ) -> float:
        import numpy as np

        vad_samples = self._prepare_samples(samples)
        inputs: dict[str, np.ndarray] = {}
        for name in self.input_names:
            if name in {"input", "x"}:
                inputs[name] = vad_samples.reshape(1, -1).astype(np.float32)
            elif name in {"sr", "sample_rate", "sampling_rate"}:
                inputs[name] = np.array(self.sample_rate, dtype=np.int64)
            elif name == "state":
                inputs[name] = self.state
            elif name == "h":
                inputs[name] = self.h
            elif name == "c":
                inputs[name] = self.c

        outputs = self.session.run(None, inputs)
        for name, value in zip(self.output_names, outputs, strict=False):
            if name in {"state", "stateN"}:
                self.state = value
            elif name in {"hn", "h"}:
                self.h = value
            elif name in {"cn", "c"}:
                self.c = value
        return float(np.asarray(outputs[0]).reshape(-1)[0])

    def _prepare_samples(self, samples: Sequence[float]):
        import numpy as np

        target_samples = 512 if self.sample_rate == 16000 else 256
        vad_samples = np.asarray(samples, dtype=np.float32)
        if vad_samples.size >= target_samples:
            return vad_samples
        return np.pad(vad_samples, (0, target_samples - vad_samples.size))


@dataclass(frozen=True)
class VadChoice:
    backend: VadBackend
    warning: str | None = None


def create_vad(config: ClientConfig) -> VadChoice:
    requested = config.vad_backend
    warnings: list[str] = []

    if requested in {"auto", "silero"} and config.silero_vad_onnx_path:
        try:
            return VadChoice(
                backend=SileroOnnxVad(
                    config.silero_vad_onnx_path,
                    config.sample_rate,
                )
            )
        except Exception as exc:
            if requested == "silero":
                raise
            warnings.append(f"Silero VAD unavailable: {exc}")

    if requested in {"auto", "webrtc"}:
        try:
            return VadChoice(
                backend=WebRtcVad(
                    config.sample_rate,
                    config.frame_ms,
                    config.webrtc_vad_aggressiveness,
                )
            )
        except Exception as exc:
            if requested == "webrtc":
                raise
            warnings.append(f"WebRTC VAD unavailable: {exc}")

    return VadChoice(backend=EnergyVad(), warning="; ".join(warnings) or None)
