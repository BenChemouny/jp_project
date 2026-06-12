from __future__ import annotations

from dataclasses import dataclass
import wave


@dataclass(frozen=True)
class AudioData:
    wav_path: str
    sample_rate: int
    channels: int
    frames: int

    @property
    def duration_s(self) -> float:
        if self.frames == 0:
            return 0.0
        return float(self.frames) / float(self.sample_rate)

    @classmethod
    def from_wav_path(cls, wav_path: str) -> "AudioData":
        with wave.open(wav_path, "rb") as wav_file:
            return cls(
                wav_path=wav_path,
                sample_rate=wav_file.getframerate(),
                channels=wav_file.getnchannels(),
                frames=wav_file.getnframes(),
            )
