from __future__ import annotations

from abc import ABC, abstractmethod

from server.asr_types import AudioData


class ASR(ABC):
    @abstractmethod
    def transcribe(self, audio: AudioData) -> str:
        raise NotImplementedError
