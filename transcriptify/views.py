from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


@dataclass
class AudioChunk:
    data: bytes
    sample_rate: int = 16_000
    channels: int = 1
    encoding: str = "pcm_s16le"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TranscriptionResult:
    text: str
    language: str | None = None
    confidence: float | None = None
    segments: list[dict[str, Any]] = field(default_factory=list)
    raw: Any = None


class TranscriptionDeltaType(StrEnum):
    DELTA = "delta"
    SEGMENT = "segment"
    DONE = "done"


@dataclass
class TranscriptionDelta:
    """A single chunk emitted during streaming transcription."""

    text: str
    type: TranscriptionDeltaType
    full_text: str | None = None
    speaker: str | None = None
    start: float | None = None
    end: float | None = None
