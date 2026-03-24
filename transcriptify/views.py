from dataclasses import dataclass, field
from typing import Any, Literal


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


@dataclass
class TranscriptionDelta:
    """A single chunk emitted during streaming transcription."""

    text: str
    type: Literal["delta", "segment", "done"]
    full_text: str | None = None
    speaker: str | None = None
    start: float | None = None
    end: float | None = None
