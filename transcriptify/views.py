from dataclasses import dataclass, field
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
