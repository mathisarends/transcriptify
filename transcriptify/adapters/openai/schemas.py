from enum import StrEnum
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


# ------------------------------------------------------------------
# Batch response schemas
# ------------------------------------------------------------------


class TranscriptionSegment(BaseModel):
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: list[int]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float


class TranscriptionWord(BaseModel):
    word: str
    start: float
    end: float


class Transcription(BaseModel):
    text: str


class VerboseTranscription(BaseModel):
    duration: float
    language: str
    text: str
    segments: list[TranscriptionSegment] = []
    words: list[TranscriptionWord] | None = None


class DiarizedSegment(BaseModel):
    speaker: str
    text: str
    start: float
    end: float


class DiarizedTranscription(BaseModel):
    text: str
    segments: list[DiarizedSegment] = []


TranscriptionCreateResponse = Union[
    Transcription, VerboseTranscription, DiarizedTranscription
]


# ------------------------------------------------------------------
# Streaming event schemas
# ------------------------------------------------------------------


class TranscriptEventType(StrEnum):
    TEXT_DELTA = "transcript.text.delta"
    TEXT_DONE = "transcript.text.done"
    TEXT_SEGMENT = "transcript.text.segment"


class TranscriptTextDelta(BaseModel):
    type: Literal[TranscriptEventType.TEXT_DELTA] = TranscriptEventType.TEXT_DELTA
    delta: str
    logprobs: list[float] | None = None


class TranscriptTextDone(BaseModel):
    type: Literal[TranscriptEventType.TEXT_DONE] = TranscriptEventType.TEXT_DONE
    text: str


class TranscriptTextSegment(BaseModel):
    type: Literal[TranscriptEventType.TEXT_SEGMENT] = TranscriptEventType.TEXT_SEGMENT
    text: str
    speaker: str | None = None
    start: float
    end: float


TranscriptStreamEvent = Annotated[
    Union[TranscriptTextDelta, TranscriptTextDone, TranscriptTextSegment],
    Field(discriminator="type"),
]


# ------------------------------------------------------------------
# Request params
# ------------------------------------------------------------------


class TranscriptionRequestParams(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: str
    file: Any
    response_format: str
    language: str | None = None
    prompt: str | None = None
    chunking_strategy: str | None = None
    extra_body: dict | None = None

    def to_api_kwargs(self) -> dict:
        """Return a dict suitable for openai client calls.

        ``file`` is kept as the raw object (BytesIO / bytes / PathLike)
        so the openai SDK can handle it correctly.
        """
        kwargs = self.model_dump(exclude_none=True, exclude={"file"})
        kwargs["file"] = self.file
        return kwargs
