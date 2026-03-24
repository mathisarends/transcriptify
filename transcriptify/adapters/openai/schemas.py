from typing import Literal, Union

from pydantic import BaseModel


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


class TranscriptTextDelta(BaseModel):
    type: Literal["transcript.text.delta"] = "transcript.text.delta"
    delta: str
    logprobs: list[float] | None = None


class TranscriptTextDone(BaseModel):
    type: Literal["transcript.text.done"] = "transcript.text.done"
    text: str


class TranscriptTextSegment(BaseModel):
    type: Literal["transcript.text.segment"] = "transcript.text.segment"
    text: str
    speaker: str | None = None
    start: float
    end: float
