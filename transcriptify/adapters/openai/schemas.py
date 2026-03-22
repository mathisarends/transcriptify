from typing import TypeAlias

from pydantic import BaseModel


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
    words: list[TranscriptionWord] = []


TranscriptionCreateResponse: TypeAlias = Transcription | VerboseTranscription
