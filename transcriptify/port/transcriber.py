from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from transcriptify.views import (
    AudioChunk,
    TranscriptionDelta,
    TranscriptionDeltaType,
    TranscriptionResult,
)


class Transcriber(ABC):
    """Adapter interface. Every transcription backend implements this."""

    @abstractmethod
    async def transcribe(self, audio: AudioChunk) -> TranscriptionResult:
        """Transcribe a single audio chunk."""
        ...

    async def stream(self, audio: AudioChunk) -> AsyncIterator[TranscriptionDelta]:
        """Stream transcription deltas for a single audio chunk.

        Adapters with native streaming (e.g. OpenAI stream=True) should
        override this to yield real incremental deltas.  The default
        implementation falls back to a single batch call.
        """
        result = await self.transcribe(audio)
        yield TranscriptionDelta(
            text=result.text, type=TranscriptionDeltaType.DONE, full_text=result.text
        )
