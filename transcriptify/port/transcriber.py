from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from transcriptify.views import AudioChunk, TranscriptionResult


class Transcriber(ABC):
    """Adapter interface. Every transcription backend implements this."""

    @abstractmethod
    async def transcribe(self, audio: AudioChunk) -> TranscriptionResult:
        """Transcribe a single audio chunk."""
        ...

    async def transcribe_stream(
        self, chunks: AsyncIterator[AudioChunk]
    ) -> AsyncIterator[TranscriptionResult]:
        """Transcribe chunks sequentially.

        Adapters with native streaming support can override this.
        """
        async for chunk in chunks:
            yield await self.transcribe(chunk)
