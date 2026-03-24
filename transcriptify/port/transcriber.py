from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from transcriptify.views import AudioChunk, TranscriptionDelta, TranscriptionResult


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

    async def transcribe_streaming(
        self, audio: AudioChunk
    ) -> AsyncIterator[TranscriptionDelta]:
        """Stream transcription deltas for a single audio chunk.

        Adapters with native streaming (e.g. OpenAI stream=True) should
        override this to yield real incremental deltas.  The default
        implementation falls back to a single batch call.
        """
        result = await self.transcribe(audio)
        yield TranscriptionDelta(text=result.text, type="done", full_text=result.text)
