import pytest

from transcriptify.port import Transcriber
from transcriptify.views import AudioChunk, TranscriptionDeltaType, TranscriptionResult


class _ConcreteTranscriber(Transcriber):
    async def transcribe(self, _: AudioChunk) -> TranscriptionResult:
        return TranscriptionResult(text="hello")


class TestTranscriberStreamFallback:
    @pytest.mark.asyncio
    async def test_stream_yields_single_done_delta(self) -> None:
        transcriber = _ConcreteTranscriber()
        audio = AudioChunk(data=b"\x00" * 64)

        deltas = [d async for d in transcriber.stream(audio)]

        assert len(deltas) == 1

    @pytest.mark.asyncio
    async def test_stream_done_delta_carries_full_text(self) -> None:
        transcriber = _ConcreteTranscriber()
        audio = AudioChunk(data=b"\x00" * 64)

        deltas = [d async for d in transcriber.stream(audio)]

        assert deltas[0].type == TranscriptionDeltaType.DONE
        assert deltas[0].text == "hello"
        assert deltas[0].full_text == "hello"

    @pytest.mark.asyncio
    async def test_stream_delegates_to_transcribe(self) -> None:
        call_log: list[AudioChunk] = []

        class _TrackingTranscriber(Transcriber):
            async def transcribe(self, audio: AudioChunk) -> TranscriptionResult:
                call_log.append(audio)
                return TranscriptionResult(text="tracked")

        audio = AudioChunk(data=b"data")
        transcriber = _TrackingTranscriber()
        _ = [d async for d in transcriber.stream(audio)]

        assert len(call_log) == 1
        assert call_log[0] is audio
