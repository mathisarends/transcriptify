from transcriptify.views import (
    AudioChunk,
    TranscriptionDelta,
    TranscriptionDeltaType,
    TranscriptionResult,
)


class TestAudioChunk:
    def test_defaults(self) -> None:
        chunk = AudioChunk(data=b"audio")
        assert chunk.sample_rate == 16_000
        assert chunk.channels == 1
        assert chunk.encoding == "pcm_s16le"
        assert chunk.metadata == {}

    def test_custom_values(self) -> None:
        chunk = AudioChunk(
            data=b"audio",
            sample_rate=44_100,
            channels=2,
            encoding="mp3",
            metadata={"source": "mic"},
        )
        assert chunk.sample_rate == 44_100
        assert chunk.channels == 2
        assert chunk.encoding == "mp3"
        assert chunk.metadata == {"source": "mic"}

    def test_metadata_is_independent_per_instance(self) -> None:
        chunk_a = AudioChunk(data=b"a")
        chunk_b = AudioChunk(data=b"b")
        chunk_a.metadata["key"] = "value"
        assert "key" not in chunk_b.metadata


class TestTranscriptionResult:
    def test_defaults(self) -> None:
        result = TranscriptionResult(text="hello")
        assert result.text == "hello"
        assert result.language is None
        assert result.confidence is None
        assert result.segments == []
        assert result.raw is None

    def test_with_all_fields(self) -> None:
        result = TranscriptionResult(
            text="guten tag",
            language="de",
            confidence=0.95,
            segments=[{"id": 0, "text": "guten tag"}],
            raw={"model": "whisper-1"},
        )
        assert result.language == "de"
        assert result.confidence == 0.95
        assert len(result.segments) == 1
        assert result.raw == {"model": "whisper-1"}

    def test_segments_list_is_independent_per_instance(self) -> None:
        result_a = TranscriptionResult(text="a")
        result_b = TranscriptionResult(text="b")
        result_a.segments.append({"id": 1})
        assert result_b.segments == []


class TestTranscriptionDelta:
    def test_delta_type_defaults(self) -> None:
        delta = TranscriptionDelta(text="partial", type=TranscriptionDeltaType.DELTA)
        assert delta.text == "partial"
        assert delta.type == TranscriptionDeltaType.DELTA
        assert delta.full_text is None
        assert delta.speaker is None
        assert delta.start is None
        assert delta.end is None

    def test_done_carries_full_text(self) -> None:
        delta = TranscriptionDelta(
            text="full",
            type=TranscriptionDeltaType.DONE,
            full_text="full",
        )
        assert delta.type == TranscriptionDeltaType.DONE
        assert delta.full_text == "full"

    def test_segment_carries_speaker_and_timestamps(self) -> None:
        delta = TranscriptionDelta(
            text="hello",
            type=TranscriptionDeltaType.SEGMENT,
            speaker="A",
            start=1.0,
            end=2.5,
        )
        assert delta.speaker == "A"
        assert delta.start == 1.0
        assert delta.end == 2.5


class TestTranscriptionDeltaType:
    def test_string_values(self) -> None:
        assert TranscriptionDeltaType.DELTA == "delta"
        assert TranscriptionDeltaType.SEGMENT == "segment"
        assert TranscriptionDeltaType.DONE == "done"

    def test_is_str_enum(self) -> None:
        assert isinstance(TranscriptionDeltaType.DELTA, str)
