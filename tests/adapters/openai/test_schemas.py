import io

from pydantic import TypeAdapter

from transcriptify.adapters.openai.schemas import (
    DiarizedSegment,
    DiarizedTranscription,
    Transcription,
    TranscriptionRequestParams,
    TranscriptTextDelta,
    TranscriptTextDone,
    TranscriptTextSegment,
    VerboseTranscription,
)

TranscriptStreamEvent = TranscriptTextDelta | TranscriptTextDone | TranscriptTextSegment


class TestTranscription:
    def test_stores_text(self) -> None:
        t = Transcription(text="hello world")
        assert t.text == "hello world"


class TestVerboseTranscription:
    def test_required_fields(self) -> None:
        vt = VerboseTranscription(duration=3.0, language="en", text="hello")
        assert vt.text == "hello"
        assert vt.language == "en"
        assert vt.duration == 3.0

    def test_segments_default_to_empty_list(self) -> None:
        vt = VerboseTranscription(duration=1.0, language="fr", text="bonjour")
        assert vt.segments == []

    def test_words_default_to_none(self) -> None:
        vt = VerboseTranscription(duration=1.0, language="en", text="hi")
        assert vt.words is None

    def test_validates_from_attributes(self) -> None:
        vt = VerboseTranscription(duration=2.5, language="de", text="hallo")
        validated = VerboseTranscription.model_validate(vt, from_attributes=True)
        assert validated.text == "hallo"
        assert validated.language == "de"


class TestDiarizedTranscription:
    def test_segments_contain_speaker_and_timestamps(self) -> None:
        data = {
            "text": "hello there",
            "segments": [
                {"speaker": "A", "text": "hello", "start": 0.0, "end": 1.0},
                {"speaker": "B", "text": "there", "start": 1.0, "end": 2.0},
            ],
        }
        dt = DiarizedTranscription.model_validate(data)
        assert dt.text == "hello there"
        assert len(dt.segments) == 2
        assert dt.segments[0].speaker == "A"
        assert dt.segments[0].start == 0.0
        assert dt.segments[1].speaker == "B"
        assert dt.segments[1].end == 2.0

    def test_segments_default_to_empty_list(self) -> None:
        dt = DiarizedTranscription(text="no speakers")
        assert dt.segments == []


class TestDiarizedSegment:
    def test_fields(self) -> None:
        seg = DiarizedSegment(speaker="C", text="hey", start=5.0, end=6.5)
        assert seg.speaker == "C"
        assert seg.text == "hey"
        assert seg.start == 5.0
        assert seg.end == 6.5


class TestTranscriptStreamEventDiscriminator:
    _adapter: TypeAdapter = TypeAdapter(
        TranscriptTextDelta | TranscriptTextDone | TranscriptTextSegment
    )

    def test_text_delta_dispatched_correctly(self) -> None:
        parsed = TranscriptTextDelta.model_validate(
            {"type": "transcript.text.delta", "delta": "partial "}
        )
        assert isinstance(parsed, TranscriptTextDelta)
        assert parsed.delta == "partial "

    def test_text_done_dispatched_correctly(self) -> None:
        parsed = TranscriptTextDone.model_validate(
            {"type": "transcript.text.done", "text": "full sentence"}
        )
        assert isinstance(parsed, TranscriptTextDone)
        assert parsed.text == "full sentence"

    def test_text_segment_dispatched_correctly(self) -> None:
        parsed = TranscriptTextSegment.model_validate(
            {
                "type": "transcript.text.segment",
                "text": "hi",
                "speaker": "A",
                "start": 0.0,
                "end": 1.0,
            }
        )
        assert isinstance(parsed, TranscriptTextSegment)
        assert parsed.speaker == "A"
        assert parsed.start == 0.0

    def test_text_segment_speaker_is_optional(self) -> None:
        parsed = TranscriptTextSegment.model_validate(
            {"type": "transcript.text.segment", "text": "hi", "start": 0.0, "end": 1.0}
        )
        assert parsed.speaker is None

    def test_delta_logprobs_default_to_none(self) -> None:
        parsed = TranscriptTextDelta.model_validate(
            {"type": "transcript.text.delta", "delta": "x"}
        )
        assert parsed.logprobs is None


class TestTranscriptionRequestParams:
    def test_to_api_kwargs_excludes_none_fields(self) -> None:
        buf = io.BytesIO(b"audio")
        params = TranscriptionRequestParams(
            model="gpt-4o-transcribe",
            file=buf,
            response_format="json",
        )
        kwargs = params.to_api_kwargs()

        assert "language" not in kwargs
        assert "prompt" not in kwargs
        assert "chunking_strategy" not in kwargs
        assert "extra_body" not in kwargs

    def test_to_api_kwargs_includes_explicitly_set_fields(self) -> None:
        buf = io.BytesIO(b"audio")
        params = TranscriptionRequestParams(
            model="whisper-1",
            file=buf,
            response_format="verbose_json",
            language="en",
            prompt="Be concise.",
        )
        kwargs = params.to_api_kwargs()

        assert kwargs["language"] == "en"
        assert kwargs["prompt"] == "Be concise."
        assert kwargs["response_format"] == "verbose_json"

    def test_file_object_is_passed_through_without_serialization(self) -> None:
        buf = io.BytesIO(b"raw bytes")
        params = TranscriptionRequestParams(
            model="whisper-1", file=buf, response_format="json"
        )
        kwargs = params.to_api_kwargs()

        assert kwargs["file"] is buf

    def test_extra_body_included_when_set(self) -> None:
        buf = io.BytesIO(b"audio")
        params = TranscriptionRequestParams(
            model="gpt-4o-transcribe-diarize",
            file=buf,
            response_format="diarized_json",
            extra_body={"known_speaker_names": ["Alice", "Bob"]},
        )
        kwargs = params.to_api_kwargs()

        assert kwargs["extra_body"] == {"known_speaker_names": ["Alice", "Bob"]}
