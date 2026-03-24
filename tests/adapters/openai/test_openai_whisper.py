import io
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from transcriptify.adapters.openai.openai_whisper import OpenAIWhisper
from transcriptify.adapters.openai.schemas import (
    DiarizedTranscription,
    Transcription,
    TranscriptTextDelta,
    TranscriptTextDone,
    TranscriptTextSegment,
    VerboseTranscription,
)
from transcriptify.adapters.openai.views import ResponseFormat, WhisperModel
from transcriptify.views import AudioChunk, TranscriptionDeltaType

_DUMMY_KEY = "sk-test"
_SILENT_AUDIO = AudioChunk(data=b"\x00" * 128, sample_rate=16_000)


class TestToFile:
    def test_pcm_encoding_produces_wav_filename(self) -> None:
        chunk = AudioChunk(data=b"data", encoding="pcm_s16le")
        buf = OpenAIWhisper._to_file(chunk)
        assert buf.name == "audio.wav"

    def test_mp3_encoding_produces_mp3_filename(self) -> None:
        chunk = AudioChunk(data=b"data", encoding="mp3")
        buf = OpenAIWhisper._to_file(chunk)
        assert buf.name == "audio.mp3"

    def test_returns_bytesio_containing_original_bytes(self) -> None:
        data = b"raw audio bytes"
        chunk = AudioChunk(data=data, encoding="pcm_s16le")
        buf = OpenAIWhisper._to_file(chunk)
        assert isinstance(buf, io.BytesIO)
        assert buf.read() == data


class TestValidateConfig:
    def test_gpt4o_with_verbose_json_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(
            logging.WARNING, logger="transcriptify.adapters.openai.openai_whisper"
        ):
            OpenAIWhisper(
                api_key=_DUMMY_KEY,
                model=WhisperModel.GPT_4O_TRANSCRIBE,
                response_format=ResponseFormat.VERBOSE_JSON,
            )
        assert "only supports response_format" in caplog.text

    def test_whisper1_with_diarized_json_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(
            logging.WARNING, logger="transcriptify.adapters.openai.openai_whisper"
        ):
            OpenAIWhisper(
                api_key=_DUMMY_KEY,
                model=WhisperModel.WHISPER_1,
                response_format=ResponseFormat.DIARIZED_JSON,
            )
        assert "diarized_json" in caplog.text

    def test_non_diarize_model_with_speaker_names_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(
            logging.WARNING, logger="transcriptify.adapters.openai.openai_whisper"
        ):
            OpenAIWhisper(
                api_key=_DUMMY_KEY,
                model=WhisperModel.GPT_4O_TRANSCRIBE,
                known_speaker_names=["Alice", "Bob"],
            )
        assert "known_speaker_names" in caplog.text

    def test_diarize_model_with_prompt_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(
            logging.WARNING, logger="transcriptify.adapters.openai.openai_whisper"
        ):
            OpenAIWhisper(
                api_key=_DUMMY_KEY,
                model=WhisperModel.GPT_4O_TRANSCRIBE_DIARIZE,
                prompt="please transcribe accurately",
            )
        assert "prompt" in caplog.text

    def test_non_diarize_model_with_chunking_strategy_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(
            logging.WARNING, logger="transcriptify.adapters.openai.openai_whisper"
        ):
            OpenAIWhisper(
                api_key=_DUMMY_KEY,
                model=WhisperModel.GPT_4O_TRANSCRIBE,
                chunking_strategy="auto",
            )
        assert "chunking_strategy" in caplog.text

    def test_valid_gpt4o_json_config_produces_no_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(
            logging.WARNING, logger="transcriptify.adapters.openai.openai_whisper"
        ):
            OpenAIWhisper(
                api_key=_DUMMY_KEY,
                model=WhisperModel.GPT_4O_TRANSCRIBE,
                response_format=ResponseFormat.JSON,
            )
        assert caplog.text == ""


class TestAzureClientCreation:
    def test_uses_azure_client_when_endpoint_provided(self) -> None:
        with patch(
            "transcriptify.adapters.openai.openai_whisper.openai.AsyncAzureOpenAI"
        ) as MockAzure:
            OpenAIWhisper(
                api_key=_DUMMY_KEY,
                azure_endpoint="https://my-resource.openai.azure.com",
            )
        MockAzure.assert_called_once()
        call_kwargs = MockAzure.call_args.kwargs
        assert call_kwargs["azure_endpoint"] == "https://my-resource.openai.azure.com"

    def test_azure_client_uses_default_api_version_when_none_given(self) -> None:
        with patch(
            "transcriptify.adapters.openai.openai_whisper.openai.AsyncAzureOpenAI"
        ) as MockAzure:
            OpenAIWhisper(
                api_key=_DUMMY_KEY,
                azure_endpoint="https://my-resource.openai.azure.com",
            )
        call_kwargs = MockAzure.call_args.kwargs
        assert call_kwargs["api_version"] == OpenAIWhisper._DEFAULT_API_VERSION

    def test_azure_client_uses_model_as_deployment_when_not_specified(self) -> None:
        with patch(
            "transcriptify.adapters.openai.openai_whisper.openai.AsyncAzureOpenAI"
        ) as MockAzure:
            OpenAIWhisper(
                api_key=_DUMMY_KEY,
                model=WhisperModel.GPT_4O_TRANSCRIBE,
                azure_endpoint="https://my-resource.openai.azure.com",
            )
        call_kwargs = MockAzure.call_args.kwargs
        assert call_kwargs["azure_deployment"] == WhisperModel.GPT_4O_TRANSCRIBE


class TestBuildParams:
    def test_assembles_basic_params(self) -> None:
        with patch("transcriptify.adapters.openai.openai_whisper.openai.AsyncOpenAI"):
            whisper = OpenAIWhisper(
                api_key=_DUMMY_KEY,
                model=WhisperModel.GPT_4O_TRANSCRIBE,
                response_format=ResponseFormat.JSON,
                language="en",
            )
        params = whisper._build_params(_SILENT_AUDIO)
        assert params.model == WhisperModel.GPT_4O_TRANSCRIBE
        assert params.response_format == ResponseFormat.JSON
        assert params.language == "en"

    def test_extra_body_set_when_speaker_names_provided(self) -> None:
        with patch("transcriptify.adapters.openai.openai_whisper.openai.AsyncOpenAI"):
            whisper = OpenAIWhisper(
                api_key=_DUMMY_KEY,
                model=WhisperModel.GPT_4O_TRANSCRIBE_DIARIZE,
                known_speaker_names=["Alice", "Bob"],
            )
        params = whisper._build_params(_SILENT_AUDIO)
        assert params.extra_body == {"known_speaker_names": ["Alice", "Bob"]}

    def test_extra_body_none_without_speaker_options(self) -> None:
        with patch("transcriptify.adapters.openai.openai_whisper.openai.AsyncOpenAI"):
            whisper = OpenAIWhisper(api_key=_DUMMY_KEY)
        params = whisper._build_params(_SILENT_AUDIO)
        assert params.extra_body is None

    def test_file_is_bytesio_with_audio_data(self) -> None:
        with patch("transcriptify.adapters.openai.openai_whisper.openai.AsyncOpenAI"):
            whisper = OpenAIWhisper(api_key=_DUMMY_KEY)
        params = whisper._build_params(_SILENT_AUDIO)
        assert isinstance(params.file, io.BytesIO)
        assert params.file.read() == _SILENT_AUDIO.data


class TestParse:
    def _whisper(self, response_format: str) -> OpenAIWhisper:
        with patch("transcriptify.adapters.openai.openai_whisper.openai.AsyncOpenAI"):
            return OpenAIWhisper(api_key=_DUMMY_KEY, response_format=response_format)

    def test_json_format_parses_transcription_object(self) -> None:
        whisper = self._whisper(ResponseFormat.JSON)
        result = whisper._parse(Transcription(text="hello world"))
        assert result.text == "hello world"
        assert result.language is None

    def test_text_format_parses_raw_string(self) -> None:
        whisper = self._whisper(ResponseFormat.TEXT)
        result = whisper._parse("plain text response")
        assert result.text == "plain text response"

    def test_verbose_json_extracts_language_and_segments(self) -> None:
        whisper = self._whisper(ResponseFormat.VERBOSE_JSON)
        verbose = VerboseTranscription(duration=2.0, language="fr", text="bonjour")
        result = whisper._parse(verbose)
        assert result.text == "bonjour"
        assert result.language == "fr"
        assert isinstance(result.raw, VerboseTranscription)

    def test_diarized_json_populates_segments(self) -> None:
        whisper = self._whisper(ResponseFormat.DIARIZED_JSON)
        diarized = DiarizedTranscription(
            text="hello there",
            segments=[
                {"speaker": "A", "text": "hello", "start": 0.0, "end": 1.0},
                {"speaker": "B", "text": "there", "start": 1.0, "end": 2.0},
            ],
        )
        result = whisper._parse(diarized)
        assert result.text == "hello there"
        assert len(result.segments) == 2
        assert isinstance(result.raw, DiarizedTranscription)


class TestTranscribe:
    @pytest.mark.asyncio
    async def test_calls_openai_api_and_maps_result(self) -> None:
        with patch(
            "transcriptify.adapters.openai.openai_whisper.openai.AsyncOpenAI"
        ) as MockClient:
            mock_client = MagicMock()
            MockClient.return_value = mock_client
            mock_client.audio.transcriptions.create = AsyncMock(
                return_value=Transcription(text="transcribed audio")
            )
            whisper = OpenAIWhisper(
                api_key=_DUMMY_KEY, response_format=ResponseFormat.JSON
            )
            result = await whisper.transcribe(_SILENT_AUDIO)

        assert result.text == "transcribed audio"

    @pytest.mark.asyncio
    async def test_passes_correct_model_to_api(self) -> None:
        with patch(
            "transcriptify.adapters.openai.openai_whisper.openai.AsyncOpenAI"
        ) as MockClient:
            mock_client = MagicMock()
            MockClient.return_value = mock_client
            mock_client.audio.transcriptions.create = AsyncMock(
                return_value=Transcription(text="ok")
            )
            whisper = OpenAIWhisper(api_key=_DUMMY_KEY, model=WhisperModel.WHISPER_1)
            await whisper.transcribe(_SILENT_AUDIO)

        call_kwargs = mock_client.audio.transcriptions.create.call_args.kwargs
        assert call_kwargs["model"] == WhisperModel.WHISPER_1


class TestStream:
    @pytest.mark.asyncio
    async def test_non_streamable_model_falls_back_to_single_done_delta(self) -> None:
        with patch(
            "transcriptify.adapters.openai.openai_whisper.openai.AsyncOpenAI"
        ) as MockClient:
            mock_client = MagicMock()
            MockClient.return_value = mock_client
            mock_client.audio.transcriptions.create = AsyncMock(
                return_value=Transcription(text="batch result")
            )
            whisper = OpenAIWhisper(api_key=_DUMMY_KEY, model=WhisperModel.WHISPER_1)
            deltas = [d async for d in whisper.stream(_SILENT_AUDIO)]

        assert len(deltas) == 1
        assert deltas[0].type == TranscriptionDeltaType.DONE
        assert deltas[0].text == "batch result"
        assert deltas[0].full_text == "batch result"

    @pytest.mark.asyncio
    async def test_streamable_model_yields_delta_and_done_events(self) -> None:
        delta_event = TranscriptTextDelta(delta="hello ")
        done_event = TranscriptTextDone(text="hello world")

        async def _mock_stream():
            yield delta_event
            yield done_event

        with patch(
            "transcriptify.adapters.openai.openai_whisper.openai.AsyncOpenAI"
        ) as MockClient:
            mock_client = MagicMock()
            MockClient.return_value = mock_client
            mock_client.audio.transcriptions.create = AsyncMock(
                return_value=_mock_stream()
            )

            whisper = OpenAIWhisper(
                api_key=_DUMMY_KEY, model=WhisperModel.GPT_4O_TRANSCRIBE
            )
            deltas = [d async for d in whisper.stream(_SILENT_AUDIO)]

        types = [d.type for d in deltas]
        assert TranscriptionDeltaType.DELTA in types
        assert TranscriptionDeltaType.DONE in types

    @pytest.mark.asyncio
    async def test_stream_segment_event_carries_speaker_and_timestamps(self) -> None:
        segment_event = TranscriptTextSegment(
            text="hi", speaker="A", start=0.0, end=1.0
        )
        done_event = TranscriptTextDone(text="hi")

        async def _mock_stream():
            yield segment_event
            yield done_event

        with patch(
            "transcriptify.adapters.openai.openai_whisper.openai.AsyncOpenAI"
        ) as MockClient:
            mock_client = MagicMock()
            MockClient.return_value = mock_client
            mock_client.audio.transcriptions.create = AsyncMock(
                return_value=_mock_stream()
            )

            whisper = OpenAIWhisper(
                api_key=_DUMMY_KEY, model=WhisperModel.GPT_4O_TRANSCRIBE
            )
            deltas = [d async for d in whisper.stream(_SILENT_AUDIO)]

        segment_deltas = [d for d in deltas if d.type == TranscriptionDeltaType.SEGMENT]
        assert len(segment_deltas) == 1
        assert segment_deltas[0].speaker == "A"
        assert segment_deltas[0].start == 0.0
        assert segment_deltas[0].end == 1.0

    @pytest.mark.asyncio
    async def test_stream_done_event_sets_full_text(self) -> None:
        done_event = TranscriptTextDone(text="complete sentence")

        async def _mock_stream():
            yield done_event

        with patch(
            "transcriptify.adapters.openai.openai_whisper.openai.AsyncOpenAI"
        ) as MockClient:
            mock_client = MagicMock()
            MockClient.return_value = mock_client
            mock_client.audio.transcriptions.create = AsyncMock(
                return_value=_mock_stream()
            )

            whisper = OpenAIWhisper(
                api_key=_DUMMY_KEY, model=WhisperModel.GPT_4O_TRANSCRIBE
            )
            deltas = [d async for d in whisper.stream(_SILENT_AUDIO)]

        done = next(d for d in deltas if d.type == TranscriptionDeltaType.DONE)
        assert done.full_text == "complete sentence"
