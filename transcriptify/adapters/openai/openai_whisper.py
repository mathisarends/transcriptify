import io
import logging
from collections.abc import AsyncIterator

from pydantic import TypeAdapter, ValidationError

from transcriptify.adapters.openai.schemas import (
    DiarizedTranscription,
    Transcription,
    TranscriptionRequestParams,
    TranscriptStreamEvent,
    TranscriptTextDelta,
    TranscriptTextDone,
    TranscriptTextSegment,
    VerboseTranscription,
)
from transcriptify.adapters.openai.views import ResponseFormat, WhisperModel
from transcriptify.port.transcriber import Transcriber
from transcriptify.views import (
    AudioChunk,
    TranscriptionDelta,
    TranscriptionDeltaType,
    TranscriptionResult,
)

import openai

logger = logging.getLogger(__name__)

_stream_event_adapter: TypeAdapter[TranscriptStreamEvent] = TypeAdapter(
    TranscriptStreamEvent
)


class OpenAIWhisper(Transcriber):
    _GPT4O_FORMATS: set[str] = {
        ResponseFormat.JSON,
        ResponseFormat.TEXT,
    }
    _DIARIZE_FORMATS: set[str] = {
        ResponseFormat.JSON,
        ResponseFormat.TEXT,
        ResponseFormat.DIARIZED_JSON,
    }
    _STREAMABLE_MODELS: set[str] = {
        WhisperModel.GPT_4O_TRANSCRIBE,
        WhisperModel.GPT_4O_MINI_TRANSCRIBE,
        WhisperModel.GPT_4O_TRANSCRIBE_DIARIZE,
    }
    _DIARIZE_MODELS: set[str] = {
        WhisperModel.GPT_4O_TRANSCRIBE_DIARIZE,
    }

    _DEFAULT_API_VERSION = "2024-06-01"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: WhisperModel | str = WhisperModel.GPT_4O_TRANSCRIBE,
        language: str | None = None,
        response_format: ResponseFormat | str = ResponseFormat.JSON,
        prompt: str | None = None,
        chunking_strategy: str | None = None,
        known_speaker_names: list[str] | None = None,
        known_speaker_references: list[str] | None = None,
        azure_endpoint: str | None = None,
        azure_deployment: str | None = None,
        api_version: str | None = None,
    ) -> None:
        self._model = str(model)
        self._language = language
        self._response_format = str(response_format)
        self._prompt = prompt
        self._chunking_strategy = chunking_strategy
        self._known_speaker_names = known_speaker_names
        self._known_speaker_references = known_speaker_references

        self._validate_config()

        if azure_endpoint:
            self._client = openai.AsyncAzureOpenAI(
                api_key=api_key,
                azure_endpoint=azure_endpoint,
                azure_deployment=azure_deployment or self._model,
                api_version=api_version or self._DEFAULT_API_VERSION,
            )
        else:
            self._client = openai.AsyncOpenAI(api_key=api_key)

    def _validate_config(self) -> None:
        model = self._model
        fmt = self._response_format

        is_whisper = model == WhisperModel.WHISPER_1
        is_diarize = model in self._DIARIZE_MODELS
        is_gpt_4o = model in (
            WhisperModel.GPT_4O_TRANSCRIBE,
            WhisperModel.GPT_4O_MINI_TRANSCRIBE,
        )

        if is_gpt_4o and fmt not in self._GPT4O_FORMATS:
            logger.warning(
                "Model %r only supports response_format 'json' or 'text', "
                "got %r — the API will reject this request.",
                model,
                fmt,
            )

        if is_whisper and fmt == ResponseFormat.DIARIZED_JSON:
            logger.warning(
                "Model %r does not support response_format 'diarized_json'.",
                model,
            )

        if is_diarize and fmt not in self._DIARIZE_FORMATS:
            logger.warning(
                "Model %r only supports response_format 'json', 'text', "
                "or 'diarized_json', got %r.",
                model,
                fmt,
            )

        if not is_diarize:
            if self._known_speaker_names or self._known_speaker_references:
                logger.warning(
                    "known_speaker_names/known_speaker_references are only "
                    "supported with model %r, current model is %r.",
                    WhisperModel.GPT_4O_TRANSCRIBE_DIARIZE,
                    model,
                )
            if self._chunking_strategy:
                logger.warning(
                    "chunking_strategy is only supported with model %r, "
                    "current model is %r.",
                    WhisperModel.GPT_4O_TRANSCRIBE_DIARIZE,
                    model,
                )

        if is_diarize and self._prompt:
            logger.warning(
                "Model %r does not support the prompt parameter.",
                model,
            )

        if is_whisper:
            logger.debug(
                "Model %r does not support streaming transcription.",
                model,
            )

    @staticmethod
    def _to_file(audio: AudioChunk) -> io.BytesIO:
        ext = "wav" if audio.encoding == "pcm_s16le" else audio.encoding
        buf = io.BytesIO(audio.data)
        buf.name = f"audio.{ext}"
        return buf

    def _build_params(self, audio: AudioChunk) -> TranscriptionRequestParams:
        extra_body: dict | None = None
        raw_extra: dict = {}
        if self._known_speaker_names:
            raw_extra["known_speaker_names"] = self._known_speaker_names
        if self._known_speaker_references:
            raw_extra["known_speaker_references"] = self._known_speaker_references
        if raw_extra:
            extra_body = raw_extra

        return TranscriptionRequestParams(
            model=self._model,
            file=self._to_file(audio),
            response_format=self._response_format,
            language=self._language,
            prompt=self._prompt,
            chunking_strategy=self._chunking_strategy,
            extra_body=extra_body,
        )

    async def transcribe(self, audio: AudioChunk) -> TranscriptionResult:
        transcription_request_params = self._build_params(audio)

        response = await self._client.audio.transcriptions.create(
            **transcription_request_params.to_api_kwargs(),
        )
        return self._parse(response)

    def _parse(self, response: object) -> TranscriptionResult:
        if self._response_format == ResponseFormat.VERBOSE_JSON:
            verbose = VerboseTranscription.model_validate(
                response, from_attributes=True
            )
            return TranscriptionResult(
                text=verbose.text,
                language=verbose.language,
                segments=verbose.segments,
                raw=verbose,
            )

        if self._response_format == ResponseFormat.DIARIZED_JSON:
            diarized = DiarizedTranscription.model_validate(
                response, from_attributes=True
            )
            return TranscriptionResult(
                text=diarized.text,
                segments=diarized.segments,
                raw=diarized,
            )

        if isinstance(response, str):
            return TranscriptionResult(text=response, raw=response)

        plain = Transcription.model_validate(response, from_attributes=True)
        return TranscriptionResult(text=plain.text, raw=plain)

    async def stream(self, audio: AudioChunk) -> AsyncIterator[TranscriptionDelta]:
        """Stream transcription deltas using OpenAI ``stream=True``.

        Falls back to batch if the model does not support streaming.
        """
        if self._model not in self._STREAMABLE_MODELS:
            logger.warning(
                "Model %r does not support streaming — falling back to batch.",
                self._model,
            )
            result = await self.transcribe(audio)
            yield TranscriptionDelta(
                text=result.text,
                type=TranscriptionDeltaType.DONE,
                full_text=result.text,
            )
            return

        transcription_request_params = self._build_params(audio)
        kwargs = transcription_request_params.to_api_kwargs()
        kwargs["stream"] = True

        stream = await self._client.audio.transcriptions.create(**kwargs)

        full_text = ""
        async for event in stream:
            try:
                parsed = _stream_event_adapter.validate_python(
                    event, from_attributes=True
                )
            except ValidationError:
                continue

            if isinstance(parsed, TranscriptTextDelta):
                full_text += parsed.delta
                yield TranscriptionDelta(
                    text=parsed.delta, type=TranscriptionDeltaType.DELTA
                )

            elif isinstance(parsed, TranscriptTextSegment):
                yield TranscriptionDelta(
                    text=parsed.text,
                    type=TranscriptionDeltaType.SEGMENT,
                    speaker=parsed.speaker,
                    start=parsed.start,
                    end=parsed.end,
                )

            elif isinstance(parsed, TranscriptTextDone):
                full_text = parsed.text
                yield TranscriptionDelta(
                    text=full_text,
                    type=TranscriptionDeltaType.DONE,
                    full_text=full_text,
                )
