import io
import logging
from collections.abc import AsyncIterator

from transcriptify.adapters.openai.schemas import (
    DiarizedTranscription,
    Transcription,
    TranscriptTextDelta,
    TranscriptTextDone,
    TranscriptTextSegment,
    VerboseTranscription,
)
from transcriptify.adapters.openai.views import ResponseFormat, WhisperModel
from transcriptify.port.transcriber import Transcriber
from transcriptify.views import AudioChunk, TranscriptionDelta, TranscriptionResult

import openai

logger = logging.getLogger(__name__)


_WHISPER_ONLY_FORMATS: set[str] = {
    ResponseFormat.VERBOSE_JSON,
    ResponseFormat.SRT,
    ResponseFormat.VTT,
}

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

_PROMPT_MODELS: set[str] = {
    WhisperModel.WHISPER_1,
    WhisperModel.GPT_4O_TRANSCRIBE,
    WhisperModel.GPT_4O_MINI_TRANSCRIBE,
}


class OpenAIWhisper(Transcriber):
    """OpenAI / Azure OpenAI transcription adapter.

    Supports batch and streaming modes, including speaker diarization
    via ``gpt-4o-transcribe-diarize``.
    """

    _DEFAULT_API_VERSION = "2024-06-01"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: WhisperModel | str = WhisperModel.GPT_4O_TRANSCRIBE,
        language: str | None = None,
        response_format: ResponseFormat | str = ResponseFormat.JSON,
        prompt: str | None = None,
        # --- diarization options ---
        chunking_strategy: str | None = None,
        known_speaker_names: list[str] | None = None,
        known_speaker_references: list[str] | None = None,
        # --- azure options ---
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
        is_diarize = model in _DIARIZE_MODELS
        is_gpt4o = model in (
            WhisperModel.GPT_4O_TRANSCRIBE,
            WhisperModel.GPT_4O_MINI_TRANSCRIBE,
        )

        # response_format compatibility
        if is_gpt4o and fmt not in _GPT4O_FORMATS:
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

        if is_diarize and fmt not in _DIARIZE_FORMATS:
            logger.warning(
                "Model %r only supports response_format 'json', 'text', "
                "or 'diarized_json', got %r.",
                model,
                fmt,
            )

        # diarization options on non-diarize model
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

        # prompt not supported on diarize model
        if is_diarize and self._prompt:
            logger.warning(
                "Model %r does not support the prompt parameter.",
                model,
            )

        # whisper-1 does not support streaming
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

    def _base_kwargs(self, audio: AudioChunk) -> dict:
        kwargs: dict = {
            "model": self._model,
            "file": self._to_file(audio),
            "response_format": self._response_format,
        }
        if self._language:
            kwargs["language"] = self._language
        if self._prompt:
            kwargs["prompt"] = self._prompt
        if self._chunking_strategy:
            kwargs["chunking_strategy"] = self._chunking_strategy

        extra_body: dict = {}
        if self._known_speaker_names:
            extra_body["known_speaker_names"] = self._known_speaker_names
        if self._known_speaker_references:
            extra_body["known_speaker_references"] = self._known_speaker_references
        if extra_body:
            kwargs["extra_body"] = extra_body

        return kwargs

    async def transcribe(self, audio: AudioChunk) -> TranscriptionResult:
        response = await self._client.audio.transcriptions.create(
            **self._base_kwargs(audio),
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
                segments=[
                    {"start": s.start, "end": s.end, "text": s.text}
                    for s in verbose.segments
                ],
                raw=verbose,
            )

        if self._response_format == ResponseFormat.DIARIZED_JSON:
            diarized = DiarizedTranscription.model_validate(
                response, from_attributes=True
            )
            return TranscriptionResult(
                text=diarized.text,
                segments=[
                    {
                        "speaker": s.speaker,
                        "start": s.start,
                        "end": s.end,
                        "text": s.text,
                    }
                    for s in diarized.segments
                ],
                raw=diarized,
            )

        if isinstance(response, str):
            return TranscriptionResult(text=response, raw=response)

        plain = Transcription.model_validate(response, from_attributes=True)
        return TranscriptionResult(text=plain.text, raw=plain)


    async def transcribe_streaming(
        self, audio: AudioChunk
    ) -> AsyncIterator[TranscriptionDelta]:
        """Stream transcription deltas using OpenAI ``stream=True``.

        Raises a warning and falls back to batch if the model does not
        support streaming.
        """
        if self._model not in _STREAMABLE_MODELS:
            logger.warning(
                "Model %r does not support streaming — falling back to batch.",
                self._model,
            )
            result = await self.transcribe(audio)
            yield TranscriptionDelta(
                text=result.text, type="done", full_text=result.text
            )
            return

        kwargs = self._base_kwargs(audio)
        kwargs["stream"] = True

        stream = await self._client.audio.transcriptions.create(**kwargs)

        full_text = ""
        async for event in stream:
            event_type = getattr(event, "type", None)

            if event_type == "transcript.text.delta":
                delta = TranscriptTextDelta.model_validate(
                    event, from_attributes=True
                )
                full_text += delta.delta
                yield TranscriptionDelta(text=delta.delta, type="delta")

            elif event_type == "transcript.text.segment":
                seg = TranscriptTextSegment.model_validate(
                    event, from_attributes=True
                )
                yield TranscriptionDelta(
                    text=seg.text,
                    type="segment",
                    speaker=seg.speaker,
                    start=seg.start,
                    end=seg.end,
                )

            elif event_type == "transcript.text.done":
                done = TranscriptTextDone.model_validate(
                    event, from_attributes=True
                )
                full_text = done.text
                yield TranscriptionDelta(
                    text=full_text, type="done", full_text=full_text
                )
