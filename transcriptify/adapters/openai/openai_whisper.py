import io

from transcriptify.views import AudioChunk, TranscriptionResult
from transcriptify.port.transcriber import Transcriber
from transcriptify.adapters.openai.schemas import Transcription, VerboseTranscription
from transcriptify.adapters.openai.views import ResponseFormat, WhisperModel

try:
    import openai
except ImportError as exc:
    raise ImportError(
        "Install with: pip install transcriptify[openai]"
    ) from exc


class OpenAIWhisper(Transcriber):
    _DEFAULT_API_VERSION = "2024-06-01"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: WhisperModel | str = WhisperModel.WHISPER_1,
        language: str | None = None,
        response_format: ResponseFormat | str = ResponseFormat.VERBOSE_JSON,
        azure_endpoint: str | None = None,
        azure_deployment: str | None = None,
        api_version: str | None = None,
    ) -> None:
        self._model = str(model)
        self._language = language
        self._response_format = str(response_format)

        if azure_endpoint:
            self._client = openai.AsyncAzureOpenAI(
                api_key=api_key,
                azure_endpoint=azure_endpoint,
                azure_deployment=azure_deployment or self._model,
                api_version=api_version or self._DEFAULT_API_VERSION,
            )
        else:
            self._client = openai.AsyncOpenAI(api_key=api_key)

    @staticmethod
    def _to_file(audio: AudioChunk) -> io.BytesIO:
        ext = "wav" if audio.encoding == "pcm_s16le" else audio.encoding
        print(f"encoding={audio.encoding!r}, ext={ext}, bytes={len(audio.data)}")
        buf = io.BytesIO(audio.data)
        buf.name = f"audio.{ext}"
        return buf

    async def transcribe(self, audio: AudioChunk) -> TranscriptionResult:
        kwargs: dict = {
            "model": self._model,
            "file": self._to_file(audio),
            "response_format": self._response_format,
        }
        if self._language:
            kwargs["language"] = self._language

        response = await self._client.audio.transcriptions.create(**kwargs)
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

        if isinstance(response, str):
            return TranscriptionResult(text=response, raw=response)

        plain = Transcription.model_validate(response, from_attributes=True)
        return TranscriptionResult(text=plain.text, raw=plain)
