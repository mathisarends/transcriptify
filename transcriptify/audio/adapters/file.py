import asyncio
import wave
from pathlib import Path
from collections.abc import AsyncIterator

from transcriptify.views import AudioChunk
from transcriptify.audio.port import AudioDevice


class FileAudioDevice(AudioDevice):
    """Reads audio from a file on disk.

    Supports .wav natively. For other formats, the raw bytes are passed
    through with the encoding set to the file extension.
    """

    def __init__(self, path: str | Path, chunk_duration_ms: int = 0) -> None:
        self._path = Path(path)
        self._chunk_duration_ms = chunk_duration_ms

    def _read_wav(self) -> AudioChunk:
        with wave.open(str(self._path), "rb") as wf:
            return AudioChunk(
                data=self._path.read_bytes(),
                sample_rate=wf.getframerate(),
                channels=wf.getnchannels(),
                encoding="pcm_s16le",
            )

    def _read_raw(self) -> AudioChunk:
        data = self._path.read_bytes()
        ext = self._path.suffix.lstrip(".").lower() or "wav"
        return AudioChunk(data=data, encoding=ext)

    async def read(self) -> AudioChunk:
        if self._path.suffix.lower() == ".wav":
            return await asyncio.to_thread(self._read_wav)
        return await asyncio.to_thread(self._read_raw)

    async def stream(self) -> AsyncIterator[AudioChunk]:
        if self._chunk_duration_ms <= 0:
            yield await self.read()
            return

        if self._path.suffix.lower() != ".wav":
            yield await self.read()
            return

        def _read_chunks() -> list[AudioChunk]:
            chunks: list[AudioChunk] = []
            with wave.open(str(self._path), "rb") as wf:
                sample_rate = wf.getframerate()
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                frames_per_chunk = int(sample_rate * self._chunk_duration_ms / 1000)
                bytes_per_frame = channels * sample_width

                while True:
                    data = wf.readframes(frames_per_chunk)
                    if not data or len(data) < bytes_per_frame:
                        break
                    chunks.append(
                        AudioChunk(
                            data=data,
                            sample_rate=sample_rate,
                            channels=channels,
                            encoding="pcm_s16le",
                        )
                    )
            return chunks

        for chunk in await asyncio.to_thread(_read_chunks):
            yield chunk
