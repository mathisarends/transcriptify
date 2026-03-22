import asyncio
from collections.abc import AsyncIterator
from typing import Self

from transcriptify.views import AudioChunk
from transcriptify.port.audio_device import AudioDevice


class BytesAudioDevice(AudioDevice):
    """Wraps raw bytes or an async queue as an audio source.

    Use the constructor for static byte buffers, or `from_stream()`
    for real-time ingestion (e.g. WebSocket frames).
    """

    def __init__(
        self,
        data: bytes,
        sample_rate: int = 16_000,
        encoding: str = "wav",
    ) -> None:
        self._data = data
        self._sample_rate = sample_rate
        self._encoding = encoding
        self._queue: asyncio.Queue[bytes | None] | None = None

    @classmethod
    def from_stream(
        cls,
        queue: asyncio.Queue[bytes | None],
        sample_rate: int = 16_000,
        encoding: str = "wav",
    ) -> Self:
        device = cls(data=b"", sample_rate=sample_rate, encoding=encoding)
        device._queue = queue
        return device

    async def read(self) -> AudioChunk:
        if self._queue is not None:
            data = await self._queue.get()
            if data is None:
                return AudioChunk(
                    data=b"",
                    sample_rate=self._sample_rate,
                    encoding=self._encoding,
                )
            return AudioChunk(
                data=data,
                sample_rate=self._sample_rate,
                encoding=self._encoding,
            )
        return AudioChunk(
            data=self._data,
            sample_rate=self._sample_rate,
            encoding=self._encoding,
        )

    async def stream(self) -> AsyncIterator[AudioChunk]:
        if self._queue is not None:
            while True:
                data = await self._queue.get()
                if data is None:
                    break
                yield AudioChunk(
                    data=data,
                    sample_rate=self._sample_rate,
                    encoding=self._encoding,
                )
        else:
            yield await self.read()
