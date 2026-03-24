import asyncio
from collections.abc import AsyncIterator
import io
from typing import Self
import wave

from transcriptify.audio.port import AudioDevice
from transcriptify.views import AudioChunk

try:
    import sounddevice as sd
except ImportError as exc:
    raise ImportError("Install with: pip install transcriptify[mic]") from exc


class MicrophoneAudioDevice(AudioDevice):
    def __init__(
        self,
        sample_rate: int = 16_000,
        channels: int = 1,
        chunk_duration_ms: int = 100,
        device: int | str | None = None,
    ) -> None:
        self._sample_rate = sample_rate
        self._channels = channels
        self._chunk_frames = int(sample_rate * chunk_duration_ms / 1000)
        self._device = device
        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._stream: sd.RawInputStream | None = None

    async def __aenter__(self) -> Self:
        loop = asyncio.get_event_loop()

        def _callback(indata: bytes, frames: int, time, status) -> None:
            loop.call_soon_threadsafe(self._queue.put_nowait, bytes(indata))

        self._stream = sd.RawInputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype="int16",
            blocksize=self._chunk_frames,
            device=self._device,
            callback=_callback,
        )
        self._stream.start()
        return self

    async def __aexit__(self, *_) -> None:
        if self._stream:
            self._stream.stop()
            self._stream.close()
        await self._queue.put(None)

    async def record(self, seconds: float) -> AudioChunk:
        chunks: list[bytes] = []

        async def _collect() -> None:
            async for chunk in self.stream():
                chunks.append(chunk.data)

        collector = asyncio.create_task(_collect())
        await asyncio.sleep(seconds)
        await self._queue.put(None)
        await collector

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(self._channels)
            wf.setsampwidth(2)
            wf.setframerate(self._sample_rate)
            wf.writeframes(b"".join(chunks))
        buf.seek(0)

        return AudioChunk(
            data=buf.read(),
            sample_rate=self._sample_rate,
            channels=self._channels,
            encoding="pcm_s16le",
        )

    async def read(self) -> AudioChunk:
        data = await self._queue.get()
        return AudioChunk(
            data=data or b"",
            sample_rate=self._sample_rate,
            channels=self._channels,
            encoding="pcm_s16le",
        )

    async def stream(self) -> AsyncIterator[AudioChunk]:
        while True:
            data = await self._queue.get()
            if data is None:
                break
            yield AudioChunk(
                data=data,
                sample_rate=self._sample_rate,
                channels=self._channels,
                encoding="pcm_s16le",
            )
