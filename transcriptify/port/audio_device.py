from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Self

from transcriptify.views import AudioChunk


class AudioDevice(ABC):
    """Abstract audio source.

    Implementations can be file-based, byte-stream-based,
    microphone-based, or any other audio input.
    """

    @abstractmethod
    async def read(self) -> AudioChunk:
        ...

    @abstractmethod
    def stream(self) -> AsyncIterator[AudioChunk]:
        ...

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        pass
