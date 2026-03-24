from .bytes_device import BytesAudioDevice
from .file import FileAudioDevice

__all__ = [
    "BytesAudioDevice",
    "FileAudioDevice",
    "MicrophoneAudioDevice",
]


def __getattr__(name: str):
    if name == "MicrophoneAudioDevice":
        from .mic import MicrophoneAudioDevice

        return MicrophoneAudioDevice
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
