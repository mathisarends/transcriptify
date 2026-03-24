# transcriptify

Pluggable, async-first transcription library for Python. Provides a unified interface over transcription backends (currently OpenAI Whisper / GPT-4o) with swappable audio sources (files, raw bytes, microphone).

## Installation

```bash
pip install transcriptify
```

### Optional dependencies

| Extra | Install command                  | What it adds                           |
| ----- | -------------------------------- | -------------------------------------- |
| `mic` | `pip install transcriptify[mic]` | Microphone recording via `sounddevice` |
| `dev` | `pip install transcriptify[dev]` | pytest, ruff, mypy, pre-commit         |

## Quickstart

### Transcribe a file

```python
import asyncio
from transcriptify import OpenAIWhisper
from transcriptify.audio import FileAudioDevice

async def main():
    whisper = OpenAIWhisper()  # uses OPENAI_API_KEY env var

    async with FileAudioDevice("recording.wav") as device:
        audio = await device.read()
        result = await whisper.transcribe(audio)

    print(result.text)

asyncio.run(main())
```

### Transcribe from microphone

Requires the `mic` extra (`pip install transcriptify[mic]`).

```python
import asyncio
from transcriptify import OpenAIWhisper
from transcriptify.audio import MicrophoneAudioDevice

async def main():
    whisper = OpenAIWhisper()

    async with MicrophoneAudioDevice(sample_rate=16_000) as mic:
        audio = await mic.record(seconds=5)

    result = await whisper.transcribe(audio)
    print(result.text)

asyncio.run(main())
```

### Streaming transcription

Models that support streaming (`gpt-4o-transcribe`, `gpt-4o-mini-transcribe`, `gpt-4o-transcribe-diarize`) yield incremental deltas:

```python
async with FileAudioDevice("recording.wav") as device:
    audio = await device.read()
    async for delta in whisper.stream(audio):
        print(delta.text, end="", flush=True)
```

### Transcribe raw bytes

```python
from transcriptify.audio import BytesAudioDevice

device = BytesAudioDevice(wav_bytes, sample_rate=16_000, encoding="wav")
audio = await device.read()
result = await whisper.transcribe(audio)
```

## OpenAIWhisper configuration

```python
from transcriptify import OpenAIWhisper, WhisperModel, ResponseFormat

whisper = OpenAIWhisper(
    api_key="sk-...",                            # or set OPENAI_API_KEY env var
    model=WhisperModel.GPT_4O_TRANSCRIBE,        # default
    language="en",                               # optional ISO-639-1
    response_format=ResponseFormat.VERBOSE_JSON,  # json, text, srt, vtt, verbose_json, diarized_json
    prompt="Technical discussion about Python",   # optional context hint
)
```

### Available models

| Enum                                     | Value                       | Streaming | Diarization |
| ---------------------------------------- | --------------------------- | --------- | ----------- |
| `WhisperModel.WHISPER_1`                 | `whisper-1`                 | No        | No          |
| `WhisperModel.GPT_4O_TRANSCRIBE`         | `gpt-4o-transcribe`         | Yes       | No          |
| `WhisperModel.GPT_4O_MINI_TRANSCRIBE`    | `gpt-4o-mini-transcribe`    | Yes       | No          |
| `WhisperModel.GPT_4O_TRANSCRIBE_DIARIZE` | `gpt-4o-transcribe-diarize` | Yes       | Yes         |

### Azure OpenAI

```python
whisper = OpenAIWhisper(
    azure_endpoint="https://your-resource.openai.azure.com",
    azure_deployment="your-whisper-deployment",
    api_key="your-azure-key",
    api_version="2024-06-01",  # optional, this is the default
)
```

## Audio devices

| Class                   | Import                | Description                                                                                                     |
| ----------------------- | --------------------- | --------------------------------------------------------------------------------------------------------------- |
| `FileAudioDevice`       | `transcriptify.audio` | Reads `.wav` (parsed) or any other format (raw bytes). Supports chunked streaming via `chunk_duration_ms`.      |
| `BytesAudioDevice`      | `transcriptify.audio` | Wraps raw bytes. Also supports real-time push via `BytesAudioDevice.from_stream()` + `push()`.                  |
| `MicrophoneAudioDevice` | `transcriptify.audio` | Records from system mic. Requires `transcriptify[mic]`. Supports `record(seconds=N)` and continuous `stream()`. |

## Implementing custom adapters

### Custom transcription backend

```python
from transcriptify import Transcriber, AudioChunk, TranscriptionResult

class MyTranscriber(Transcriber):
    async def transcribe(self, audio: AudioChunk) -> TranscriptionResult:
        text = await my_backend.recognize(audio.data)
        return TranscriptionResult(text=text)
```

### Custom audio device

```python
from transcriptify import AudioDevice, AudioChunk
from collections.abc import AsyncIterator

class MyAudioDevice(AudioDevice):
    async def read(self) -> AudioChunk:
        data = await my_source.get_audio()
        return AudioChunk(data=data, sample_rate=16_000)

    async def stream(self) -> AsyncIterator[AudioChunk]:
        async for frame in my_source.frames():
            yield AudioChunk(data=frame, sample_rate=16_000)
```

## Requirements

- Python >= 3.11
- `pydantic >= 2.12.5`
- `openai >= 2.29.0`
- `python-dotenv >= 1.2.2`
