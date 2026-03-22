# transcriptify

Pluggable transcription library with audio device abstraction. Async-first, zero mandatory dependencies beyond pydantic.

## Installation

```bash
pip install transcriptify[openai]
```

## Quick Start

```python
import asyncio
from transcriptify.adapters.openai import OpenAIWhisper, WhisperModel
from transcriptify.audio import FileAudioDevice

async def main():
    whisper = OpenAIWhisper(model=WhisperModel.WHISPER_1)

    async with FileAudioDevice("recording.wav") as device:
        audio = await device.read()
        result = await whisper.transcribe(audio)

    print(result.text)

asyncio.run(main())
```

## Azure OpenAI

```python
whisper = OpenAIWhisper(
    api_key="<azure-key>",
    azure_endpoint="https://<resource>.openai.azure.com",
    azure_deployment="whisper",
    api_version="2024-06-01",
)
```

## Available Models

| Enum                                  | Value                    |
| ------------------------------------- | ------------------------ |
| `WhisperModel.WHISPER_1`              | `whisper-1`              |
| `WhisperModel.GPT_4O_TRANSCRIBE`      | `gpt-4o-transcribe`      |
| `WhisperModel.GPT_4O_MINI_TRANSCRIBE` | `gpt-4o-mini-transcribe` |

## Response Formats

| Enum                          | Value                    |
| ----------------------------- | ------------------------ |
| `ResponseFormat.VERBOSE_JSON` | `verbose_json` (default) |
| `ResponseFormat.JSON`         | `json`                   |
| `ResponseFormat.TEXT`         | `text`                   |
| `ResponseFormat.SRT`          | `srt`                    |
| `ResponseFormat.VTT`          | `vtt`                    |

When using `VERBOSE_JSON`, the `result.raw` field contains a typed `VerboseTranscription` pydantic model with segments and words:

```python
from transcriptify.adapters.openai import VerboseTranscription

result = await whisper.transcribe(audio)

if isinstance(result.raw, VerboseTranscription):
    for seg in result.raw.segments:
        print(f"[{seg.start:.1f}s - {seg.end:.1f}s] {seg.text}")
    for word in result.raw.words:
        print(f"[{word.start:.1f}s - {word.end:.1f}s] {word.word}")
```

## Audio Devices

### FileAudioDevice

Reads from `.wav`, `.mp3`, or any audio file on disk.

```python
from transcriptify.audio import FileAudioDevice

# Read entire file as one chunk
device = FileAudioDevice("recording.wav")
audio = await device.read()

# Stream in chunks (wav only)
device = FileAudioDevice("recording.wav", chunk_duration_ms=3000)
async for chunk in device.stream():
    result = await whisper.transcribe(chunk)
```

### BytesAudioDevice

Wraps raw bytes or an `asyncio.Queue` for real-time ingestion (e.g. WebSocket).

```python
from transcriptify.audio import BytesAudioDevice

# Static bytes
device = BytesAudioDevice(data=raw_audio_bytes, encoding="wav")

# Streaming via queue
queue: asyncio.Queue[bytes | None] = asyncio.Queue()
device = BytesAudioDevice.from_stream(queue)

# Push audio frames, None to signal end
await queue.put(frame)
await queue.put(None)
```
