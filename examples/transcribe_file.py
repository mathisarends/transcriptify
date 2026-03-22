import asyncio
import os
import sys

from transcriptify.adapters.openai import OpenAIWhisper, WhisperModel, VerboseTranscription
from transcriptify.audio import FileAudioDevice


async def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <audio-file> [--azure]")
        sys.exit(1)

    path = sys.argv[1]
    use_azure = "--azure" in sys.argv
    api_key = os.environ.get("OPENAI_API_KEY")

    if use_azure:
        whisper = OpenAIWhisper(
            api_key=api_key
        )
    else:
        whisper = OpenAIWhisper(api_key=api_key, model=WhisperModel.WHISPER_1)

    async with FileAudioDevice(path) as device:
        audio = await device.read()
        result = await whisper.transcribe(audio)

    print(f"Language: {result.language}")
    print(f"Text: {result.text}")

    if isinstance(result.raw, VerboseTranscription):
        for seg in result.raw.segments:
            print(f"  [{seg.start:.1f}s - {seg.end:.1f}s] {seg.text}")
        for word in result.raw.words:
            print(f"  [{word.start:.1f}s - {word.end:.1f}s] {word.word}")


if __name__ == "__main__":
    asyncio.run(main())
