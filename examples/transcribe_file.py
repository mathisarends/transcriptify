import asyncio
import sys
from pathlib import Path

from transcriptify import OpenAIWhisper, VerboseTranscription
from transcriptify.audio import FileAudioDevice

from dotenv import load_dotenv

load_dotenv(override=True)


async def main() -> None:
    path = (
        Path(sys.argv[1]) if len(sys.argv) >= 2 else Path(__file__).parent / "audio.wav"
    )
    whisper = OpenAIWhisper()

    async with FileAudioDevice(path) as device:
        audio = await device.read()
        result = await whisper.transcribe(audio)

    print(f"Language: {result.language}")
    print(f"Text: {result.text}")

    if isinstance(result.raw, VerboseTranscription):
        for seg in result.raw.segments:
            print(f"  [{seg.start:.1f}s - {seg.end:.1f}s] {seg.text}")
        if result.raw.words:
            for word in result.raw.words:
                print(f"  [{word.start:.1f}s - {word.end:.1f}s] {word.word}")


if __name__ == "__main__":
    asyncio.run(main())
