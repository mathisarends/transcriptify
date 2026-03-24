import asyncio
import sys

from dotenv import load_dotenv

from transcriptify.adapters.openai import OpenAIWhisper, WhisperModel
from transcriptify.audio import MicrophoneAudioDevice

load_dotenv(override=True)

RECORD_SECONDS = int(sys.argv[1]) if len(sys.argv) > 1 else 5


async def main() -> None:
    whisper = OpenAIWhisper(model=WhisperModel.WHISPER_1)
    print(f"Recording for {RECORD_SECONDS}s...")

    async with MicrophoneAudioDevice(sample_rate=16_000) as mic:
        audio = await mic.record(seconds=RECORD_SECONDS)

    result = await whisper.transcribe(audio)
    print(f"Text: {result.text}")


if __name__ == "__main__":
    asyncio.run(main())
