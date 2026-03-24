import wave
from pathlib import Path

import pytest

from transcriptify.audio.adapters.file import FileAudioDevice


def _write_wav(path: Path, num_frames: int = 16_000, sample_rate: int = 16_000) -> Path:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * num_frames)
    return path


class TestFileAudioDeviceRead:
    @pytest.mark.asyncio
    async def test_wav_file_returns_chunk_with_correct_metadata(
        self, tmp_path: Path
    ) -> None:
        wav = _write_wav(tmp_path / "audio.wav", num_frames=8_000, sample_rate=8_000)
        device = FileAudioDevice(wav)
        chunk = await device.read()

        assert chunk.data == wav.read_bytes()
        assert chunk.sample_rate == 8_000
        assert chunk.channels == 1
        assert chunk.encoding == "pcm_s16le"

    @pytest.mark.asyncio
    async def test_non_wav_file_uses_extension_as_encoding(
        self, tmp_path: Path
    ) -> None:
        mp3 = tmp_path / "audio.mp3"
        mp3.write_bytes(b"fake mp3 data")
        device = FileAudioDevice(mp3)
        chunk = await device.read()

        assert chunk.data == b"fake mp3 data"
        assert chunk.encoding == "mp3"

    @pytest.mark.asyncio
    async def test_non_wav_file_uses_default_sample_rate(self, tmp_path: Path) -> None:
        ogg = tmp_path / "audio.ogg"
        ogg.write_bytes(b"fake ogg data")
        device = FileAudioDevice(ogg)
        chunk = await device.read()

        assert chunk.sample_rate == 16_000

    @pytest.mark.asyncio
    async def test_accepts_string_path(self, tmp_path: Path) -> None:
        wav = _write_wav(tmp_path / "audio.wav")
        device = FileAudioDevice(str(wav))
        chunk = await device.read()

        assert len(chunk.data) > 0


class TestFileAudioDeviceStream:
    @pytest.mark.asyncio
    async def test_zero_chunk_duration_yields_entire_file_as_one_chunk(
        self, tmp_path: Path
    ) -> None:
        wav = _write_wav(tmp_path / "audio.wav", num_frames=16_000)
        device = FileAudioDevice(wav, chunk_duration_ms=0)
        chunks = [c async for c in device.stream()]

        assert len(chunks) == 1
        assert chunks[0].data == wav.read_bytes()

    @pytest.mark.asyncio
    async def test_wav_with_chunk_duration_splits_into_multiple_chunks(
        self, tmp_path: Path
    ) -> None:
        wav = _write_wav(tmp_path / "audio.wav", num_frames=16_000, sample_rate=16_000)
        device = FileAudioDevice(wav, chunk_duration_ms=500)
        chunks = [c async for c in device.stream()]

        assert len(chunks) == 2

    @pytest.mark.asyncio
    async def test_chunks_carry_correct_sample_rate(self, tmp_path: Path) -> None:
        wav = _write_wav(tmp_path / "audio.wav", num_frames=16_000, sample_rate=16_000)
        device = FileAudioDevice(wav, chunk_duration_ms=500)
        chunks = [c async for c in device.stream()]

        for chunk in chunks:
            assert chunk.sample_rate == 16_000

    @pytest.mark.asyncio
    async def test_chunks_carry_pcm_encoding(self, tmp_path: Path) -> None:
        wav = _write_wav(tmp_path / "audio.wav", num_frames=16_000, sample_rate=16_000)
        device = FileAudioDevice(wav, chunk_duration_ms=500)
        chunks = [c async for c in device.stream()]

        for chunk in chunks:
            assert chunk.encoding == "pcm_s16le"

    @pytest.mark.asyncio
    async def test_non_wav_with_chunk_duration_falls_back_to_single_chunk(
        self, tmp_path: Path
    ) -> None:
        flac = tmp_path / "audio.flac"
        flac.write_bytes(b"fake flac data")
        device = FileAudioDevice(flac, chunk_duration_ms=200)
        chunks = [c async for c in device.stream()]

        assert len(chunks) == 1
        assert chunks[0].data == b"fake flac data"

    @pytest.mark.asyncio
    async def test_stream_chunk_data_covers_full_audio(self, tmp_path: Path) -> None:
        wav = _write_wav(tmp_path / "audio.wav", num_frames=16_000, sample_rate=16_000)
        device = FileAudioDevice(wav, chunk_duration_ms=500)
        chunks = [c async for c in device.stream()]

        total_bytes = sum(len(c.data) for c in chunks)
        expected_bytes = 16_000 * 2
        assert total_bytes == expected_bytes


class TestFileAudioDeviceContextManager:
    @pytest.mark.asyncio
    async def test_usable_as_async_context_manager(self, tmp_path: Path) -> None:
        wav = _write_wav(tmp_path / "audio.wav")
        async with FileAudioDevice(wav) as device:
            chunk = await device.read()
        assert len(chunk.data) > 0
