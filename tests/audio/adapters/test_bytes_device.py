import pytest

from transcriptify.audio.adapters.bytes_device import BytesAudioDevice


class TestBytesAudioDeviceRead:
    @pytest.mark.asyncio
    async def test_returns_chunk_with_provided_bytes(self) -> None:
        data = b"\x01\x02\x03\x04"
        device = BytesAudioDevice(data=data)
        chunk = await device.read()
        assert chunk.data == data

    @pytest.mark.asyncio
    async def test_default_sample_rate_is_16k(self) -> None:
        device = BytesAudioDevice(data=b"audio")
        chunk = await device.read()
        assert chunk.sample_rate == 16_000

    @pytest.mark.asyncio
    async def test_custom_sample_rate_is_preserved(self) -> None:
        device = BytesAudioDevice(data=b"audio", sample_rate=44_100)
        chunk = await device.read()
        assert chunk.sample_rate == 44_100

    @pytest.mark.asyncio
    async def test_encoding_is_passed_through(self) -> None:
        device = BytesAudioDevice(data=b"audio", encoding="mp3")
        chunk = await device.read()
        assert chunk.encoding == "mp3"


class TestBytesAudioDeviceStream:
    @pytest.mark.asyncio
    async def test_static_device_yields_single_chunk(self) -> None:
        device = BytesAudioDevice(data=b"audio bytes")
        chunks = [chunk async for chunk in device.stream()]
        assert len(chunks) == 1
        assert chunks[0].data == b"audio bytes"

    @pytest.mark.asyncio
    async def test_static_device_chunk_has_correct_sample_rate(self) -> None:
        device = BytesAudioDevice(data=b"audio", sample_rate=8_000)
        chunks = [chunk async for chunk in device.stream()]
        assert chunks[0].sample_rate == 8_000


class TestBytesAudioDeviceContextManager:
    @pytest.mark.asyncio
    async def test_usable_as_async_context_manager(self) -> None:
        async with BytesAudioDevice(data=b"audio") as device:
            chunk = await device.read()
        assert chunk.data == b"audio"


class TestBytesAudioDeviceFromStream:
    @pytest.mark.asyncio
    async def test_push_then_read_returns_pushed_chunk(self) -> None:
        device = BytesAudioDevice.from_stream()
        await device.push(b"frame1")
        chunk = await device.read()
        assert chunk.data == b"frame1"

    @pytest.mark.asyncio
    async def test_push_none_signals_eof_in_stream(self) -> None:
        device = BytesAudioDevice.from_stream()
        await device.push(b"first")
        await device.push(b"second")
        await device.push(None)

        chunks = [c async for c in device.stream()]
        assert len(chunks) == 2
        assert chunks[0].data == b"first"
        assert chunks[1].data == b"second"

    @pytest.mark.asyncio
    async def test_eof_sentinel_produces_empty_chunk_via_read(self) -> None:
        device = BytesAudioDevice.from_stream()
        await device.push(None)
        chunk = await device.read()
        assert chunk.data == b""

    @pytest.mark.asyncio
    async def test_push_on_static_device_raises_runtime_error(self) -> None:
        device = BytesAudioDevice(data=b"static")
        with pytest.raises(RuntimeError, match="push()"):
            await device.push(b"data")

    def test_from_stream_uses_default_sample_rate(self) -> None:
        device = BytesAudioDevice.from_stream()
        assert device._sample_rate == 16_000

    def test_from_stream_uses_custom_encoding(self) -> None:
        device = BytesAudioDevice.from_stream(encoding="pcm_s16le")
        assert device._encoding == "pcm_s16le"
