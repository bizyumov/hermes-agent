"""Authorized Discord PCM must reach native STT before a completed WAV (yandex-speech #43)."""

import asyncio
import contextvars
import struct
import threading
from types import SimpleNamespace

import pytest


class InnerSession:
    def __init__(self):
        self.received = bytearray()
        self.fed = threading.Event()
        self.cancelled = threading.Event()

    def push_audio(self, data):
        self.received.extend(data)
        self.fed.set()

    def partial_transcript(self):
        return "preview" if self.received else ""

    def finalize(self):
        return {"success": True, "provider": "fixture", "transcript": "complete tail"}

    def cancel(self):
        self.cancelled.set()


def adapter(allowed=True):
    return SimpleNamespace(
        _owner_profile="fixture-profile", _voice_text_channels={1: 22},
        _voice_sources={1: {"chat_id": "22", "profile": "fixture-profile"}},
        _client=SimpleNamespace(get_guild=lambda guild: object()),
        _is_allowed_user=lambda user, **kwargs: allowed,
    )


def frame(sequence, timestamp, *, source=(17, 0)):
    from plugins.platforms.discord.voice_streaming_input import PCMFrame
    import time
    return PCMFrame(source, 7, struct.pack("<hh", 600, -300) * 960,
                    sequence, timestamp, time.monotonic())


@pytest.mark.asyncio
async def test_source_feeds_during_capture_and_one_final_tail_wins(monkeypatch):
    from plugins.platforms.discord import voice_streaming_input as live

    inner = InnerSession()
    opened_scope = []
    scope = contextvars.ContextVar("source-owner", default="foreign")
    scope.set("owning-profile")
    monkeypatch.setattr(live, "open_native_stream_session",
                        lambda: opened_scope.append(scope.get()) or inner)
    ingress = live.VoiceStreamIngress(adapter(), 1)
    scope.set("foreign")
    ingress.offer(frame(65535, 0xFFFFFC40))
    ingress._drain()
    assert await asyncio.to_thread(inner.fed.wait, 2)
    assert opened_scope == ["owning-profile"]
    assert inner.received == struct.pack("<h", 150) * 320
    ingress.offer(frame(0, 0))
    result = await ingress.finish_source((17, 0), 7, b"x" * 7680, 22)
    assert result["success"] and result["transcript"] == "complete tail"
    assert result["stream_source"]["input_bytes"] == 7680
    assert result["stream_source"]["frames"] == 2
    assert result["stream_source"]["partial_events"] >= 1
    assert (await ingress.finish_source((17, 0), 7, b"x" * 7680, 22))["stale"] is True
    ingress.close()


@pytest.mark.asyncio
async def test_authorization_and_binding_move_prevent_foreign_or_stale_input(monkeypatch):
    from plugins.platforms.discord import voice_streaming_input as live

    opens = []
    monkeypatch.setattr(live, "open_native_stream_session", lambda: opens.append(1) or InnerSession())
    blocked = live.VoiceStreamIngress(adapter(allowed=False), 1)
    blocked.offer(frame(1, 0)); blocked._drain()
    assert opens == []
    assert (await blocked.finish_source((17, 0), 7, b"x" * 3840, 22))["stale"] is True
    blocked.close()

    owner = adapter()
    inner = InnerSession()
    monkeypatch.setattr(live, "open_native_stream_session", lambda: inner)
    ingress = live.VoiceStreamIngress(owner, 1)
    ingress.offer(frame(1, 0)); ingress._drain()
    assert await asyncio.to_thread(inner.fed.wait, 2)
    owner._voice_text_channels[1] = 33
    ingress.rebind()
    result = await ingress.finish_source((17, 0), 7, b"x" * 3840, 22)
    assert result["stale"] is True
    assert inner.cancelled.is_set()
    ingress.close()


@pytest.mark.asyncio
async def test_capture_gap_never_becomes_a_successful_native_completion(monkeypatch):
    from plugins.platforms.discord import voice_streaming_input as live

    inner = InnerSession()
    monkeypatch.setattr(live, "open_native_stream_session", lambda: inner)
    ingress = live.VoiceStreamIngress(adapter(), 1)
    ingress.offer(frame(10, 0)); ingress.offer(frame(12, 1920))
    result = await ingress.finish_source((17, 0), 7, b"x" * 7680, 22)
    assert result["success"] is False
    assert result["stream_source"]["gap"] is True
    assert inner.cancelled.is_set()
    ingress.close()


@pytest.mark.asyncio
async def test_native_decline_is_explicit_and_source_buffer_loss_requires_file_recovery(monkeypatch):
    from plugins.platforms.discord import voice_streaming_input as live

    monkeypatch.setattr(live, "open_native_stream_session", lambda: None)
    ingress = live.VoiceStreamIngress(adapter(), 1)
    ingress.offer(frame(1, 0)); ingress._drain()
    assert await ingress.finish_source((17, 0), 7, b"x" * 3840, 22) is None
    assert ingress.last_status["route"] == "declined"
    ingress.close()

    monkeypatch.setattr(live, "open_native_stream_session", lambda: InnerSession())
    ingress = live.VoiceStreamIngress(adapter(), 1)
    ingress.offer(frame(1, 0))
    result = await ingress.finish_source((17, 0), 7, b"x" * 7680, 22)
    assert result["success"] is False
    assert result["stream_source"]["input_bytes"] == 3840
    ingress.close()


@pytest.mark.asyncio
async def test_real_receiver_packet_hook_feeds_native_before_silence_and_never_runs_file_twice(monkeypatch):
    import time
    from unittest.mock import AsyncMock
    from nacl.secret import Aead
    from plugins.platforms.discord import adapter as discord_adapter
    from plugins.platforms.discord import voice_streaming_input as live

    inner = InnerSession()
    monkeypatch.setattr(live, "open_native_stream_session", lambda: inner)
    pcm = struct.pack("<hh", 600, -300) * 960
    monkeypatch.setattr(discord_adapter.discord.opus, "Decoder",
                        lambda: SimpleNamespace(decode=lambda packet: pcm))
    owner = adapter()
    owner._voice_input_callback = AsyncMock()
    owner._reset_voice_timeout = lambda guild: None
    owner._process_voice_input = lambda *args, **kwargs: live.process_voice_input(owner, *args, **kwargs)
    ingress = live.VoiceStreamIngress(owner, 1)
    owner._voice_live_ingress = {1: ingress}
    receiver = discord_adapter.VoiceReceiver(SimpleNamespace(), on_pcm=ingress.offer)
    key = bytes(range(32))
    receiver._creds = (key, None, 0, False)
    receiver._running = True
    receiver.map_ssrc(17, 7)
    box = Aead(key)

    def send(index):
        nonce_tail = index.to_bytes(4, "big")
        nonce = nonce_tail + bytes(20)
        header = struct.pack(">BBHII", 0x80, 0x78, index, (index - 1) * 960, 17)
        encrypted = box.encrypt(b"synthetic-opus-frame", header, nonce).ciphertext
        receiver._on_packet(header + encrypted + nonce_tail)

    def forbid_file(*args, **kwargs):
        raise AssertionError("native completion must not also invoke file transcription")

    monkeypatch.setattr(discord_adapter.VoiceReceiver, "pcm_to_wav", forbid_file)
    send(1); ingress._drain()
    assert await asyncio.to_thread(inner.fed.wait, 2)
    assert owner._voice_input_callback.await_count == 0
    for index in range(2, 31):
        send(index)
    receiver._last_packet_time[17] = time.monotonic() - 2 * receiver.SILENCE_THRESHOLD
    completed = receiver.check_silence()
    assert receiver._completed_source_ids == ((17, 0),)
    await live.process_completed_inputs(owner, 1, receiver, completed, 22)
    assert inner.received == struct.pack("<h", 150) * (320 * 30)
    assert owner._voice_input_callback.await_count == 1
    delivered = owner._voice_input_callback.await_args.kwargs
    assert {key: delivered[key] for key in ("guild_id", "user_id", "transcript")} == {
        "guild_id": 1, "user_id": 7, "transcript": "complete tail"}
    assert delivered["source_is_current"]() is True
    assert ingress.last_status["route"] == "native"
    ingress.close()


@pytest.mark.asyncio
async def test_cancel_while_provider_opens_closes_the_late_wire(monkeypatch):
    from plugins.platforms.discord import voice_streaming_input as live

    entered, release = threading.Event(), threading.Event()
    inner = InnerSession()

    def opening():
        entered.set()
        assert release.wait(2)
        return inner

    monkeypatch.setattr(live, "open_native_stream_session", opening)
    ingress = live.VoiceStreamIngress(adapter(), 1)
    ingress.offer(frame(1, 0)); ingress._drain()
    assert await asyncio.to_thread(entered.wait, 2)
    ingress.close()
    release.set()
    assert await asyncio.to_thread(inner.cancelled.wait, 2)


@pytest.mark.asyncio
async def test_binding_move_during_final_drain_discards_the_late_success(monkeypatch):
    from plugins.platforms.discord import voice_streaming_input as live

    finalizing, release = threading.Event(), threading.Event()

    class DelayedFinal(InnerSession):
        def finalize(self):
            finalizing.set()
            assert release.wait(2)
            return super().finalize()

    inner, owner = DelayedFinal(), adapter()
    monkeypatch.setattr(live, "open_native_stream_session", lambda: inner)
    ingress = live.VoiceStreamIngress(owner, 1)
    ingress.offer(frame(1, 0))
    pending = asyncio.create_task(ingress.finish_source((17, 0), 7, b"x" * 3840, 22))
    assert await asyncio.to_thread(finalizing.wait, 2)
    owner._voice_text_channels[1] = 33
    ingress.rebind()
    release.set()
    assert (await pending)["stale"] is True
    ingress.close()


@pytest.mark.asyncio
async def test_native_queue_overrun_is_bounded_and_retained_file_remains_recoverable(monkeypatch):
    from plugins.platforms.discord import voice_streaming_input as live

    entered, release = threading.Event(), threading.Event()

    class SlowProvider(InnerSession):
        def push_audio(self, data):
            entered.set()
            assert release.wait(2)
            super().push_audio(data)

        def cancel(self):
            super().cancel()
            release.set()

    inner = SlowProvider()
    monkeypatch.setattr(live, "open_native_stream_session", lambda: inner)
    ingress = live.VoiceStreamIngress(adapter(), 1)
    ingress.offer(frame(0, 0)); ingress._drain()
    assert await asyncio.to_thread(entered.wait, 2)
    for index in range(1, 500):
        ingress.offer(frame(index, index * 960))
        ingress._drain()
    result = await ingress.finish_source((17, 0), 7, b"x" * (500 * 3840), 22)
    assert result["success"] is False
    assert result["worker_pending"] is False
    assert result["stream_source"]["capture_gap"] is False
    assert inner.cancelled.is_set()
    ingress.close()


@pytest.mark.asyncio
async def test_known_missing_source_audio_is_never_dispatched_as_a_complete_file(monkeypatch):
    from unittest.mock import AsyncMock
    from plugins.platforms.discord import adapter as discord_adapter
    from plugins.platforms.discord import voice_streaming_input as live

    owner = adapter()
    owner._voice_input_callback = AsyncMock()
    inner = InnerSession()
    monkeypatch.setattr(live, "open_native_stream_session", lambda: inner)
    ingress = live.VoiceStreamIngress(owner, 1)
    ingress.offer(frame(1, 0)); ingress.offer(frame(3, 1920))
    result = await ingress.finish_source((17, 0), 7, b"x" * 7680, 22)
    assert result["stream_source"]["capture_gap"] is True

    def forbid_file(*args, **kwargs):
        raise AssertionError("a retained file cannot repair missing transport audio")

    monkeypatch.setattr(discord_adapter.VoiceReceiver, "pcm_to_wav", forbid_file)
    await live.process_voice_input(owner, 1, 7, b"x" * 7680, 22, stream_result=result)
    assert owner._voice_input_callback.await_count == 0
    assert ingress.last_status["route"] == "incomplete"
    ingress.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("offer_before_rebind", [True, False])
async def test_pending_and_delayed_old_frames_never_open_in_a_new_binding(monkeypatch, offer_before_rebind):
    from plugins.platforms.discord import voice_streaming_input as live

    opened = []
    monkeypatch.setattr(live, "open_native_stream_session", lambda: opened.append(1) or InnerSession())
    owner = adapter()
    ingress = live.VoiceStreamIngress(owner, 1)
    old_frame = frame(1, 0)
    if offer_before_rebind:
        ingress.offer(old_frame)
    owner._voice_text_channels[1] = 33
    ingress.rebind()
    if not offer_before_rebind:
        ingress.offer(old_frame)
    result = await ingress.finish_source((17, 0), 7, b"x" * 3840, 33)
    try:
        assert result["stale"] is True
        assert opened == []
    finally:
        ingress.close()


@pytest.mark.asyncio
async def test_discarded_short_captures_retire_wires_and_leave_capacity_for_speech(monkeypatch):
    import queue
    import time
    from plugins.platforms.discord import adapter as discord_adapter
    from plugins.platforms.discord import voice_streaming_input as live

    opened, sessions = queue.Queue(), []

    def opening():
        inner = InnerSession()
        sessions.append(inner)
        opened.put(inner)
        return inner

    monkeypatch.setattr(live, "open_native_stream_session", opening)
    owner = adapter()
    owner._reset_voice_timeout = lambda guild: None
    ingress = live.VoiceStreamIngress(owner, 1)
    owner._voice_live_ingress = {1: ingress}
    receiver = discord_adapter.VoiceReceiver(SimpleNamespace(), on_pcm=ingress.offer)
    receiver._running = True
    receiver.map_ssrc(17, 7)
    pcm = struct.pack("<hh", 600, -300) * 960
    try:
        for index in range(8):
            receiver._accept_pcm(17, pcm, index, index * 960)
            ingress._drain()
            inner = await asyncio.to_thread(opened.get, True, 2)
            assert await asyncio.to_thread(inner.fed.wait, 2)
            receiver._last_packet_time[17] = time.monotonic() - 3 * receiver.SILENCE_THRESHOLD
            completed = receiver.check_silence()
            assert completed == []
            await live.process_completed_inputs(owner, 1, receiver, completed, 22)
        for index in range(8, 38):
            receiver._accept_pcm(17, pcm, index, index * 960)
        receiver._last_packet_time[17] = time.monotonic() - 3 * receiver.SILENCE_THRESHOLD
        completed = receiver.check_silence()
        source_id = receiver._completed_source_ids[0]
        result = await ingress.finish_source(source_id, 7, completed[0][1], 22)
        assert result is not None and result["success"] is True
        assert all(inner.cancelled.is_set() for inner in sessions[:8])
    finally:
        ingress.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["close", "rebind", "cancel_waiter"])
async def test_finalizing_wire_remains_owned_by_every_cancellation_path(monkeypatch, action):
    import contextlib
    from plugins.platforms.discord import voice_streaming_input as live

    finalizing, release = threading.Event(), threading.Event()

    class FinalizingProvider(InnerSession):
        def finalize(self):
            finalizing.set()
            assert release.wait(5)
            return super().finalize()

        def cancel(self):
            super().cancel()
            release.set()

    inner, owner = FinalizingProvider(), adapter()
    monkeypatch.setattr(live, "open_native_stream_session", lambda: inner)
    ingress = live.VoiceStreamIngress(owner, 1)
    ingress.offer(frame(1, 0))
    pending = asyncio.create_task(ingress.finish_source((17, 0), 7, b"x" * 3840, 22))
    try:
        assert await asyncio.to_thread(finalizing.wait, 2)
        if action == "close":
            ingress.close()
        elif action == "rebind":
            owner._voice_text_channels[1] = 33
            ingress.rebind()
        else:
            pending.cancel()
        assert await asyncio.to_thread(inner.cancelled.wait, 2)
    finally:
        release.set()
        with contextlib.suppress(asyncio.CancelledError):
            await pending
        ingress.close()


@pytest.mark.asyncio
async def test_provider_failure_never_masks_a_later_capture_gap(monkeypatch):
    from plugins.platforms.discord import voice_streaming_input as live

    failed = threading.Event()

    class FailingProvider(InnerSession):
        def push_audio(self, data):
            failed.set()
            raise ValueError("synthetic provider failure")

    monkeypatch.setattr(live, "open_native_stream_session", lambda: FailingProvider())
    ingress = live.VoiceStreamIngress(adapter(), 1)
    try:
        ingress.offer(frame(1, 0)); ingress._drain()
        assert await asyncio.to_thread(failed.wait, 2)
        ingress.offer(frame(3, 1920))
        result = await ingress.finish_source((17, 0), 7, b"x" * 7680, 22)
        assert result["success"] is False
        assert result["stream_source"]["capture_gap"] is True
        assert ingress.last_status["route"] == "incomplete"
    finally:
        ingress.close()


@pytest.mark.asyncio
async def test_retained_file_recovery_uses_the_captured_profile_context(monkeypatch):
    from unittest.mock import AsyncMock
    from plugins.platforms.discord import adapter as discord_adapter
    from plugins.platforms.discord import voice_streaming_input as live
    from tools import transcription_tools

    scope = contextvars.ContextVar("file-recovery-profile", default="foreign")
    scope.set("owning-profile")
    owner = adapter()
    owner._voice_input_callback = AsyncMock()
    owner._reset_voice_timeout = lambda guild: None
    owner._process_voice_input = lambda *args, **kwargs: live.process_voice_input(owner, *args, **kwargs)
    ingress = live.VoiceStreamIngress(owner, 1)
    owner._voice_live_ingress = {1: ingress}
    monkeypatch.setattr(live, "open_native_stream_session", lambda: None)
    seen = []

    def transcribe(path):
        seen.append(scope.get())
        return {"success": True, "transcript": "file completion"}

    monkeypatch.setattr(transcription_tools, "transcribe_audio", transcribe)
    monkeypatch.setattr(discord_adapter.VoiceReceiver, "pcm_to_wav", lambda pcm, path: None)
    ingress.offer(frame(1, 0))
    scope.set("foreign")
    receiver = SimpleNamespace(_completed_source_ids=((17, 0),), _discarded_source_ids=())
    await live.process_completed_inputs(owner, 1, receiver, [(7, b"x" * 3840)], 22)
    assert seen == ["owning-profile"]
    assert owner._voice_input_callback.await_count == 1
    ingress.close()


@pytest.mark.asyncio
async def test_binding_move_during_retained_file_conversion_prevents_provider_submission(monkeypatch):
    from unittest.mock import AsyncMock
    from plugins.platforms.discord import adapter as discord_adapter
    from plugins.platforms.discord import voice_streaming_input as live
    from tools import transcription_tools

    converting, release = threading.Event(), threading.Event()
    owner = adapter()
    owner._voice_input_callback = AsyncMock()
    owner._reset_voice_timeout = lambda guild: None
    owner._process_voice_input = lambda *args, **kwargs: live.process_voice_input(owner, *args, **kwargs)
    ingress = live.VoiceStreamIngress(owner, 1)
    owner._voice_live_ingress = {1: ingress}
    monkeypatch.setattr(live, "open_native_stream_session", lambda: None)

    def convert(pcm, path):
        converting.set()
        assert release.wait(5)

    calls = []
    monkeypatch.setattr(discord_adapter.VoiceReceiver, "pcm_to_wav", convert)
    monkeypatch.setattr(transcription_tools, "transcribe_audio", lambda path: calls.append(path) or {})
    ingress.offer(frame(1, 0))
    receiver = SimpleNamespace(_completed_source_ids=((17, 0),), _discarded_source_ids=())
    pending = asyncio.create_task(live.process_completed_inputs(owner, 1, receiver, [(7, b"x" * 3840)], 22))
    try:
        assert await asyncio.to_thread(converting.wait, 2)
        owner._voice_text_channels[1] = 33
        ingress.rebind()
        release.set()
        await pending
        assert calls == []
        assert owner._voice_input_callback.await_count == 0
    finally:
        release.set()
        await pending
        ingress.close()


@pytest.mark.asyncio
async def test_inferred_legacy_ssrc_mapping_is_not_native_source_authorization(monkeypatch):
    from plugins.platforms.discord import adapter as discord_adapter
    from plugins.platforms.discord import voice_streaming_input as live

    opened = []
    monkeypatch.setattr(live, "open_native_stream_session", lambda: opened.append(1) or InnerSession())
    ingress = live.VoiceStreamIngress(adapter(), 1)
    vc = SimpleNamespace(channel=SimpleNamespace(members=[SimpleNamespace(id=7)]),
                         user=SimpleNamespace(id=1))
    receiver = discord_adapter.VoiceReceiver(vc, allowed_user_ids={"7"}, on_pcm=ingress.offer)
    receiver._running = True
    assert receiver._infer_user_for_ssrc(17) == 7
    receiver._accept_pcm(17, struct.pack("<hh", 600, -300) * 960, 1, 0)
    ingress._drain()
    assert opened == []
    ingress.close()
