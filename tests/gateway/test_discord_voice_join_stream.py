"""A slash voice join binds capture before the first native streaming frame."""
from __future__ import annotations

import asyncio
import struct
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import SessionSource
from gateway.run_voice import GatewayVoiceMixin
from plugins.platforms.discord import adapter as discord_adapter
from plugins.platforms.discord import voice_streaming_input as live


class _NativeSession:
    def __init__(self):
        self.fed = threading.Event()
        self.received = bytearray()

    def push_audio(self, data):
        self.received.extend(data)
        self.fed.set()

    def partial_transcript(self):
        return "preview" if self.received else ""

    def finalize(self):
        return {"success": True, "transcript": "complete control tail"}

    def cancel(self):
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize("initial_text_channel", [None, 22, 33])
async def test_slash_join_streams_into_its_bound_conversation(
    monkeypatch, initial_text_channel,
):
    inner = _NativeSession()
    monkeypatch.setattr(live, "open_native_stream_session", lambda: inner)
    monkeypatch.setattr(discord_adapter, "DISCORD_AVAILABLE", True)
    monkeypatch.setattr(discord_adapter.VoiceReceiver, "start",
                        lambda receiver: setattr(receiver, "_running", True))

    owner = discord_adapter.DiscordAdapter(PlatformConfig(enabled=True, token="fake"))
    owner._owner_profile = "fixture-profile"
    owner._client = SimpleNamespace(get_guild=lambda guild: object())
    owner._is_allowed_user = lambda *args, **kwargs: True
    owner._reset_voice_timeout = lambda guild: None
    owner._voice_listen_loop = AsyncMock()
    owner._voice_fx_cfg = {"enabled": False}
    channel = SimpleNamespace(guild=SimpleNamespace(id=1), id=10, name="fixture")
    voice_client = SimpleNamespace(channel=channel, is_connected=lambda: True)
    channel.connect = AsyncMock(return_value=voice_client)
    owner.get_user_voice_channel = AsyncMock(return_value=channel)
    if initial_text_channel is not None:
        owner._voice_clients[1] = voice_client
        owner._bind_voice_text_channel(
            1, initial_text_channel,
            {"chat_id": str(initial_text_channel), "profile": "fixture-profile"},
        )
        ingress = live.VoiceStreamIngress(owner, 1)
        owner._voice_live_ingress[1] = ingress
        receiver = discord_adapter.VoiceReceiver(voice_client, on_pcm=ingress.offer)
        receiver._running = True
        owner._voice_receivers[1] = receiver

    runner = object.__new__(GatewayVoiceMixin)
    runner._voice_mode = {}
    runner._delivery_adapter_for = lambda source: owner
    runner._adapter_profile_for_source = lambda source: "fixture-profile"
    runner._voice_key_for_source = lambda source: "fixture-lane"
    runner._apply_voice_mode = lambda *args: None
    runner._handle_voice_channel_input = callback = AsyncMock()
    source = SessionSource(
        platform=Platform.DISCORD, chat_id="22", user_id="7", profile="fixture-profile",
    )
    event = SimpleNamespace(source=source, raw_message=SimpleNamespace(guild_id=1))
    await runner._handle_voice_channel_join(event)
    ingress = owner._voice_live_ingress[1]
    receiver = owner._voice_receivers[1]
    try:
        receiver.map_ssrc(17, 7)
        pcm = struct.pack("<hh", 600, -300) * 960
        receiver._accept_pcm(17, pcm, 1, 0)
        ingress._drain()
        assert await asyncio.to_thread(inner.fed.wait, 2), "capture was discarded before native STT"
        assert callback.await_count == 0
        for sequence in range(2, 31):
            receiver._accept_pcm(17, pcm, sequence, (sequence - 1) * 960)
        receiver._last_packet_time[17] = time.monotonic() - 2 * receiver.SILENCE_THRESHOLD
        completed = receiver.check_silence()
        await live.process_completed_inputs(owner, 1, receiver, completed, 22)
        assert inner.received == struct.pack("<h", 150) * (320 * 30)
        assert callback.await_count == 1
        assert callback.await_args.kwargs["transcript"] == "complete control tail"
        assert callback.await_args.kwargs["source_is_current"]() is True
        assert ingress.last_status["route"] == "native"
    finally:
        ingress.close()
        for task in owner._voice_listen_tasks.values():
            await task
