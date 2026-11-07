"""Voice authentication is collected during the handshake and fenced by the live websocket."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import discord
import pytest

from plugins.platforms.discord.voice_capture_transport import capture_voice_client_class


@pytest.mark.asyncio
async def test_handshake_source_collection_and_original_hook_run_before_connect_returns(monkeypatch):
    client = object.__new__(capture_voice_client_class())
    websocket = object()
    original = AsyncMock()
    client._connection = connection = SimpleNamespace(hook=original, ws=websocket)
    calls = []

    async def connect(instance, *args, **kwargs):
        calls.append((args, kwargs))
        await connection.hook(websocket, {"op": 5, "d": {"ssrc": 17, "user_id": "7"}})
        assert instance._voice_capture_sources == {17: 7}

    monkeypatch.setattr(discord.VoiceClient, "connect", connect)
    await client.connect(timeout=30, reconnect=True, self_deaf=False, self_mute=False)
    installed = connection.hook
    await client.connect(timeout=40, reconnect=False)
    assert connection.hook is installed
    assert calls == [
        ((), {"timeout": 30, "reconnect": True, "self_deaf": False, "self_mute": False}),
        ((), {"timeout": 40, "reconnect": False}),
    ]
    assert original.await_count == 2


@pytest.mark.asyncio
async def test_stale_socket_malformed_mappings_and_reset_do_not_leak_trust(monkeypatch):
    client = object.__new__(capture_voice_client_class())
    websocket = object()
    original = AsyncMock()
    client._connection = connection = SimpleNamespace(hook=original, ws=websocket)

    async def connect(instance, **kwargs):
        pass

    monkeypatch.setattr(discord.VoiceClient, "connect", connect)
    await client.connect(timeout=30, reconnect=True)
    await connection.hook(object(), {"op": 5, "d": {"ssrc": 17, "user_id": "7"}})
    assert client._voice_capture_sources == {}
    assert original.await_count == 0
    for data in (None, [], {"ssrc": "bad", "user_id": "7"},
                 {"ssrc": 17, "user_id": True}, {"ssrc": -1, "user_id": "7"},
                 {"ssrc": "9" * 5000, "user_id": "7"}):
        await connection.hook(websocket, {"op": 5, "d": data})
    assert client._voice_capture_sources == {}
    assert original.await_count == 6
    await connection.hook(websocket, {"op": 5, "d": {"ssrc": 17, "user_id": "7"}})
    await connection.hook(websocket, {"op": 5, "d": {"ssrc": 18, "user_id": "8"}})
    await connection.hook(websocket, {"op": 13, "d": {"user_id": "7"}})
    assert client._voice_capture_sources == {18: 8}
    await connection.hook(websocket, {"op": 2, "d": {}})
    assert client._voice_capture_sources == {}
