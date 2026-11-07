"""Authenticated voice-source discovery before and after Discord's voice handshake."""
from __future__ import annotations

from functools import lru_cache

_MAX_SOURCES = 1024


def _identifier(value, maximum):
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        if len(value) > len(str(maximum)) or not value.isascii() or not value.isdecimal():
            return None
        value = int(value)
    return value if isinstance(value, int) and 0 < value <= maximum else None


def voice_websocket_current(connection, websocket):
    current = getattr(connection, "ws", None)
    return not current or websocket is current


@lru_cache(maxsize=1)
def capture_voice_client_class():
    import discord

    class CaptureVoiceClient(discord.VoiceClient):
        async def connect(self, *args, **kwargs):
            if not hasattr(self, "_voice_capture_sources"):
                self._voice_capture_sources = {}
                connection = self._connection
                original = connection.hook

                async def capture_hook(websocket, message):
                    if not voice_websocket_current(connection, websocket):
                        return
                    if isinstance(message, dict):
                        data = message.get("d")
                        data = data if isinstance(data, dict) else {}
                        sources = self._voice_capture_sources
                        if message.get("op") == 2:
                            sources.clear()
                        elif message.get("op") == 5:
                            ssrc = _identifier(data.get("ssrc"), 0xFFFFFFFF)
                            user = _identifier(data.get("user_id"), 0xFFFFFFFFFFFFFFFF)
                            if ssrc is not None and user is not None:
                                if ssrc in sources or len(sources) < _MAX_SOURCES:
                                    sources[ssrc] = user
                        elif message.get("op") == 13:
                            user = _identifier(data.get("user_id"), 0xFFFFFFFFFFFFFFFF)
                            for ssrc, owner in list(sources.items()):
                                if owner == user:
                                    sources.pop(ssrc, None)
                    if original:
                        await original(websocket, message)

                connection.hook = capture_hook
            await super().connect(*args, **kwargs)

    return CaptureVoiceClient



def capture_packet_claim(receiver, ssrc):
    """Snapshot identity and crypto together before decrypt/decode can yield."""
    with receiver._lock:
        connection = getattr(receiver._vc, "_connection", None)
        claim = (
            receiver._source_reset_generation,
            receiver._source_epochs.get(ssrc, 0),
            receiver._trusted_ssrc_users.get(ssrc, 0),
            connection, getattr(connection, "ws", None),
        )
        return claim, receiver._creds, receiver._ssrc_to_user.get(ssrc, 0)


def packet_claim_current(receiver, ssrc, claim):
    """Caller holds the receiver lock."""
    connection = getattr(receiver._vc, "_connection", None)
    return (
        receiver._source_reset_generation == claim[0]
        and receiver._source_epochs.get(ssrc, 0) == claim[1]
        and receiver._trusted_ssrc_users.get(ssrc, 0) == claim[2]
        and connection is claim[3]
        and getattr(connection, "ws", None) is claim[4]
    )


def _retire_locked(receiver, affected):
    retired = [(key, receiver._source_epochs[key]) for key in affected]
    for key in affected:
        receiver._source_epochs[key] += 1
        receiver._buffers.pop(key, None)
        receiver._last_packet_time.pop(key, None)
        receiver._decoders.pop(key, None)
        receiver._ssrc_to_user.pop(key, None)
        receiver._trusted_ssrc_users.pop(key, None)
    receiver._completed_source_ids = ()
    return retired


def revoke_receiver_user(receiver, user_id):
    user = _identifier(user_id, 0xFFFFFFFFFFFFFFFF)
    if user is not None:
        reset_receiver_sources(receiver, user_id=user)

def reset_receiver_sources(receiver, *, user_id=None, ssrc=None):
    """Revoke trust and buffered generations before notifying the native owner."""
    with receiver._lock:
        if user_id is None and ssrc is None:
            receiver._source_reset_generation += 1
        known = (set(receiver._ssrc_to_user) | set(receiver._trusted_ssrc_users)
                 | set(receiver._buffers) | set(receiver._decoders))
        affected = [
            key for key in known
            if (ssrc is None or key == ssrc)
            and (user_id is None or receiver._ssrc_to_user.get(key) == user_id
                 or receiver._trusted_ssrc_users.get(key) == user_id)
        ]
        retired = _retire_locked(receiver, affected)
    if user_id is None and ssrc is None:
        callback = receiver._on_source_reset
        if callback:
            callback()
    elif receiver._on_source_revoke:
        for source in retired:
            receiver._on_source_revoke(source)


def map_receiver_source(receiver, ssrc, user_id):
    ssrc = _identifier(ssrc, 0xFFFFFFFF)
    user_id = _identifier(user_id, 0xFFFFFFFFFFFFFFFF)
    if ssrc is None or user_id is None:
        return False
    retired = []
    with receiver._lock:
        previous = receiver._trusted_ssrc_users.get(ssrc)
        decoder_only = ssrc in receiver._decoders and not receiver._buffers.get(ssrc)
        if previous != user_id and (previous is not None or decoder_only):
            retired = _retire_locked(receiver, [ssrc])
        if ssrc not in receiver._trusted_ssrc_users and len(receiver._trusted_ssrc_users) >= _MAX_SOURCES:
            return False
        receiver._ssrc_to_user[ssrc] = user_id
        receiver._trusted_ssrc_users[ssrc] = user_id
    if receiver._on_source_revoke:
        for source in retired:
            receiver._on_source_revoke(source)
    return True


def receive_decoded_packet(receiver, ssrc, payload, sequence, timestamp, claim, decoder_factory):
    """A retired decode cannot publish PCM or remove its successor's decoder."""
    import logging

    decoder = None
    try:
        with receiver._lock:
            if not packet_claim_current(receiver, ssrc, claim):
                return
            decoder = receiver._decoders.get(ssrc)
            if decoder is None:
                decoder = decoder_factory()
                receiver._decoders[ssrc] = decoder
        pcm = decoder.decode(payload)
        receiver._decode_ok += 1
        receiver._accept_pcm(ssrc, pcm, sequence, timestamp, _packet_claim=claim)
    except Exception as error:  # health: allow BLE001 -- Discord packet decoder boundary
        with receiver._lock:
            if decoder is not None and receiver._decoders.get(ssrc) is decoder:
                receiver._decoders.pop(ssrc, None)
        logging.getLogger(__name__).debug(
            "Opus decode error for SSRC %s; reset decoder: %s", ssrc, error,
        )
