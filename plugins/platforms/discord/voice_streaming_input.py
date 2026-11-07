"""Bounded, profile-owned native transcription of decoded Discord source packets."""

from __future__ import annotations

import asyncio
import contextvars
import logging
import queue
import struct
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass

from agent.memory_provider import spawn_context_thread
from tools.transcription_native_stream import open_native_stream_session

logger = logging.getLogger(__name__)
_END = object()
_MAX_FRAME_BYTES = 23040  # Maximum 120 ms of 48 kHz stereo s16le.
_QUEUE_FRAMES = 128
_MAX_SOURCES = 8


@dataclass(frozen=True)
class PCMFrame:
    source_id: tuple[int, int]
    user_id: int
    pcm: bytes
    sequence: int
    timestamp: int
    arrived_at: float


class _LiveSource:
    def __init__(self, first: PCMFrame, binding: tuple):
        self.binding = binding
        self.user_id = first.user_id
        self._queue = queue.Queue(maxsize=_QUEUE_FRAMES)
        self._cancelled = threading.Event()
        self._done = threading.Event()
        self._inner = None
        self._result = None
        self._error = ""
        self._previous = None
        self._delivery_start = None
        self._carry = b""
        self._partial = ""
        self.stats = {
            "source_id": uuid.uuid4().hex, "ssrc": first.source_id[0],
            "source_generation": first.source_id[1], "frames": 0, "input_bytes": 0,
            "partial_events": 0, "gap": False, "capture_gap": False, "first_sequence": first.sequence,
            "first_timestamp": first.timestamp, "first_partial_monotonic": None,
            "last_audio_monotonic": None, "capture_end_monotonic": None,
            "finalize_started_monotonic": None,
        }
        self._thread = spawn_context_thread(self._run, name="discord-native-stt")
        self._thread.start()

    def offer(self, frame: PCMFrame) -> None:
        if self._cancelled.is_set() or self._done.is_set():
            return
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            self.cancel("native source queue overflow")

    def cancel(self, reason: str = "cancelled") -> None:
        self._error = self._error or reason
        self._cancelled.set()
        inner = self._inner
        if inner is not None and callable(cancel := getattr(inner, "cancel", None)):
            try:
                cancel()
            except Exception:  # health: allow BLE001 -- optional plugin cleanup boundary
                logger.warning("Native voice cleanup failed", exc_info=True)

    def _convert(self, pcm: bytes, *, final: bool = False) -> bytes:
        data = self._carry + pcm
        usable = len(data) - len(data) % 12
        out = bytearray()
        for samples in struct.iter_unpack("<hhhhhh", data[:usable]):
            out.extend(struct.pack("<h", sum(samples) // 6))
        self._carry = data[usable:]
        if final and self._carry:
            samples = struct.unpack("<" + "h" * (len(self._carry) // 2), self._carry)
            out.extend(struct.pack("<h", sum(samples) // len(samples)))
            self._carry = b""
        return bytes(out)

    def _feed(self, frame: PCMFrame) -> None:
        if not frame.pcm or len(frame.pcm) % 4 or len(frame.pcm) > _MAX_FRAME_BYTES:
            self.stats["gap"] = self.stats["capture_gap"] = True
            self.cancel("invalid source PCM frame")
            return
        if self._previous is not None:
            previous = self._previous
            sequence_delta = (frame.sequence - previous.sequence) & 0xFFFF
            timestamp_delta = (frame.timestamp - previous.timestamp) & 0xFFFFFFFF
            if sequence_delta != 1 or timestamp_delta != len(previous.pcm) // 4:
                self.stats["gap"] = self.stats["capture_gap"] = True
                self.cancel("source sequence or timestamp gap")
                return
        self._previous = frame
        self.stats["frames"] += 1
        self.stats["input_bytes"] += len(frame.pcm)
        self.stats["last_audio_monotonic"] = frame.arrived_at
        self.stats["last_sequence"] = frame.sequence
        self.stats["last_timestamp"] = frame.timestamp
        converted = self._convert(frame.pcm)
        if converted:
            if self._delivery_start is None:
                self._delivery_start = time.monotonic()
            offset = (self.stats["input_bytes"] - len(frame.pcm)) / 192000
            delay = max(0.0, self._delivery_start + offset - time.monotonic())
            if not self._cancelled.wait(delay):
                self._inner.push_audio(converted)

    def _poll_partial(self) -> None:
        text = str(self._inner.partial_transcript() or "").strip()
        if text and text != self._partial and not self._cancelled.is_set():
            self._partial = text
            self.stats["partial_events"] += 1
            if self.stats["first_partial_monotonic"] is None:
                self.stats["first_partial_monotonic"] = time.monotonic()
            logger.info("Native voice preview source=%s frames=%d characters=%d",
                        self.stats["source_id"], self.stats["frames"], len(text))

    def _run(self) -> None:
        try:
            self._inner = open_native_stream_session()
            if self._inner is None:
                return
            while not self._cancelled.is_set():
                try:
                    frame = self._queue.get(timeout=0.1)
                except queue.Empty:
                    self._poll_partial()
                    continue
                if frame is _END:
                    tail = self._convert(b"", final=True)
                    if tail:
                        self._inner.push_audio(tail)
                    self.stats["finalize_started_monotonic"] = time.monotonic()
                    self._result = dict(self._inner.finalize())
                    break
                self._feed(frame)
                if not self._cancelled.is_set():
                    self._poll_partial()
        except Exception as exc:  # health: allow BLE001 -- native provider worker boundary
            self.cancel(f"native source failed: {type(exc).__name__}")
        finally:
            if self._cancelled.is_set():
                self.cancel()
            self._done.set()

    def finish(self, expected_bytes: int, timeout: float = 30.0):
        self.stats["capture_end_monotonic"] = time.monotonic()
        if not self._done.is_set():
            try:
                self._queue.put_nowait(_END)
            except queue.Full:
                self.cancel("native source queue overflow at EOF")
            if not self._done.wait(timeout):
                self.cancel("native source finalization timed out")
        if self._done.is_set():
            self._thread.join(timeout=1)
        if self._inner is None and not self._error:
            return None
        if self.stats["input_bytes"] != expected_bytes:
            self.stats["gap"] = True
            self._error = self._error or "native source coverage differs from retained capture"
        if self._error or self._cancelled.is_set():
            result = {"success": False, "transcript": "", "error": self._error or "cancelled"}
        else:
            result = dict(self._result or {"success": False, "transcript": "",
                                           "error": "native source did not complete"})
        result["stream_source"] = dict(self.stats)
        result["worker_pending"] = not self._done.is_set()
        return result


class _Capture:
    """Capture continuity survives provider failure and delivery-queue saturation."""

    def __init__(self, frame: PCMFrame, binding: tuple, generation: int, context):
        self.binding, self.generation, self.context = binding, generation, context
        self.previous = None
        self.gap = False
        self.frames = self.input_bytes = 0
        self.first_sequence, self.first_timestamp = frame.sequence, frame.timestamp
        self.last_sequence, self.last_timestamp = frame.sequence, frame.timestamp

    def observe(self, frame: PCMFrame) -> None:
        if self.previous is not None:
            sequence, timestamp, samples = self.previous
            if ((frame.sequence - sequence) & 0xFFFF != 1
                    or (frame.timestamp - timestamp) & 0xFFFFFFFF != samples):
                self.gap = True
        if not frame.pcm or len(frame.pcm) % 4 or len(frame.pcm) > _MAX_FRAME_BYTES:
            self.gap = True
        self.previous = (frame.sequence, frame.timestamp, len(frame.pcm) // 4)
        self.frames += 1
        self.input_bytes += len(frame.pcm)
        self.last_sequence, self.last_timestamp = frame.sequence, frame.timestamp

    def snapshot(self) -> dict:
        return {"captured_frames": self.frames, "captured_bytes": self.input_bytes,
                "capture_gap": self.gap, "first_sequence": self.first_sequence,
                "first_timestamp": self.first_timestamp, "last_sequence": self.last_sequence,
                "last_timestamp": self.last_timestamp}


class VoiceStreamIngress:
    """One connection's capture and finalization ownership; partials never dispatch turns."""

    def __init__(self, adapter, guild_id: int):
        self._adapter, self._guild_id = adapter, guild_id
        self._loop = asyncio.get_running_loop()
        self._context = contextvars.copy_context()
        self._binding = self._current_binding()
        self._generation = 0
        self._not_before = 0.0
        self._sources, self._finalizing, self._captures = {}, {}, {}
        self._retired = deque(maxlen=256)
        self._queue = queue.Queue(maxsize=_QUEUE_FRAMES)
        self._lock = threading.Lock()
        self._scheduled = self._closed = self._overflow = self._capture_overflow = False
        self.last_status = {"route": "waiting"}

    def _current_binding(self) -> tuple:
        adapter = self._adapter
        source = getattr(adapter, "_voice_sources", {}).get(self._guild_id) or {}
        return (getattr(adapter, "_owner_profile", None),
                getattr(adapter, "_voice_text_channels", {}).get(self._guild_id),
                tuple((key, source.get(key)) for key in ("profile", "chat_id", "thread_id", "session_id")))

    def rebind(self) -> None:
        current = self._current_binding()
        if current != self._binding:
            self.invalidate()
            self._binding = current
            self._context = contextvars.copy_context()

    def _schedule(self) -> None:
        with self._lock:
            if self._scheduled or self._closed:
                return
            self._scheduled = True
        self._loop.call_soon_threadsafe(self._drain, context=self._context.copy())

    def offer(self, frame: PCMFrame) -> None:
        """Capture-side continuity runs before any bounded delivery queue or worker."""
        with self._lock:
            if self._closed:
                return
            if frame.source_id in self._retired or frame.arrived_at < self._not_before:
                if frame.source_id not in self._retired:
                    self._retired.append(frame.source_id)
                return
            capture = self._captures.get(frame.source_id)
            if capture is None:
                if len(self._captures) >= 256:
                    self._capture_overflow = True
                    return
                capture = _Capture(frame, self._binding, self._generation, self._context.copy())
                self._captures[frame.source_id] = capture
            capture.observe(frame)
            if not frame.user_id:
                return
            try:
                self._queue.put_nowait((self._generation, frame))
            except queue.Full:
                self._overflow = True
        self._schedule()

    def _authorized(self, user_id: int) -> bool:
        client = getattr(self._adapter, "_client", None)
        guild = client.get_guild(self._guild_id) if client is not None else None
        return self._context.copy().run(
            self._adapter._is_allowed_user, str(user_id), guild=guild, is_dm=False) is True

    def _start_or_feed(self, frame: PCMFrame) -> None:
        if frame.source_id in self._retired:
            return
        if not self._authorized(frame.user_id) or self._current_binding() != self._binding:
            self.discard_source(frame.source_id)
            return
        if self._overflow:
            for source in self._sources.values():
                source.cancel("Discord ingress queue overflow")
            self.last_status = {"route": "fallback_required", "reason": "Discord ingress queue overflow"}
            return
        source = self._sources.get(frame.source_id)
        if source is None:
            if (len(self._sources) + len(self._finalizing) >= _MAX_SOURCES
                    or self._current_binding()[1] is None):
                self.last_status = {"route": "declined", "reason": "source capacity or binding unavailable"}
                return
            source = self._context.copy().run(_LiveSource, frame, (self._binding, self._generation))
            self._sources[frame.source_id] = source
        if source.user_id != frame.user_id:
            self.discard_source(frame.source_id)
            return
        source.offer(frame)

    def _drain(self) -> None:
        with self._lock:
            self._scheduled = False
        if self._closed:
            return
        for _ in range(_QUEUE_FRAMES):
            try:
                generation, frame = self._queue.get_nowait()
            except queue.Empty:
                break
            if generation != self._generation:
                self.discard_source(frame.source_id)
            else:
                self._start_or_feed(frame)
        if not self._queue.empty():
            self._schedule()

    def source_claim(self, source_id: tuple):
        with self._lock:
            return self._captures.get(source_id)

    def claim_is_current(self, claim) -> bool:
        return (not self._closed and claim is not None and claim.binding == self._current_binding()
                and claim.generation == self._generation)

    def _incomplete_capture(self, captured: dict):
        self.last_status = {"route": "incomplete", "source": captured}
        return {"success": False, "transcript": "", "error": "source capture is incomplete",
                "stream_source": {**captured, "capture_gap": True, "gap": True}}

    async def finish_source(self, source_id: tuple, user_id: int, pcm: bytes, captured_for: int | None):
        self._drain()
        with self._lock:
            if source_id in self._retired:
                return {"success": False, "transcript": "", "stale": True}
            capture = self._captures.pop(source_id, None)
            self._retired.append(source_id)
        source = self._sources.pop(source_id, None)
        captured = capture.snapshot() if capture is not None else {}
        if self._capture_overflow or captured.get("capture_gap"):
            if source is not None:
                source.cancel("source capture is incomplete")
            return self._incomplete_capture(captured)
        if source is None:
            self.last_status = {"route": "declined", "source": captured}
            return None
        if (not self.claim_is_current(capture) or source.user_id != user_id
                or captured_for != self._current_binding()[1] or not self._authorized(user_id)):
            source.cancel("stale source binding")
            return {"success": False, "transcript": "", "stale": True}
        self._finalizing[source_id] = source
        try:
            result = await asyncio.to_thread(source.finish, len(pcm))
        except asyncio.CancelledError:
            source.cancel("source finalization waiter cancelled")
            raise
        finally:
            self._finalizing.pop(source_id, None)
        if not self.claim_is_current(capture) or not self._authorized(user_id):
            return {"success": False, "transcript": "", "stale": True}
        if result is not None:
            result["stream_source"].update(captured)
        route = "declined" if result is None else ("native" if result.get("success") else "fallback_required")
        self.last_status = {"route": route, "source": {**source.stats, **captured}}
        logger.info("Native voice source completion: %s", self.last_status)
        return result

    def discard_source(self, source_id: tuple) -> None:
        sources = [self._sources.pop(source_id, None), self._finalizing.pop(source_id, None)]
        with self._lock:
            self._captures.pop(source_id, None)
            if source_id not in self._retired:
                self._retired.append(source_id)
        for source in sources:
            if source is not None:
                source.cancel("source authorization or ownership changed")

    def invalidate(self) -> None:
        sources = list(self._sources.values()) + list(self._finalizing.values())
        self._sources.clear()
        self._finalizing.clear()
        with self._lock:
            self._generation += 1
            self._not_before = time.monotonic()
            self._retired.extend(source_id for source_id in self._captures if source_id not in self._retired)
            self._captures.clear()
            while not self._queue.empty():
                self._queue.get_nowait()
        for source in sources:
            source.cancel("source binding invalidated")

    def close(self) -> None:
        self._closed = True
        self.invalidate()


async def _deliver_transcript(adapter, guild_id: int, user_id: int, result: dict, captured_for,
                              source_guard=None):
    import inspect
    from tools.voice_mode_transcript import is_whisper_hallucination

    transcript = str(result.get("transcript") or "").strip()
    if not transcript or is_whisper_hallucination(transcript):
        return
    if (getattr(adapter, "_voice_text_channels", {}).get(guild_id) != captured_for
            or source_guard is not None and not source_guard()):
        return
    callback = adapter._voice_input_callback
    if callback:
        options = {"guild_id": guild_id, "user_id": user_id, "transcript": transcript}
        if source_guard is not None:
            try:
                parameters = inspect.signature(callback).parameters
            except (TypeError, ValueError):
                parameters = {}
            if "source_is_current" in parameters or any(p.kind is inspect.Parameter.VAR_KEYWORD
                                                        for p in parameters.values()):
                options["source_is_current"] = source_guard
        await callback(**options)


async def process_voice_input(adapter, guild_id: int, user_id: int, pcm_data: bytes,
                              captured_for, *, stream_result=None, source_guard=None):
    """Choose native completion or retained addressed-file recovery, exactly once."""
    import os
    import tempfile
    from plugins.platforms.discord.adapter import VoiceReceiver

    if source_guard is not None and not source_guard():
        return
    if stream_result is not None:
        if stream_result.get("stale"):
            return
        if stream_result.get("success"):
            await _deliver_transcript(adapter, guild_id, user_id, stream_result, captured_for, source_guard)
            return
        if stream_result.get("stream_source", {}).get("capture_gap"):
            logger.warning("Voice capture incomplete: retained audio cannot repair its source gap")
            return
        logger.info("Native source incomplete; using its retained addressed file")
    tmp_f = tempfile.NamedTemporaryFile(suffix=".wav", prefix="vc_listen_", delete=False)
    wav_path = tmp_f.name
    tmp_f.close()
    try:
        await asyncio.to_thread(VoiceReceiver.pcm_to_wav, pcm_data, wav_path)
        if source_guard is not None and not source_guard():
            return
        from tools.transcription_tools import transcribe_audio
        result = await asyncio.to_thread(transcribe_audio, wav_path)
        if result.get("success"):
            await _deliver_transcript(adapter, guild_id, user_id, result, captured_for, source_guard)
    except Exception as exc:  # health: allow BLE001 -- voice input pipeline boundary
        stderr = getattr(exc, "stderr", None)
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", "replace")
        detail = f"{exc} (ffmpeg: {stderr.strip()})" if stderr else str(exc)
        logger.warning("Voice input processing failed: %s", detail, exc_info=True)
    finally:
        try:
            os.unlink(wav_path)
        except FileNotFoundError:
            pass


async def process_completed_inputs(adapter, guild_id: int, receiver, completed, captured_for):
    source_ids = getattr(receiver, "_completed_source_ids", ())
    source_ids = source_ids if isinstance(source_ids, tuple) and len(source_ids) == len(completed) else ()
    client = getattr(adapter, "_client", None)
    guild = client.get_guild(guild_id) if client is not None else None
    ingress = getattr(adapter, "_voice_live_ingress", {}).get(guild_id)
    discarded = getattr(receiver, "_discarded_source_ids", ())
    if ingress is not None and isinstance(discarded, tuple):
        for source_id in discarded:
            ingress.discard_source(source_id)
    for index, (user_id, pcm_data) in enumerate(completed):
        if not adapter._is_allowed_user(str(user_id), guild=guild, is_dm=False):
            if ingress is not None and source_ids:
                ingress.discard_source(source_ids[index])
            continue
        adapter._reset_voice_timeout(guild_id)
        claim = ingress.source_claim(source_ids[index]) if ingress is not None and source_ids else None
        stream_result = None
        if ingress is not None and source_ids:
            stream_result = await ingress.finish_source(source_ids[index], user_id, pcm_data, captured_for)
        if claim is not None:
            guard = lambda owner=ingress, captured=claim: owner.claim_is_current(captured)
            operation = adapter._process_voice_input(
                guild_id, user_id, pcm_data, captured_for, stream_result=stream_result, source_guard=guard)
            await asyncio.create_task(operation, context=claim.context.copy())
        elif stream_result is None:
            await adapter._process_voice_input(guild_id, user_id, pcm_data, captured_for)
        else:
            await adapter._process_voice_input(guild_id, user_id, pcm_data, captured_for,
                                               stream_result=stream_result)


async def voice_listen_loop(adapter, guild_id: int):
    receiver = adapter._voice_receivers.get(guild_id)
    if not receiver:
        return
    last_keepalive = time.monotonic()
    try:
        while receiver._running:
            await asyncio.sleep(0.2)
            now = time.monotonic()
            if now - last_keepalive >= adapter._KEEPALIVE_INTERVAL:
                last_keepalive = now
                try:
                    vc = adapter._voice_clients.get(guild_id)
                    if vc and vc.is_connected():
                        vc._connection.send_packet(bytes((0xF8, 0xFF, 0xFE)))
                except Exception:  # health: allow BLE001 -- transport keepalive is best effort
                    logger.debug("Voice keepalive failed", exc_info=True)
            completed = receiver.check_silence()
            captured_for = adapter._voice_text_channels.get(guild_id)
            await process_completed_inputs(adapter, guild_id, receiver, completed, captured_for)
    except asyncio.CancelledError:
        pass
    except Exception as exc:  # health: allow BLE001 -- adapter listener task boundary
        logger.error("Voice listen loop error: %s", exc, exc_info=True)


async def leave_voice_channel(adapter, guild_id: int):
    async with adapter._voice_locks.setdefault(guild_id, asyncio.Lock()):
        receiver = adapter._voice_receivers.pop(guild_id, None)
        pending_inputs = []
        if receiver:
            receiver._running = False
            pending_inputs = receiver.flush_pending()
            receiver.stop()
        listen_task = adapter._voice_listen_tasks.pop(guild_id, None)
        if listen_task:
            listen_task.cancel()
        captured_for = adapter._voice_text_channels.get(guild_id)
        await process_completed_inputs(adapter, guild_id, receiver, pending_inputs, captured_for)
        ingress = getattr(adapter, "_voice_live_ingress", {}).pop(guild_id, None)
        if ingress is not None:
            ingress.close()
        if getattr(adapter, "_voice_mixers", None) is not None:
            adapter._voice_mixers.pop(guild_id, None)
        vc = adapter._voice_clients.pop(guild_id, None)
        if vc and vc.is_connected():
            try:
                if vc.is_playing():
                    vc.stop()
            except Exception:  # health: allow BLE001 -- preserve transport disconnect after stop failure
                logger.debug("Voice playback stop failed", exc_info=True)
            await vc.disconnect()
        task = adapter._voice_timeout_tasks.pop(guild_id, None)
        if task:
            task.cancel()
        adapter._voice_text_channels.pop(guild_id, None)
        adapter._voice_sources.pop(guild_id, None)
