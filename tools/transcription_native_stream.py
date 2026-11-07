"""Selected native STT sessions for live source consumers; never switch providers."""

from __future__ import annotations

import inspect
from typing import Any, Callable, Optional

from agent.transcription_provider import ContinuousTranscriptionStreamSession, TranscriptionStreamSession


def open_native_stream_session(
    *, stt_config: Optional[dict[str, Any]] = None, continuous: bool = False,
    on_event: Optional[Callable[[dict[str, Any]], None]] = None,
) -> Optional[TranscriptionStreamSession | ContinuousTranscriptionStreamSession]:
    """Open the selected native wire in the caller's owning profile scope.

    None explicitly declines this path. File/built-in fallback belongs to the caller,
    which must retain its addressed source and select one completion owner.
    """
    from agent.transcription_registry import get_provider
    from tools import transcription_tools as tt
    from tools.transcription_streaming import streaming_enabled

    cfg = tt._load_stt_config() if stt_config is None else stt_config
    if not streaming_enabled(cfg):
        return None
    name = tt._get_provider(cfg)
    if name in {"none", "local", "local_command", "auto"}:
        return None
    entry = cfg.get("providers", {}).get(name, {})
    if isinstance(entry, dict) and entry.get("type") == "command":
        return None
    provider = get_provider(name)
    capability = "continuous_streaming_capable" if continuous else "streaming_capable"
    if provider is None or not getattr(provider, capability, False) or not provider.is_available():
        return None
    section = cfg.get(name, {})
    section = section if isinstance(section, dict) else {}
    language = next((v.strip() for v in (section.get("language"), cfg.get("language"))
                     if isinstance(v, str) and v.strip()), None)
    options = {"language": language, "prompt": str(cfg.get("prompt") or "").strip() or None}
    method = provider.open_continuous_stream_session if continuous else provider.open_stream_session
    if continuous and on_event is not None:
        parameters = inspect.signature(method).parameters
        if "on_event" in parameters or any(p.kind is inspect.Parameter.VAR_KEYWORD
                                           for p in parameters.values()):
            options["on_event"] = on_event
    return method(**options)
