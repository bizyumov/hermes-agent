"""Optional native continuous SPI and selected-provider resolution for yandex-speech #43."""

from agent import transcription_registry
from agent.transcription_provider import TranscriptionProvider


class FileProvider(TranscriptionProvider):
    name = "continuous-contract-fixture"

    def is_available(self):
        return True

    def transcribe(self, file_path, **extra):
        return {"success": True, "transcript": "file", "provider": self.name}


def test_existing_file_provider_declines_continuous_without_new_abstract_requirements():
    import pytest

    provider = FileProvider()
    assert provider.continuous_streaming_capable is False
    with pytest.raises(NotImplementedError):
        provider.open_continuous_stream_session()


def test_selected_native_short_stream_receives_profile_language_and_prompt(monkeypatch):
    from tools.transcription_native_stream import open_native_stream_session

    inner = object()
    calls = []

    class ShortProvider(FileProvider):
        streaming_capable = True

        def open_stream_session(self, *, language=None, prompt=None):
            calls.append((language, prompt))
            return inner

    provider = ShortProvider()
    monkeypatch.setattr(transcription_registry, "get_provider", lambda name: provider if name == provider.name else None)
    result = open_native_stream_session(stt_config={
        "streaming": True, "provider": provider.name, "language": "ru-RU", "prompt": "vocabulary",
    })
    assert result is inner
    assert calls == [("ru-RU", "vocabulary")]


def test_continuous_callback_is_optional_for_a_narrow_provider_signature(monkeypatch):
    from tools.transcription_native_stream import open_native_stream_session

    inner = object()

    class NarrowProvider(FileProvider):
        continuous_streaming_capable = True

        def open_continuous_stream_session(self, *, language=None, prompt=None):
            return inner

    monkeypatch.setattr(transcription_registry, "get_provider", lambda name: NarrowProvider())
    assert open_native_stream_session(stt_config={"streaming": True, "provider": FileProvider.name},
                                      continuous=True, on_event=lambda event: None) is inner


def test_continuous_callback_reaches_an_accepting_provider(monkeypatch):
    from tools.transcription_native_stream import open_native_stream_session

    events = []

    class EventProvider(FileProvider):
        continuous_streaming_capable = True

        def open_continuous_stream_session(self, *, language=None, prompt=None, on_event=None):
            on_event({"kind": "partial", "input_complete": False})
            return self

    provider = EventProvider()
    monkeypatch.setattr(transcription_registry, "get_provider", lambda name: provider)
    assert open_native_stream_session(stt_config={"streaming": True, "provider": provider.name},
                                      continuous=True, on_event=events.append) is provider
    assert events == [{"kind": "partial", "input_complete": False}]


def test_disabled_declined_missing_and_command_providers_never_select_another_wire(monkeypatch):
    from tools.transcription_native_stream import open_native_stream_session

    calls = []
    monkeypatch.setattr(transcription_registry, "get_provider", lambda name: calls.append(name) or FileProvider())
    for config in [
        {"streaming": False, "provider": FileProvider.name},
        {"streaming": True, "provider": "none"},
        {"streaming": True, "provider": FileProvider.name, "providers": {FileProvider.name: {"type": "command"}}},
    ]:
        assert open_native_stream_session(stt_config=config, continuous=True) is None
    assert calls == []
    assert open_native_stream_session(stt_config={"streaming": True, "provider": FileProvider.name},
                                      continuous=True) is None
    assert calls == [FileProvider.name]
