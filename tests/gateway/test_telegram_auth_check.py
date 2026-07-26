"""Tests for Telegram adapter early authorization check.

Verifies that unauthorized users are blocked before any text batching,
event building, or response generation occurs.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageType


def _make_adapter(allow_from=None, allowed_chats=None, group_allowed_chats=None, callback_auth=None, **extra_overrides):
    try:
        from plugins.platforms.telegram.adapter import TelegramAdapter
    except ModuleNotFoundError:  # PR branch before Telegram plugin extraction
        from gateway.platforms.telegram import TelegramAdapter

    extra = {}
    if allow_from is not None:
        extra["allow_from"] = allow_from
    if allowed_chats is not None:
        extra["allowed_chats"] = allowed_chats
    if group_allowed_chats is not None:
        extra["group_allowed_chats"] = group_allowed_chats
    extra.update(extra_overrides)

    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="fake-token", extra=extra)
    adapter._bot = SimpleNamespace(id=999, username="test_bot")
    adapter._message_handler = AsyncMock()
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._text_batch_delay_seconds = 0.01
    adapter._text_batch_split_delay_seconds = 0.01
    adapter._mention_patterns = adapter._compile_mention_patterns()
    adapter._forum_lock = asyncio.Lock()
    adapter._forum_command_registered = set()
    adapter._active_sessions = {}
    adapter._pending_messages = {}
    if callback_auth is not None:
        adapter._is_callback_user_authorized = callback_auth
    return adapter


def _make_message(text="hello", *, from_user_id=111, chat_id=-100, chat_type="group"):
    return SimpleNamespace(
        message_id=42,
        text=text,
        caption=None,
        entities=[],
        caption_entities=[],
        message_thread_id=None,
        is_topic_message=False,
        chat=SimpleNamespace(id=chat_id, type=chat_type, title="Test", is_forum=False),
        from_user=SimpleNamespace(id=from_user_id, full_name="Test User", first_name="Test"),
        reply_to_message=None,
        date=None,
        location=None,
        photo=None,
        video=None,
        audio=None,
        voice=None,
        document=None,
        sticker=None,
        media_group_id=None,
    )


@pytest.mark.asyncio
async def test_unauthorized_user_blocked_before_event_building():
    """Unauthorized user's message should be blocked before _build_message_event."""
    adapter = _make_adapter(group_allow_from=["222"])  # Only user 222 allowed in groups

    build_called = False
    original_build = adapter._build_message_event

    def track_build(*a, **kw):
        nonlocal build_called
        build_called = True
        return original_build(*a, **kw)

    adapter._build_message_event = track_build

    update = SimpleNamespace(
        update_id=1,
        message=_make_message(from_user_id=111, chat_type="group"),  # User 111 NOT in group_allow_from
        effective_message=None,
    )

    await adapter._handle_text_message(update, SimpleNamespace())

    assert build_called is False, "build_message_event should not be called for unauthorized user"


@pytest.mark.asyncio
async def test_command_from_unauthorized_user_blocked():
    """Commands from unauthorized users should be blocked."""
    adapter = _make_adapter(group_allow_from=["222"])
    adapter.handle_message = AsyncMock()

    update = SimpleNamespace(
        update_id=1,
        message=_make_message(text="/start", from_user_id=111, chat_type="group"),
        effective_message=None,
    )

    await adapter._handle_command(update, SimpleNamespace())

    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_location_from_unauthorized_user_blocked():
    """Location messages from unauthorized users should be blocked."""
    adapter = _make_adapter(group_allow_from=["222"])

    msg = _make_message(from_user_id=111, chat_type="group")
    msg.text = None
    msg.location = SimpleNamespace(latitude=53.3498, longitude=-6.2603)

    update = SimpleNamespace(
        update_id=1,
        message=msg,
        effective_message=None,
    )

    # Should not raise — just silently return
    await adapter._handle_location_message(update, SimpleNamespace())


def test_is_user_authorized_from_message_allow_from():
    """_is_user_authorized_from_message should respect adapter-level allow_from for DMs."""
    adapter = _make_adapter(allow_from=["111", "222"])

    msg = _make_message(from_user_id=111, chat_type="dm")
    assert adapter._is_user_authorized_from_message(msg) is True

    msg = _make_message(from_user_id=333, chat_type="dm")
    assert adapter._is_user_authorized_from_message(msg) is False


@pytest.mark.parametrize("forum", [False, True])
def test_global_allow_from_remains_a_grant_in_group_context(forum):
    """The platform-wide list is ORed with group-only sender grants."""
    adapter = _make_adapter(
        allow_from=["111"],
        group_allow_from=["222"],
    )
    msg = _make_message(from_user_id=111, chat_id=-100, chat_type="supergroup")
    if forum:
        msg.chat.is_forum = True
        msg.is_topic_message = True
        msg.message_thread_id = 7

    assert adapter._is_user_authorized_from_message(msg) is True


@pytest.mark.parametrize("forum", [False, True])
def test_group_allowed_chat_is_ored_with_group_sender_list(forum):
    """A listed chat authorizes every member even when a sender list exists."""
    adapter = _make_adapter(
        group_allow_from=["222"],
        group_allowed_chats=["-100"],
    )
    msg = _make_message(from_user_id=111, chat_id=-100, chat_type="supergroup")
    if forum:
        msg.chat.is_forum = True
        msg.is_topic_message = True
        msg.message_thread_id = 7

    assert adapter._is_user_authorized_from_message(msg) is True


def test_empty_group_config_lists_defer_to_injected_authority():
    """Empty YAML defaults must not become a sole-authority rejection."""
    adapter = _make_adapter(
        group_allow_from=[],
        group_allowed_chats=[],
        callback_auth=lambda uid, **_kw: uid == "111",
    )

    assert adapter._is_user_authorized_from_message(
        _make_message(from_user_id=111, chat_id=-100, chat_type="group")
    ) is True


def test_config_allowlists_authorize_the_documented_group_union():
    """Global users, group users, and allowed chats are independent grants."""
    adapter = _make_adapter(
        allow_from=["global-user"],
        group_allow_from=["group-user"],
        group_allowed_chats=["-100"],
    )

    assert adapter._is_user_authorized_from_message(
        _make_message(from_user_id="global-user", chat_id=-200, chat_type="group")
    ) is True
    assert adapter._is_user_authorized_from_message(
        _make_message(from_user_id="group-user", chat_id=-200, chat_type="group")
    ) is True
    assert adapter._is_user_authorized_from_message(
        _make_message(from_user_id="unlisted-user", chat_id=-100, chat_type="group")
    ) is True
    assert adapter._is_user_authorized_from_message(
        _make_message(from_user_id="unlisted-user", chat_id=-200, chat_type="group")
    ) is False
    assert adapter._is_user_authorized_from_message(
        _make_message(from_user_id="group-user", chat_id=123, chat_type="private")
    ) is False


def test_runner_config_authorization_matches_telegram_intake_union(monkeypatch):
    """YAML-config and environment allowlists produce the same intake result."""
    from gateway.run import GatewayRunner

    for key in (
        "TELEGRAM_ALLOWED_USERS",
        "TELEGRAM_GROUP_ALLOWED_USERS",
        "TELEGRAM_GROUP_ALLOWED_CHATS",
        "TELEGRAM_ALLOW_ALL_USERS",
        "GATEWAY_ALLOWED_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(key, raising=False)

    adapter = _make_adapter(
        allow_from=["global-user"],
        group_allow_from=["group-user"],
        group_allowed_chats=["-100"],
    )
    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False

    cases = (
        (_make_message(from_user_id="global-user", chat_id=-200, chat_type="group"), True),
        (_make_message(from_user_id="group-user", chat_id=-200, chat_type="group"), True),
        (_make_message(from_user_id="unlisted-user", chat_id=-100, chat_type="group"), True),
        (_make_message(from_user_id="unlisted-user", chat_id=-200, chat_type="group"), False),
        (_make_message(from_user_id="group-user", chat_id=123, chat_type="private"), False),
    )
    config_intake = []
    for message, expected in cases:
        source = adapter._source_from_message_for_auth(message)
        config_intake.append(adapter._is_user_authorized_from_message(message))
        assert runner._is_user_authorized(source) is expected

    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "global-user")
    monkeypatch.setenv("TELEGRAM_GROUP_ALLOWED_USERS", "group-user")
    monkeypatch.setenv("TELEGRAM_GROUP_ALLOWED_CHATS", "-100")
    env_adapter = _make_adapter()
    env_runner = object.__new__(GatewayRunner)
    env_runner.adapters = {Platform.TELEGRAM: env_adapter}
    env_runner.pairing_store = MagicMock()
    env_runner.pairing_store.is_approved.return_value = False
    env_adapter._message_handler = env_runner._is_user_authorized

    env_intake = [
        env_adapter._is_user_authorized_from_message(message)
        for message, _expected in cases
    ]
    assert config_intake == env_intake


@pytest.mark.parametrize(
    ("env_name", "env_value", "message"),
    (
        (
            "TELEGRAM_GROUP_ALLOWED_USERS",
            "group-user",
            _make_message(from_user_id="group-user", chat_id=-200, chat_type="group"),
        ),
        (
            "TELEGRAM_GROUP_ALLOWED_CHATS",
            "-100",
            _make_message(from_user_id="unlisted-user", chat_id=-100, chat_type="group"),
        ),
    ),
)
def test_mixed_yaml_and_environment_group_grants_are_unioned(
    monkeypatch,
    env_name,
    env_value,
    message,
):
    """A non-matching YAML global list must not hide an env group grant."""
    from gateway.run import GatewayRunner

    for key in (
        "TELEGRAM_ALLOWED_USERS",
        "TELEGRAM_GROUP_ALLOWED_USERS",
        "TELEGRAM_GROUP_ALLOWED_CHATS",
        "TELEGRAM_ALLOW_ALL_USERS",
        "GATEWAY_ALLOWED_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv(env_name, env_value)

    adapter = _make_adapter(
        allow_from=["global-user"],
        group_allow_from=[],
        group_allowed_chats=[],
    )
    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False
    adapter._message_handler = runner._is_user_authorized

    assert adapter._is_user_authorized_from_message(message) is True
    assert runner._is_user_authorized(adapter._source_from_message_for_auth(message)) is True


def test_scalar_config_allowlists_match_sequence_semantics():
    """Comma-separated YAML scalars use the same scoped union as sequences."""
    adapter = _make_adapter(
        allow_from="111, 222",
        group_allow_from="333, 444",
        group_allowed_chats="-100, -200",
    )

    assert adapter._is_user_authorized_from_message(
        _make_message(from_user_id=222, chat_id=-300, chat_type="group")
    ) is True
    assert adapter._is_user_authorized_from_message(
        _make_message(from_user_id=444, chat_id=-300, chat_type="group")
    ) is True
    assert adapter._is_user_authorized_from_message(
        _make_message(from_user_id=555, chat_id=-200, chat_type="group")
    ) is True


@pytest.mark.parametrize(
    ("extra", "message"),
    (
        (
            {"group_allow_from": ["*"]},
            _make_message(from_user_id=111, chat_id=-300, chat_type="group"),
        ),
        (
            {"group_allowed_chats": ["*"]},
            _make_message(from_user_id=111, chat_id=-300, chat_type="group"),
        ),
    ),
)
def test_group_scoped_wildcards_authorize(extra, message):
    adapter = _make_adapter(**extra)

    assert adapter._is_user_authorized_from_message(message) is True


def test_allowlist_dm_with_explicit_pair_behavior_reaches_gateway(monkeypatch):
    """Allowlist + unauthorized_dm_behavior:pair must not early-drop unknown DMs.

    Regression for the gap left by #40863: early intake rejection discarded
    unauthorized DMs before gateway pairing could run, even when the operator
    explicitly set telegram.unauthorized_dm_behavior: pair.
    """
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "111")

    class Runner:
        def _is_user_authorized(self, source):
            return source.user_id == "111"

        def _get_unauthorized_dm_behavior(self, platform, *, profile=None):
            assert platform == Platform.TELEGRAM
            return "pair"

        async def handle(self, event):
            return None

    runner = Runner()
    adapter = _make_adapter()
    adapter._message_handler = runner.handle
    msg = _make_message(from_user_id=999, chat_id=999, chat_type="private")

    assert adapter._is_user_authorized_from_message(msg) is True


def test_allowlist_dm_without_pair_behavior_still_early_rejects(monkeypatch):
    """Allowlist without pairing opt-in keeps the #9337/#40863 silent drop."""
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "111")

    class Runner:
        def _is_user_authorized(self, source):
            return source.user_id == "111"

        def _get_unauthorized_dm_behavior(self, platform, *, profile=None):
            return "ignore"

        async def handle(self, event):
            return None

    runner = Runner()
    adapter = _make_adapter()
    adapter._message_handler = runner.handle
    msg = _make_message(from_user_id=999, chat_id=999, chat_type="private")

    assert adapter._is_user_authorized_from_message(msg) is False


def test_allow_from_dm_with_pair_override_reaches_gateway():
    """Adapter allow_from + unauthorized_dm_behavior:pair still forwards DMs."""
    adapter = _make_adapter(
        allow_from=["111"],
        unauthorized_dm_behavior="pair",
    )
    msg = _make_message(from_user_id=999, chat_id=999, chat_type="dm")
    assert adapter._is_user_authorized_from_message(msg) is True


def test_allowlist_group_with_pair_behavior_still_early_rejects(monkeypatch):
    """Pairing is DM-only — unauthorized group senders stay blocked early."""
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "111")

    class Runner:
        def _is_user_authorized(self, source):
            return source.user_id == "111"

        def _get_unauthorized_dm_behavior(self, platform, *, profile=None):
            return "pair"

        async def handle(self, event):
            return None

    runner = Runner()
    adapter = _make_adapter(group_allow_from=["111"])
    adapter._message_handler = runner.handle
    msg = _make_message(from_user_id=999, chat_id=-100, chat_type="group")

    assert adapter._is_user_authorized_from_message(msg) is False


@pytest.mark.asyncio
async def test_unauthorized_dm_with_pair_behavior_builds_event(monkeypatch):
    """Unknown DM under pair behavior must reach event construction."""
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "111")

    class Runner:
        def _is_user_authorized(self, source):
            return source.user_id == "111"

        def _get_unauthorized_dm_behavior(self, platform, *, profile=None):
            return "pair"

        async def handle(self, event):
            return None

    runner = Runner()
    adapter = _make_adapter()
    adapter._message_handler = runner.handle
    build_called = False
    original_build = adapter._build_message_event

    def track_build(*a, **kw):
        nonlocal build_called
        build_called = True
        return original_build(*a, **kw)

    adapter._build_message_event = track_build
    adapter._enqueue_text_event = lambda event: None
    adapter._ensure_forum_commands = AsyncMock()
    adapter._cache_replied_media = AsyncMock()
    adapter._apply_telegram_group_observe_attribution = lambda event: event
    adapter._clean_bot_trigger_text = lambda text: text
    adapter._should_process_message = lambda *a, **kw: True

    update = SimpleNamespace(
        update_id=1,
        message=_make_message(from_user_id=999, chat_id=999, chat_type="private"),
        effective_message=None,
    )
    await adapter._handle_text_message(update, SimpleNamespace())
    assert build_called is True


def test_runner_auth_gets_group_user_allowlist_context(monkeypatch):
    """Group user allowlists need a group-shaped source, not a DM-shaped one."""
    monkeypatch.setenv("TELEGRAM_GROUP_ALLOWED_USERS", "111")
    seen_sources = []

    class Runner:
        def _is_user_authorized(self, source):
            seen_sources.append(source)
            return source.chat_type == "group" and source.chat_id == "-100" and source.user_id == "111"

        async def handle(self, event):
            return None

    runner = Runner()
    adapter = _make_adapter()
    adapter._message_handler = runner.handle
    msg = _make_message(from_user_id=111, chat_id=-100, chat_type="group")

    assert adapter._is_user_authorized_from_message(msg) is True
    assert seen_sources
    assert seen_sources[0].chat_type == "group"
    assert seen_sources[0].chat_id == "-100"


@pytest.mark.asyncio
async def test_unmentioned_group_location_from_removed_user_not_observed():
    """Removed users must not persist unmentioned group locations into observed context."""
    adapter = _make_adapter(
        group_allow_from=["222"],
        allowed_chats=["-100"],
        group_allowed_chats=["-200"],
        require_mention=True,
        observe_unmentioned_group_messages=True,
    )
    observed = []
    adapter._observe_unmentioned_group_message = lambda *args, **kwargs: observed.append((args, kwargs))

    msg = _make_message(text=None, from_user_id=111, chat_id=-100, chat_type="group")
    msg.location = SimpleNamespace(latitude=53.3498, longitude=-6.2603)
    update = SimpleNamespace(update_id=1, message=msg, effective_message=None)

    await adapter._handle_location_message(update, SimpleNamespace())

    assert observed == []


def test_group_allowlist_authorized_under_multiplex_closure_handler(monkeypatch):
    """group_allowed_chats must authorize a chat member under multiplex_profiles.

    Regression for #87132: with gateway.multiplex_profiles the primary message
    handler is a closure, so its ``__self__`` is absent and the early intake
    filter could not reach GatewayRunner._is_user_authorized — it fell back to
    env-only auth and default-denied every non-global sender in an explicitly
    allowlisted group. The platform-bound callback registered via
    set_authorization_check survives the closure wrapping and must be consulted.
    """
    monkeypatch.setenv("TELEGRAM_GROUP_ALLOWED_CHATS", "-100123")

    adapter = _make_adapter(group_allowed_chats=["-100123"])

    # Multiplex: the primary handler is a closure with no ``__self__`` runner.
    def closure_handler(event):
        return None

    adapter._message_handler = closure_handler
    assert getattr(closure_handler, "__self__", None) is None

    # The runner installs this callback at adapter registration; it routes
    # through the full auth chain (here: the chat allowlist) regardless of how
    # the message handler is wrapped.
    def auth_check(user_id, chat_type=None, chat_id=None):
        return str(chat_id) in {"-100123"}

    adapter.set_authorization_check(auth_check)

    # A sender absent from any user allowlist, posting in the allowlisted group,
    # is authorized via the chat allowlist.
    allowed = _make_message(from_user_id=555, chat_id=-100123, chat_type="group")
    assert adapter._is_user_authorized_from_message(allowed) is True

    # The same sender in a NON-allowlisted group is still rejected.
    denied = _make_message(from_user_id=555, chat_id=-100999, chat_type="group")
    assert adapter._is_user_authorized_from_message(denied) is False


def test_multiplex_closure_handler_without_callback_falls_back_to_env(monkeypatch):
    """No registered callback + a closure handler (no runner) must not raise and
    falls back to env-only auth — the getattr guard keeps the legacy path safe."""
    monkeypatch.setenv("TELEGRAM_GROUP_ALLOWED_CHATS", "-100123")
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "111")

    adapter = _make_adapter(group_allowed_chats=["-100123"])
    adapter._message_handler = lambda event: None  # closure, no __self__
    # No set_authorization_check() → _authorization_check is absent/None.

    assert adapter._is_user_authorized_from_message(
        _make_message(from_user_id=111, chat_id=-100123, chat_type="group")
    ) is True
    assert adapter._is_user_authorized_from_message(
        _make_message(from_user_id=555, chat_id=-100123, chat_type="group")
    ) is False
