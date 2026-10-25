"""Regression for #125695: a rejected Codex model must not disable unrelated auxiliary work."""
import json
import time
from pathlib import Path

import pytest


@pytest.fixture
def codex_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (tmp_path / "config.yaml").write_text(
        "model:\n  provider: openai-codex\n  default: gpt-6.1-sol\n"
        "auxiliary:\n  compression:\n    provider: auto\n    model: \"\"\n"
    )
    return tmp_path


def _write_credential(home, cooled_model):
    (home / "auth.json").write_text(json.dumps({"credential_pool": {"openai-codex": [{
        "id": "synthetic-account", "label": "synthetic", "auth_type": "oauth",
        "priority": 0, "source": "manual", "access_token": "synthetic-access-token",
        "refresh_token": "synthetic-refresh-token", "expires_at_ms": 4_000_000_000_000,
        "model_cooldowns": {cooled_model: time.time() + 365 * 24 * 3600},
    }]}}))


@pytest.mark.parametrize("raw_codex", [False, True])
def test_unrelated_entitlement_does_not_disable_auto_compression(codex_home, raw_codex):
    from agent.auxiliary_client import resolve_provider_client
    _write_credential(codex_home, "gpt-5.5")
    client, model = resolve_provider_client(
        "auto", "", raw_codex=raw_codex,
        main_runtime={"provider": "openai-codex", "model": "gpt-6.1-sol"},
    )
    assert client is not None
    assert model == "gpt-6.1-sol"
    assert client._hermes_aux_effective_provider == "openai-codex"


@pytest.mark.parametrize("raw_codex", [False, True])
@pytest.mark.parametrize("model", ["gpt-6.1-sol", "openai/gpt-6.1-sol", "openai/gpt-6.1-sol-900k"])
def test_requested_model_entitlement_remains_enforced(codex_home, raw_codex, model):
    from agent.auxiliary_client import resolve_provider_client
    _write_credential(codex_home, "gpt-6.1-sol")
    client, resolved = resolve_provider_client("openai-codex", model, raw_codex=raw_codex)
    assert client is None and resolved is None
