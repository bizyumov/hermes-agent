"""Conditional native config publication preserves intervening operator updates."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from hermes_cli.plugins_discovery import suppress_plugin_discovery
from tests.waiting import wait_until

with suppress_plugin_discovery():
    from hermes_cli import config


def test_stale_expected_state_preserves_new_settings_and_comments(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("# keep operator comment\nmodel: before\n", encoding="utf-8")
    before = {"model": "before"}
    incoming = {"model": "operator-choice", "unrelated": {"preserve": True}}
    config.atomic_config_replace(path, incoming)
    with pytest.raises(ValueError, match="changed"):
        config.atomic_config_replace(path, {"model": "native-choice"}, expected_state=before)
    assert config.fast_safe_load(path.read_text()) == incoming
    assert "# keep operator comment" in path.read_text()


def test_expected_state_comparison_keeps_boolean_integer_types_distinct(tmp_path):
    path = tmp_path / "config.yaml"
    config.atomic_config_replace(path, {"setting": 1})
    with pytest.raises(ValueError, match="changed"):
        config.atomic_config_replace(path, {"setting": 2}, expected_state={"setting": True})
    assert config.fast_safe_load(path.read_text()) == {"setting": 1}


def test_shared_native_write_coordination_rechecks_expected_state(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    before = {"model": "before"}
    incoming = {"model": "operator-choice", "new_field": "preserve"}
    config.atomic_config_replace(path, before)
    ready = tmp_path / "writer-ready"
    program = (
        "import json,sys\nfrom pathlib import Path\n"
        "from hermes_cli.plugins_discovery import suppress_plugin_discovery\n"
        "with suppress_plugin_discovery():\n from hermes_cli import config\n"
        "path=Path(sys.argv[1])\n"
        "with config._env_write_lock(path):\n"
        " Path(sys.argv[2]).write_text('ready')\n"
        " sys.stdin.readline()\n"
        " config.atomic_config_replace(path,json.loads(sys.argv[3]))\n"
    )
    child = subprocess.Popen([sys.executable, "-c", program, str(path), str(ready),
                              json.dumps(incoming)], env=dict(os.environ),
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True)
    attempted = threading.Event()
    errors = []
    original = config._env_write_lock

    @contextmanager
    def observed_lock(target):
        attempted.set()
        with original(target):
            yield

    def replace():
        try:
            config.atomic_config_replace(path, {"model": "native-choice"}, expected_state=before)
        except Exception as error:
            errors.append(error)

    worker = threading.Thread(target=replace)
    try:
        wait_until(ready.exists, timeout=10, what="other native writer holding configuration")
        monkeypatch.setattr(config, "_env_write_lock", observed_lock)
        worker.start()
        assert attempted.wait(10), "conditional writer did not enter native coordination"
    finally:
        child.communicate(input="release\n", timeout=10)
        if worker.ident is not None:
            worker.join(timeout=10)
        if child.poll() is None:
            child.terminate()
            child.communicate(timeout=5)
    assert child.returncode == 0
    assert not worker.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], ValueError)
    assert config.fast_safe_load(path.read_text()) == incoming
