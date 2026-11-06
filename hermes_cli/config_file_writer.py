"""Comment-preserving configuration publication under shared native coordination."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def _same_values(left: Any, right: Any) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _same_values(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _same_values(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def write_config_state(
    config_path: Path, data: dict[str, Any], *, allow_omissions: bool,
    extra_content_on_create: str | None = None,
    expected_state: dict[str, Any] | None = None,
) -> None:
    """Validate an optional preimage while holding the native file transaction."""
    from hermes_cli import config
    from utils import atomic_roundtrip_yaml_save

    config_path = Path(config_path)
    if expected_state is not None and not isinstance(expected_state, dict):
        raise TypeError("expected configuration state must be a mapping")
    with config._env_write_lock(config_path):
        config._refuse_failed_read(config_path, data)
        if expected_state is not None or not allow_omissions:
            existing = config.require_readable_config_before_write(config_path)
            if expected_state is not None and not _same_values(existing, expected_state):
                raise ValueError("configuration changed before native publication; reload the plan")
            if not allow_omissions:
                omitted = config._omitted_config_paths(existing, data)
                if omitted:
                    shown = ", ".join(omitted[:12])
                    if len(omitted) > 12:
                        shown += f", +{len(omitted) - 12} more"
                    error = ValueError(f"omitted config paths: {shown}")
                    raise config._refuse_overwrite(
                        config_path, "would lose settings omitted by this write", error,
                        "Pass the complete current config, or use atomic_config_replace() "
                        "only when deletion by omission is deliberate.",
                    ) from error
        atomic_roundtrip_yaml_save(
            config_path, data, extra_content_on_create=extra_content_on_create)
