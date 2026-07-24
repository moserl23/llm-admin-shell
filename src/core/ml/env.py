"""Utilities for loading project-level environment variables from a local `.env`.

This keeps experiments self-contained by preferring the current working directory
and falling back to the repository root when code is executed from a subdirectory.
"""

from os import environ
from pathlib import Path

_LOADED = False


def load_project_env() -> None:
    """Load `.env` values into the process environment once.

    The loader prefers the current working directory and falls back to the
    repository root. Existing environment variables are preserved.
    """
    global _LOADED
    if _LOADED:
        return

    # Prefer a run-local `.env` so experiments can override repository defaults.
    env_path = Path.cwd() / ".env"

    # When invoked from a subdirectory, recover the repository-level `.env`.
    if not env_path.exists():
        env_path = Path(__file__).resolve().parents[3] / ".env"

    if not env_path.exists():
        _LOADED = True
        return

    # Parse simple KEY=VALUE entries and ignore comments or malformed lines.
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key:
            continue

        # Strip matching quotes so quoted values behave like unquoted shell entries.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]

        environ.setdefault(key, value)

    _LOADED = True
