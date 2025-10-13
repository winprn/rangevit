"""
Utility helpers for sending Discord webhook notifications.
"""

from __future__ import annotations

import datetime
import os
from pathlib import Path
from typing import Optional

try:
    import requests
except ImportError:  # pragma: no cover - optional dependency
    requests = None  # type: ignore[assignment]


_ENV_LOADED = False


def _load_env_file() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True

    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.is_file():
        return

    try:
        for raw_line in env_path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            key, sep, value = line.partition("=")
            if not sep:
                continue
            key = key.strip()
            value = value.strip().strip("'\"")
            os.environ.setdefault(key, value)
    except Exception as exc:
        print(f"Failed to load .env file ({env_path}): {exc}")


def _resolve_webhook_url(explicit_url: Optional[str]) -> Optional[str]:
    """
    Pick the webhook URL, giving priority to the explicit value, then the env var.
    Returning None disables notifications.
    """
    if explicit_url:
        return explicit_url
    _load_env_file()
    env_url = os.getenv("DISCORD_WEBHOOK_URL")
    if env_url:
        return env_url
    return None


def _format_duration(seconds: Optional[float]) -> Optional[str]:
    if seconds is None:
        return None
    seconds = max(0.0, float(seconds))
    delta = datetime.timedelta(seconds=int(seconds))
    return str(delta)


def post_message(
    content: str,
    *,
    username: Optional[str] = None,
    webhook_url: Optional[str] = None,
) -> bool:
    """
    Send a raw Discord webhook message.

    Returns True when the request succeeds, False otherwise.
    """
    resolved_url = _resolve_webhook_url(webhook_url)
    if not resolved_url:
        print("Discord notification skipped: set DISCORD_WEBHOOK_URL in your environment or .env file.")
        return False

    if requests is None:
        print("Discord notification skipped: 'requests' package is not available.")
        return False

    payload: dict[str, object] = {"content": content}
    if username:
        payload["username"] = username

    try:
        response = requests.post(resolved_url, json=payload, timeout=10)
        response.raise_for_status()
        return True
    except Exception as exc:  # pragma: no cover - network errors not unit tested
        print(f"Discord notification failed: {exc}")
        return False


def notify_run_completion(
    *,
    task_name: str,
    success: bool,
    elapsed_seconds: Optional[float] = None,
    extra_message: Optional[str] = None,
    webhook_url: Optional[str] = None,
) -> bool:
    """
    High-level helper to announce the completion status of a long-running task.
    """
    status = "completed" if success else "failed"
    pieces = [f"`{task_name}` {status}"]

    duration_text = _format_duration(elapsed_seconds)
    if duration_text:
        pieces.append(f"in {duration_text}")

    message = " ".join(pieces)
    if extra_message:
        message = f"{message}\n{extra_message}"

    return post_message(message, username="RangeViT Bot", webhook_url=webhook_url)


__all__ = ["notify_run_completion", "post_message"]
