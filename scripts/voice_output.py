#!/usr/bin/env python3
"""Shared HearHear voice-output pause state and audio turn-taking lock.

This module intentionally uses only the Python standard library so hook
launchers can consult it before creating an environment or importing TTS/LLM
dependencies.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import argparse
import fcntl
import json
import logging
import os
from pathlib import Path
from typing import Iterator


# This is a child of the Codex adapter logger, while still propagating to the
# root logger configured by Claude's plugin entry points.
log = logging.getLogger("codex-mlx-tts.voice-output")


@dataclass(frozen=True)
class VoiceOutputStatus:
    allowed: bool
    reason: str
    detail: str = ""


def state_path() -> Path:
    override = os.environ.get("HEARHEAR_VOICE_OUTPUT_STATE_PATH", "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / "Library/Application Support/HearHear/voice-output-state.json"


def audio_lock_path() -> Path:
    override = os.environ.get("HEARHEAR_AUDIO_LOCK_PATH", "").strip()
    if not override:
        # Keep the existing Codex-only override working for local tests/users.
        override = os.environ.get("CODEX_TTS_AUDIO_LOCK_PATH", "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / "Library/Application Support/HearHear/audio-turn-taking.lock"


def voice_output_status(path: Path | None = None) -> VoiceOutputStatus:
    """Read shared pause state, failing closed on any invalid existing file."""

    target = path or state_path()
    try:
        raw = target.read_text(encoding="utf-8")
    except FileNotFoundError:
        if target.is_symlink():
            return VoiceOutputStatus(False, "state-error", f"{target}: broken symbolic link")
        return VoiceOutputStatus(True, "active")
    except (OSError, UnicodeError) as exc:
        return VoiceOutputStatus(False, "state-error", f"{target}: {exc}")

    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        return VoiceOutputStatus(False, "state-error", f"{target}: invalid JSON ({exc})")
    if not isinstance(payload, dict) or not isinstance(payload.get("paused"), bool):
        return VoiceOutputStatus(
            False,
            "state-error",
            f"{target}: expected a JSON object with a boolean 'paused' field",
        )
    if payload["paused"]:
        return VoiceOutputStatus(False, "global-pause")
    return VoiceOutputStatus(True, "active")


def _log_suppression(status: VoiceOutputStatus, stage: str) -> None:
    if status.allowed:
        return
    suffix = f" at {stage}" if stage else ""
    if status.reason == "global-pause":
        log.info("skipped-global-pause%s", suffix)
    else:
        log.error("skipped-global-state-error%s: %s", suffix, status.detail)


def voice_output_allowed(stage: str = "") -> bool:
    status = voice_output_status()
    _log_suppression(status, stage)
    if status.allowed:
        return True
    return False


def state_suppression_reason(stage: str = "") -> str | None:
    """Return/log only shared-state suppression without probing the audio lock."""

    status = voice_output_status()
    if status.allowed:
        return None
    _log_suppression(status, stage)
    return "skipped-global-pause" if status.reason == "global-pause" else "skipped-global-state-error"


@contextmanager
def audio_turn_lock() -> Iterator[bool]:
    """Try to own the microphone/playback lock until the context exits."""

    path = audio_lock_path()
    handle = None
    try:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not os.environ.get("HEARHEAR_AUDIO_LOCK_PATH", "").strip() \
                and not os.environ.get("CODEX_TTS_AUDIO_LOCK_PATH", "").strip():
            os.chmod(path.parent, 0o700)
        handle = path.open("a+", encoding="utf-8")
        os.chmod(path, 0o600)
    except OSError as exc:
        if handle is not None:
            handle.close()
        log.error("skipped-audio-lock-error: %s: %s", path, exc)
        yield False
        return

    with handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def audio_turn_available() -> bool:
    with audio_turn_lock() as available:
        return available


def suppression_reason(stage: str = "") -> str | None:
    """Return a stable reason if shared state or the audio lock blocks work."""

    state_reason = state_suppression_reason(stage)
    if state_reason:
        return state_reason
    if not audio_turn_available():
        log.info("skipped-audio-busy%s", f" at {stage}" if stage else "")
        return "skipped-audio-busy"
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="exit nonzero when voice output is suppressed")
    parser.add_argument("--stage", default="launcher", help="diagnostic boundary name")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.check:
        return 0 if suppression_reason(args.stage) is None else 75
    parser.error("choose --check")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
