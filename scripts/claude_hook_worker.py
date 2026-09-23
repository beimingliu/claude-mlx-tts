#!/usr/bin/env python3
"""Detach one Claude voice hook while preserving its stdin payload privately."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from voice_output import suppression_reason


def _pending_dir() -> Path:
    override = os.environ.get("CLAUDE_TTS_HOOK_STATE_DIR", "").strip()
    return Path(override).expanduser() if override else Path.home() / ".local/state/claude-mlx-tts/claude"


def launch(target: str) -> int:
    if suppression_reason("before-worker-spawn") is not None:
        return 0
    directory = _pending_dir()
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(directory, 0o700)
    descriptor, payload_path = tempfile.mkstemp(prefix="hook-", suffix=".json", dir=directory)
    try:
        with os.fdopen(descriptor, "wb") as payload:
            payload.write(sys.stdin.buffer.read())
        os.chmod(payload_path, 0o600)
        log_path = directory / "voice-hooks.log"
        log_handle = log_path.open("ab")
        try:
            subprocess.Popen(
                [sys.executable, target, "--payload", payload_path],
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=log_handle,
                close_fds=True,
                start_new_session=True,
            )
        finally:
            log_handle.close()
    except BaseException:
        try:
            os.unlink(payload_path)
        except OSError:
            pass
        raise
    return 0


def read_hook_input() -> dict:
    """Read a detached payload, deleting it before hook processing begins."""

    payload_path = ""
    if len(sys.argv) == 3 and sys.argv[1] == "--payload":
        payload_path = sys.argv[2]
    try:
        if payload_path:
            with open(payload_path, encoding="utf-8") as payload:
                value = json.load(payload)
        else:
            value = json.load(sys.stdin)
    except (OSError, EOFError, json.JSONDecodeError):
        return {}
    finally:
        if payload_path:
            try:
                os.unlink(payload_path)
            except OSError:
                pass
    return value if isinstance(value, dict) else {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target")
    args = parser.parse_args(argv)
    try:
        return launch(args.target)
    except (OSError, ValueError) as exc:
        print(f"voice hook worker was not launched: {exc}", file=sys.stderr)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
