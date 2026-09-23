"""Behavioral tests for detached Claude voice-hook launch."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import stat
import sys
from unittest.mock import Mock


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import claude_hook_worker  # noqa: E402


class BinaryInput:
    def __init__(self, value: bytes):
        self.buffer = io.BytesIO(value)


def test_pause_skips_payload_and_worker_creation(monkeypatch, tmp_path):
    state = tmp_path / "voice-output-state.json"
    state.write_text('{"paused":true}', encoding="utf-8")
    pending = tmp_path / "pending"
    monkeypatch.setenv("HEARHEAR_VOICE_OUTPUT_STATE_PATH", str(state))
    monkeypatch.setenv("HEARHEAR_AUDIO_LOCK_PATH", str(tmp_path / "audio.lock"))
    monkeypatch.setenv("CLAUDE_TTS_HOOK_STATE_DIR", str(pending))
    monkeypatch.setattr(claude_hook_worker.subprocess, "Popen", Mock())

    assert claude_hook_worker.launch("hook.py") == 0
    assert not pending.exists()
    claude_hook_worker.subprocess.Popen.assert_not_called()


def test_launch_persists_private_payload_for_detached_worker(monkeypatch, tmp_path):
    pending = tmp_path / "pending"
    monkeypatch.setenv("HEARHEAR_VOICE_OUTPUT_STATE_PATH", str(tmp_path / "missing.json"))
    monkeypatch.setenv("HEARHEAR_AUDIO_LOCK_PATH", str(tmp_path / "audio.lock"))
    monkeypatch.setenv("CLAUDE_TTS_HOOK_STATE_DIR", str(pending))
    monkeypatch.setattr(sys, "stdin", BinaryInput(b'{"hook_event_name":"Stop"}'))
    process = Mock()
    monkeypatch.setattr(claude_hook_worker.subprocess, "Popen", process)

    assert claude_hook_worker.launch("hook.py") == 0

    command = process.call_args.args[0]
    payload = Path(command[-1])
    assert command[-3:-1] == ["hook.py", "--payload"]
    assert json.loads(payload.read_text(encoding="utf-8"))["hook_event_name"] == "Stop"
    assert stat.S_IMODE(payload.stat().st_mode) == 0o600
    assert stat.S_IMODE(pending.stat().st_mode) == 0o700
    payload.unlink()


def test_worker_read_deletes_payload(monkeypatch, tmp_path):
    payload = tmp_path / "payload.json"
    payload.write_text('{"tool_name":"Bash"}', encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["hook.py", "--payload", str(payload)])

    result = claude_hook_worker.read_hook_input()

    assert result == {"tool_name": "Bash"}
    assert not payload.exists()
