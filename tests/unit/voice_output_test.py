"""Behavioral tests for HearHear's shared voice-output safety contract."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import voice_output  # noqa: E402


def test_missing_state_allows_voice_output(tmp_path):
    status = voice_output.voice_output_status(tmp_path / "missing.json")

    assert status.allowed is True
    assert status.reason == "active"


def test_persistent_pause_is_read_by_a_fresh_process(tmp_path):
    state = tmp_path / "voice-output-state.json"
    state.write_text(
        json.dumps({"paused": True, "updatedAt": "2026-09-23T18:30:00Z", "updatedBy": "HearHear"}),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "voice_output.py"), "--check", "--stage", "test"],
        check=False,
        capture_output=True,
        text=True,
        env={"HEARHEAR_VOICE_OUTPUT_STATE_PATH": str(state)},
    )

    assert result.returncode == 75
    assert "skipped-global-pause" in result.stderr


def test_malformed_and_unreadable_state_fail_closed(tmp_path):
    malformed = tmp_path / "malformed.json"
    malformed.write_text('{"paused":"yes"}', encoding="utf-8")

    malformed_status = voice_output.voice_output_status(malformed)
    with patch.object(Path, "read_text", side_effect=PermissionError("denied")):
        unreadable_status = voice_output.voice_output_status(tmp_path / "unreadable.json")

    assert malformed_status.allowed is False
    assert malformed_status.reason == "state-error"
    assert unreadable_status.allowed is False
    assert unreadable_status.reason == "state-error"


def test_audio_lock_is_exclusive_across_processes(tmp_path, monkeypatch):
    lock_path = tmp_path / "audio-turn-taking.lock"
    monkeypatch.setenv("HEARHEAR_AUDIO_LOCK_PATH", str(lock_path))
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import fcntl, pathlib, sys; "
                "p=pathlib.Path(sys.argv[1]); p.touch(); "
                "f=p.open('a+'); fcntl.flock(f.fileno(), fcntl.LOCK_EX); "
                "print('locked', flush=True); sys.stdin.readline()"
            ),
            str(lock_path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "locked"
        assert voice_output.audio_turn_available() is False
    finally:
        if child.stdin is not None:
            child.stdin.write("release\n")
            child.stdin.flush()
        child.wait(timeout=5)

    assert voice_output.audio_turn_available() is True
