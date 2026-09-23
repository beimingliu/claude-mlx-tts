"""Behavioral safety tests for the Claude Stop voice hook."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import Mock

import pytest


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("tts_notify", SCRIPTS / "tts-notify.py")
assert spec is not None and spec.loader is not None
tts_notify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tts_notify)


@pytest.fixture(autouse=True)
def isolate_hearhear_contract(monkeypatch, tmp_path):
    monkeypatch.setenv("HEARHEAR_VOICE_OUTPUT_STATE_PATH", str(tmp_path / "voice-state.json"))
    monkeypatch.setenv("HEARHEAR_AUDIO_LOCK_PATH", str(tmp_path / "audio.lock"))


def test_pause_skips_stop_hook_before_summary(monkeypatch, tmp_path):
    state = tmp_path / "voice-state.json"
    state.write_text(json.dumps({"paused": True}), encoding="utf-8")
    summarize = Mock()
    speak = Mock()
    monkeypatch.setattr(tts_notify, "get_hook_input", lambda: {"transcript_path": "unused"})
    monkeypatch.setattr(tts_notify, "summarize", summarize)
    monkeypatch.setattr(tts_notify, "speak", speak)

    assert tts_notify.main() is None
    summarize.assert_not_called()
    speak.assert_not_called()


def test_playback_backend_runs_while_audio_lock_is_owned(monkeypatch):
    observed = []

    def inspect_lock(_message):
        from voice_output import audio_turn_available
        observed.append(audio_turn_available())

    monkeypatch.setattr(tts_notify, "is_mlx_available", lambda: False)
    monkeypatch.setattr(tts_notify, "speak_say", inspect_lock)

    assert tts_notify.speak("Finished") is True
    assert observed == [False]


def test_pause_set_after_summary_wins_before_playback(monkeypatch, tmp_path):
    state = tmp_path / "voice-state.json"
    state.write_text('{"paused":false}', encoding="utf-8")
    monkeypatch.setattr(tts_notify, "get_hook_input", lambda: {"transcript_path": "transcript"})
    monkeypatch.setattr(
        tts_notify,
        "should_trigger_tts",
        lambda _path: (True, "A completed response", 2, 20.0, False),
    )

    def summarize_and_pause(_text):
        state.write_text('{"paused":true}', encoding="utf-8")
        return "Finished."

    backend = Mock()
    monkeypatch.setattr(tts_notify, "summarize", summarize_and_pause)
    monkeypatch.setattr(tts_notify, "is_mlx_available", lambda: False)
    monkeypatch.setattr(tts_notify, "speak_say", backend)
    import tts_mute
    monkeypatch.setattr(tts_mute, "is_muted", lambda: False)

    assert tts_notify.main() is None
    backend.assert_not_called()
