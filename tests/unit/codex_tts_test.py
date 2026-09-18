"""Behavioral tests for the Codex Stop-hook and Luna pipeline."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import sys
from unittest.mock import Mock

import pytest


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import codex_tts  # noqa: E402
from tts_pipeline import LunaClient, ProviderSettings, build_summary_prompt  # noqa: E402
import tts_pipeline  # noqa: E402


class FakeHTTPResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class TestCodexPayload:
    def test_uses_last_assistant_message_and_turn_identity(self):
        turn = codex_tts.normalize_payload(
            {
                "hook_event_name": "Stop",
                "session_id": "session-1",
                "turn_id": "turn-9",
                "last_assistant_message": "The tests pass.",
                "transcript_path": "/tmp/rollout.jsonl",
            }
        )

        assert turn.source == "codex"
        assert turn.session_id == "session-1"
        assert turn.turn_id == "turn-9"
        assert turn.response_text == "The tests pass."
        assert turn.event_key == "codex:session-1:turn-9"

    def test_derives_stable_turn_id_when_codex_omits_one(self):
        payload = {
            "session_id": "session-1",
            "last_assistant_message": "The tests pass.",
        }

        first = codex_tts.normalize_payload(payload)
        second = codex_tts.normalize_payload(payload)

        assert first.turn_id == second.turn_id
        assert first.turn_id.startswith("derived-")

    def test_transcript_fallback_prefers_final_answer(self, tmp_path):
        transcript = tmp_path / "rollout.jsonl"
        transcript.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "type": "response_item",
                            "turn_id": "turn-1",
                            "payload": {
                                "type": "message",
                                "role": "assistant",
                                "phase": "commentary",
                                "content": [{"type": "output_text", "text": "Working."}],
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "type": "response_item",
                            "turn_id": "turn-1",
                            "payload": {
                                "type": "message",
                                "role": "assistant",
                                "phase": "final_answer",
                                "content": [
                                    {"type": "output_text", "text": "The hook is ready."}
                                ],
                            },
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        assert (
            codex_tts.extract_transcript_response(str(transcript), "turn-1")
            == "The hook is ready."
        )

    def test_transcript_fallback_does_not_speak_a_different_turn(self, tmp_path):
        transcript = tmp_path / "rollout.jsonl"
        transcript.write_text(
            json.dumps(
                {
                    "type": "response_item",
                    "turn_id": "turn-2",
                    "payload": {
                        "type": "message",
                        "role": "assistant",
                        "phase": "final_answer",
                        "content": [{"type": "output_text", "text": "Later turn."}],
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )

        assert codex_tts.extract_transcript_response(str(transcript), "turn-1") == ""


class TestLunaClient:
    def test_request_uses_luna_and_low_reasoning_effort(self, monkeypatch):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["body"] = json.loads(request.data.decode("utf-8"))
            captured["timeout"] = timeout
            return FakeHTTPResponse(
                {
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": "Tests pass."}],
                        }
                    ]
                }
            )

        monkeypatch.setattr(tts_pipeline.urllib.request, "urlopen", fake_urlopen)
        client = LunaClient(
            ProviderSettings("https://example.test/v1/responses", "test-token"),
            timeout=7,
        )

        assert client.summarize("The implementation is complete.") == "Tests pass."
        assert captured["url"].endswith("/responses")
        assert captured["body"]["model"] == "gpt-5.6-luna"
        assert captured["body"]["reasoning"] == {"effort": "low"}
        assert captured["timeout"] == 7
        assert "test-token" in str(client.settings)

    def test_prompt_preserves_language_and_identifiers(self):
        prompt = build_summary_prompt("La prueba pasó en api/v1/users.")

        assert "same natural language" in prompt
        assert "api/v1/users" in prompt
        assert "Do not use markdown" in prompt

    def test_say_failure_is_reported(self, monkeypatch):
        monkeypatch.setattr(
            tts_pipeline.subprocess,
            "run",
            Mock(return_value=Mock(returncode=1)),
        )

        with pytest.raises(RuntimeError, match="macOS say exited"):
            tts_pipeline.speak_with_say("Hello")


class TestTurnProcessing:
    def test_claims_turn_and_speaks_luna_summary(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CODEX_TTS_STATE_DIR", str(tmp_path / "state"))
        monkeypatch.setattr(codex_tts, "is_muted", lambda: False)
        summarize = Mock(return_value="The Codex hook is ready.")
        speak = Mock(return_value="say")
        monkeypatch.setattr(codex_tts, "summarize_with_luna", summarize)
        monkeypatch.setattr(codex_tts, "speak_text", speak)
        turn = codex_tts.NormalizedTurn(
            source="codex",
            session_id="session-1",
            turn_id="turn-1",
            response_text="A long response that should be summarized.",
        )

        result = codex_tts.process_turn(turn)

        assert result == "say"
        summarize.assert_called_once_with(
            turn.response_text,
            language_hint=None,
        )
        speak.assert_called_once_with("The Codex hook is ready.")
        assert codex_tts.process_turn(turn) == "skipped-duplicate"
        assert speak.call_count == 1

    def test_summary_failure_falls_back_to_excerpt(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CODEX_TTS_STATE_DIR", str(tmp_path / "state"))
        monkeypatch.setattr(codex_tts, "is_muted", lambda: False)
        monkeypatch.setattr(
            codex_tts,
            "summarize_with_luna",
            Mock(side_effect=RuntimeError("provider unavailable")),
        )
        speak = Mock(return_value="say")
        monkeypatch.setattr(codex_tts, "speak_text", speak)
        turn = codex_tts.NormalizedTurn(
            source="codex",
            session_id="session-2",
            turn_id="turn-2",
            response_text="The implementation is complete. The optional server is still offline.",
        )

        assert codex_tts.process_turn(turn) == "say"
        speak.assert_called_once_with("The implementation is complete.")


class TestHookEntrypoint:
    def test_hook_returns_empty_json_without_waiting_for_worker(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CODEX_TTS_STATE_DIR", str(tmp_path / "state"))
        payload = {
            "hook_event_name": "Stop",
            "session_id": "session-1",
            "turn_id": "turn-1",
            "last_assistant_message": "Done.",
        }
        launch = Mock(return_value=True)
        monkeypatch.setattr(codex_tts, "launch_worker", launch)
        stdin = io.StringIO(json.dumps(payload))
        stdout = io.StringIO()
        monkeypatch.setattr(sys, "stdin", stdin)
        monkeypatch.setattr(sys, "stdout", stdout)

        assert codex_tts.hook_main() == 0

        assert stdout.getvalue() == "{}\n"
        launch.assert_called_once_with(payload)

    def test_hook_does_not_launch_when_stop_hook_is_already_active(self, monkeypatch):
        launch = Mock()
        monkeypatch.setattr(codex_tts, "launch_worker", launch)
        monkeypatch.setattr(
            sys,
            "stdin",
            io.StringIO(json.dumps({"stop_hook_active": True})),
        )
        stdout = io.StringIO()
        monkeypatch.setattr(sys, "stdout", stdout)

        assert codex_tts.hook_main() == 0
        assert stdout.getvalue() == "{}\n"
        launch.assert_not_called()
