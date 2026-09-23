"""Behavioral tests for the Codex Stop-hook and Luna pipeline."""

from __future__ import annotations

import io
import json
import os
import fcntl
from pathlib import Path
import sys
from unittest.mock import Mock

import pytest


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import codex_tts  # noqa: E402
from tts_pipeline import (  # noqa: E402
    LunaClient,
    ProviderSettings,
    SummaryResult,
    build_summary_prompt,
)
import tts_pipeline  # noqa: E402


@pytest.fixture(autouse=True)
def isolate_hearhear_contract(monkeypatch, tmp_path):
    """Never let a developer's live pause/recording state affect unit tests."""

    monkeypatch.setenv(
        "HEARHEAR_VOICE_OUTPUT_STATE_PATH",
        str(tmp_path / "voice-output-state.json"),
    )
    monkeypatch.setenv(
        "HEARHEAR_AUDIO_LOCK_PATH",
        str(tmp_path / "audio-turn-taking.lock"),
    )


class FakeHTTPResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class FakeAudioHTTPResponse:
    def __init__(self, audio: bytes):
        self.audio = audio

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return self.audio


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
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": json.dumps(
                                        {
                                            "metadata": {"summary_language": "English"},
                                            "summary": "Tests pass.",
                                        }
                                    ),
                                }
                            ],
                        }
                    ]
                }
            )

        monkeypatch.setattr(tts_pipeline.urllib.request, "urlopen", fake_urlopen)
        client = LunaClient(
            ProviderSettings("https://example.test/v1/responses", "test-token"),
            timeout=7,
        )

        summary = client.summarize("The implementation is complete.")

        assert summary == SummaryResult("Tests pass.", "English")
        assert captured["url"].endswith("/responses")
        assert captured["body"]["model"] == "gpt-5.6-luna"
        assert captured["body"]["reasoning"] == {"effort": "low"}
        assert captured["timeout"] == 7
        assert "test-token" in str(client.settings)

    def test_prompt_preserves_language_and_identifiers(self):
        prompt = build_summary_prompt("La prueba pasó en api/v1/users.")

        assert "same natural language" in prompt
        assert "api/v1/users" in prompt
        assert "Do not include greetings, introductions, Markdown" in prompt
        assert '"summary_language":"Chinese or English"' in prompt
        assert "15–35 words" in prompt

    def test_summary_metadata_selects_chinese(self):
        summary = tts_pipeline.parse_summary_result(
            json.dumps(
                {
                    "metadata": {"summary_language": "Chinese"},
                    "summary": "中文摘要已经生成。",
                }
            )
        )

        assert summary == SummaryResult("中文摘要已经生成。", "Chinese")

    def test_say_failure_is_reported(self, monkeypatch):
        monkeypatch.setattr(
            tts_pipeline.subprocess,
            "run",
            Mock(return_value=Mock(returncode=1)),
        )

        with pytest.raises(RuntimeError, match="macOS say exited"):
            tts_pipeline.speak_with_say("Hello")

    def test_chinese_system_fallback_uses_chinese_voice(self, monkeypatch):
        played = Mock(return_value=Mock(returncode=0))
        monkeypatch.setattr(tts_pipeline.subprocess, "run", played)

        tts_pipeline.speak_with_say("中文摘要。", language_hint="Chinese")

        assert played.call_args.args[0][2:4] == ["Tingting", "-r"]

    def test_qwen_request_plays_wav_and_cleans_up_temp_file(self, monkeypatch):
        captured = {}
        wav = b"RIFF\x00\x00\x00\x00WAVE" + b"audio"

        def fake_urlopen(request, timeout):
            captured.setdefault("requests", []).append(request)
            captured["timeout"] = timeout
            if request.method == "POST":
                captured["body"] = request.data.decode("utf-8")
                return FakeHTTPResponse({"audio_url": "/audio/chinese.wav"})
            return FakeAudioHTTPResponse(wav)

        played = Mock(return_value=Mock(returncode=0))
        monkeypatch.setattr(tts_pipeline.urllib.request, "urlopen", fake_urlopen)
        monkeypatch.setattr(tts_pipeline.subprocess, "run", played)
        monkeypatch.setenv("CODEX_TTS_QWEN_CLONE_URL", "http://127.0.0.1:9999/api/clone")
        monkeypatch.setenv("CODEX_TTS_QWEN_TIMEOUT", "9")

        tts_pipeline.speak_with_qwen("中文摘要。", language_hint="Chinese")

        assert captured["requests"][0].full_url == "http://127.0.0.1:9999/api/clone"
        assert 'name="text"' in captured["body"]
        assert "中文摘要。" in captured["body"]
        assert 'name="language"' in captured["body"]
        assert "Chinese" in captured["body"]
        assert 'name="reference_id"' in captured["body"]
        assert "陈晓卿-香料与人类进步" in captured["body"]
        assert captured["requests"][1].full_url == "http://127.0.0.1:9999/audio/chinese.wav"
        assert captured["timeout"] == 9
        played.assert_called_once()
        assert played.call_args.args[0][0] == "afplay"
        assert not Path(played.call_args.args[0][1]).exists()

    def test_qwen_failure_falls_back_to_system_speech(self, monkeypatch):
        monkeypatch.setenv("CODEX_TTS_BACKEND", "qwen")
        monkeypatch.setattr(
            tts_pipeline,
            "speak_with_qwen",
            Mock(side_effect=tts_pipeline.QwenRequestError("service offline")),
        )
        say = Mock()
        monkeypatch.setattr(tts_pipeline, "speak_with_say", say)

        result = tts_pipeline.speak_text("The hook is ready.")

        assert result == "say"
        say.assert_called_once_with("The hook is ready.", language_hint=None)


class TestTurnProcessing:
    @pytest.fixture(autouse=True)
    def disable_cmux_context_by_default(self, monkeypatch, tmp_path):
        """Keep non-CMUX behavior independent of the test runner's shell."""

        monkeypatch.setenv("CODEX_TTS_INCLUDE_SESSION_NAME", "false")
        monkeypatch.setenv(
            "CODEX_TTS_AUDIO_LOCK_PATH",
            str(tmp_path / "audio-turn-taking.lock"),
        )

    def test_microphone_lock_skips_summary_and_speech(self, monkeypatch, tmp_path):
        lock_path = tmp_path / "audio-turn-taking.lock"
        monkeypatch.setenv("CODEX_TTS_AUDIO_LOCK_PATH", str(lock_path))
        summarize = Mock()
        speak = Mock()
        monkeypatch.setattr(codex_tts, "summarize_with_luna", summarize)
        monkeypatch.setattr(codex_tts, "speak_text", speak)
        turn = codex_tts.NormalizedTurn(
            source="codex",
            session_id="session-mic",
            turn_id="turn-mic",
            response_text="This must not be summarized while recording.",
        )

        with lock_path.open("a+") as microphone:
            fcntl.flock(microphone.fileno(), fcntl.LOCK_EX)
            assert codex_tts.process_turn(turn) == "skipped-audio-busy"

        summarize.assert_not_called()
        speak.assert_not_called()

    def test_attention_prefix_is_available_when_enabled(self, monkeypatch):
        monkeypatch.setenv("CODEX_TTS_INCLUDE_ATTENTION", "true")
        monkeypatch.setenv("CODEX_TTS_ATTENTION_PREFIX", "注意")

        assert codex_tts._attention_prefix() == "注意"

    def test_disabled_flag_skips_summary_and_speech(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CODEX_TTS_ENABLED", "false")
        monkeypatch.setenv("CODEX_TTS_STATE_DIR", str(tmp_path / "state"))
        summarize = Mock()
        speak = Mock()
        monkeypatch.setattr(codex_tts, "summarize_with_luna", summarize)
        monkeypatch.setattr(codex_tts, "speak_text", speak)
        turn = codex_tts.NormalizedTurn(
            source="codex",
            session_id="session-disabled",
            turn_id="turn-disabled",
            response_text="This should not be summarized or spoken.",
        )

        assert codex_tts.process_turn(turn) == "skipped-disabled"
        summarize.assert_not_called()
        speak.assert_not_called()

    def test_global_pause_skips_expensive_summary_and_speech(self, monkeypatch, tmp_path):
        state = tmp_path / "voice-output-state.json"
        state.write_text('{"paused":true}', encoding="utf-8")
        monkeypatch.setenv("HEARHEAR_VOICE_OUTPUT_STATE_PATH", str(state))
        monkeypatch.setenv("CODEX_TTS_STATE_DIR", str(tmp_path / "state"))
        summarize = Mock()
        speak = Mock()
        monkeypatch.setattr(codex_tts, "summarize_with_luna", summarize)
        monkeypatch.setattr(codex_tts, "speak_text", speak)
        turn = codex_tts.NormalizedTurn(
            source="codex",
            session_id="session-paused",
            turn_id="turn-paused",
            response_text="This should not reach Luna.",
        )

        assert codex_tts.process_turn(turn) == "skipped-global-pause"
        summarize.assert_not_called()
        speak.assert_not_called()

    def test_pause_race_after_summary_skips_playback(self, monkeypatch, tmp_path):
        state = tmp_path / "voice-output-state.json"
        state.write_text('{"paused":false}', encoding="utf-8")
        monkeypatch.setenv("HEARHEAR_VOICE_OUTPUT_STATE_PATH", str(state))
        monkeypatch.setenv("CODEX_TTS_STATE_DIR", str(tmp_path / "state"))

        def summarize_and_pause(*_args, **_kwargs):
            state.write_text('{"paused":true}', encoding="utf-8")
            return SummaryResult("Finished.", "English")

        speak = Mock()
        monkeypatch.setattr(codex_tts, "summarize_with_luna", summarize_and_pause)
        monkeypatch.setattr(codex_tts, "speak_text", speak)
        monkeypatch.setattr(codex_tts, "is_muted", lambda: False)
        turn = codex_tts.NormalizedTurn(
            source="codex",
            session_id="session-race",
            turn_id="turn-race",
            response_text="Finish this work.",
        )

        assert codex_tts.process_turn(turn) == "skipped-global-pause"
        speak.assert_not_called()

    def test_claims_turn_and_speaks_luna_summary(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CODEX_TTS_STATE_DIR", str(tmp_path / "state"))
        monkeypatch.setattr(codex_tts, "is_muted", lambda: False)
        summarize = Mock(return_value=SummaryResult("中文摘要已经生成。", "Chinese"))
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
        speak.assert_called_once_with(
            "中文摘要已经生成。", language_hint="Chinese"
        )
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
        speak.assert_called_once_with(
            "The implementation is complete.", language_hint="English"
        )

    def test_chinese_summary_includes_distinct_cmux_group_and_session(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("CODEX_TTS_STATE_DIR", str(tmp_path / "state"))
        monkeypatch.setenv("CODEX_TTS_INCLUDE_SESSION_NAME", "true")
        monkeypatch.setenv("CMUX_AGENT_LAUNCH_CWD", "/tmp/analytics-ai-enablement")
        monkeypatch.setenv("CMUX_WORKSPACE_ID", "workspace-1")
        monkeypatch.setenv("CMUX_SURFACE_ID", "surface-1")
        monkeypatch.delenv("CMUX_BUNDLED_CLI_PATH", raising=False)
        monkeypatch.setattr(codex_tts.shutil, "which", lambda name: "/usr/bin/cmux")
        tree_payload = {
            "windows": [
                {
                    "workspaces": [
                        {
                            "id": "workspace-1",
                            "ref": "workspace:1",
                            "title": "自动标题 | analytics-ai-enablement",
                            "panes": [
                                {
                                    "surfaces": [
                                        {
                                            "id": "surface-1",
                                            "title": "修复接口 | analytics-ai-enablement",
                                        }
                                    ]
                                }
                            ],
                        }
                    ]
                }
            ]
        }

        def fake_cmux_run(command, **kwargs):
            if command[1:3] == ["workspace-group", "list"]:
                return Mock(
                    returncode=0,
                    stdout=json.dumps(
                        {
                            "groups": [
                                {
                                    "name": "平台组",
                                    "member_workspace_refs": ["workspace:1"],
                                }
                            ]
                        },
                        ensure_ascii=False,
                    ),
                )
            return Mock(
                returncode=0,
                stdout=json.dumps(tree_payload, ensure_ascii=False),
            )

        monkeypatch.setattr(codex_tts.subprocess, "run", fake_cmux_run)
        monkeypatch.setattr(codex_tts, "is_muted", lambda: False)
        monkeypatch.setattr(
            codex_tts,
            "summarize_with_luna",
            Mock(return_value=SummaryResult("中文摘要已经生成。", "Chinese")),
        )
        speak = Mock(return_value="qwen")
        monkeypatch.setattr(codex_tts, "speak_text", speak)

        turn = codex_tts.NormalizedTurn(
            source="codex",
            session_id="session-1",
            turn_id="turn-1",
            response_text="The interface fix is complete.",
        )

        assert codex_tts.process_turn(turn) == "qwen"
        speak.assert_called_once_with(
            "来自平台组分组的修复接口会话：中文摘要已经生成。",
            language_hint="Chinese",
        )


class TestHookEntrypoint:
    @pytest.fixture(autouse=True)
    def use_test_audio_lock(self, monkeypatch, tmp_path):
        monkeypatch.setenv(
            "CODEX_TTS_AUDIO_LOCK_PATH",
            str(tmp_path / "audio-turn-taking.lock"),
        )

    def test_hook_returns_empty_json_without_waiting_for_worker(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CODEX_TTS_STATE_DIR", str(tmp_path / "state"))
        monkeypatch.delenv("CODEX_TTS_ENABLED", raising=False)
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

    def test_hook_does_not_launch_when_tts_is_disabled(self, monkeypatch):
        monkeypatch.setenv("CODEX_TTS_ENABLED", "false")
        launch = Mock()
        monkeypatch.setattr(codex_tts, "launch_worker", launch)
        monkeypatch.setattr(
            sys,
            "stdin",
            io.StringIO(
                json.dumps(
                    {
                        "hook_event_name": "Stop",
                        "session_id": "session-disabled",
                        "turn_id": "turn-disabled",
                        "last_assistant_message": "Done.",
                    }
                )
            ),
        )
        stdout = io.StringIO()
        monkeypatch.setattr(sys, "stdout", stdout)

        assert codex_tts.hook_main() == 0
        assert stdout.getvalue() == "{}\n"
        launch.assert_not_called()

    def test_hook_does_not_launch_worker_for_malformed_global_state(
        self, monkeypatch, tmp_path
    ):
        state = tmp_path / "voice-output-state.json"
        state.write_text("not json", encoding="utf-8")
        monkeypatch.setenv("HEARHEAR_VOICE_OUTPUT_STATE_PATH", str(state))
        launch = Mock()
        monkeypatch.setattr(codex_tts, "launch_worker", launch)
        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({"hook_event_name": "Stop"})))
        stdout = io.StringIO()
        monkeypatch.setattr(sys, "stdout", stdout)

        assert codex_tts.hook_main() == 0
        assert stdout.getvalue() == "{}\n"
        launch.assert_not_called()

    def test_hook_does_not_launch_while_microphone_is_recording(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("CODEX_TTS_ENABLED", "true")
        lock_path = tmp_path / "audio-turn-taking.lock"
        launch = Mock()
        monkeypatch.setattr(codex_tts, "launch_worker", launch)
        monkeypatch.setattr(
            sys,
            "stdin",
            io.StringIO(
                json.dumps(
                    {
                        "hook_event_name": "Stop",
                        "session_id": "session-mic",
                        "turn_id": "turn-mic",
                        "last_assistant_message": "Do not speak this.",
                    }
                )
            ),
        )
        stdout = io.StringIO()
        monkeypatch.setattr(sys, "stdout", stdout)

        with lock_path.open("a+") as microphone:
            fcntl.flock(microphone.fileno(), fcntl.LOCK_EX)
            assert codex_tts.hook_main() == 0

        assert stdout.getvalue() == "{}\n"
        launch.assert_not_called()

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

    def test_worker_skips_commentary_before_final_answer(self, monkeypatch, tmp_path):
        transcript = tmp_path / "rollout.jsonl"
        previous_final = {
            "type": "response_item",
            "turn_id": "turn-1",
            "payload": {
                "type": "message",
                "role": "assistant",
                "phase": "final_answer",
                "content": [
                    {"type": "output_text", "text": "An earlier reply is complete."}
                ],
            },
        }
        commentary = {
            "type": "response_item",
            "turn_id": "turn-1",
            "payload": {
                "type": "message",
                "role": "assistant",
                "phase": "commentary",
                "content": [
                    {"type": "output_text", "text": "Running tests now."}
                ],
            },
        }
        transcript.write_text(
            json.dumps(previous_final) + "\n" + json.dumps(commentary) + "\n",
            encoding="utf-8",
        )
        payload_path = tmp_path / "payload.json"
        payload_path.write_text(
            json.dumps(
                {
                    "session_id": "session-1",
                    "turn_id": "turn-1",
                    "last_assistant_message": "Running tests now.",
                    "transcript_path": str(transcript),
                }
            ),
            encoding="utf-8",
        )
        process = Mock()
        monkeypatch.setattr(codex_tts, "process_turn", process)

        assert codex_tts.worker_main(str(payload_path)) == 0

        process.assert_not_called()

    def test_worker_uses_matching_final_answer(
        self, monkeypatch, tmp_path
    ):
        transcript = tmp_path / "rollout.jsonl"
        records = [
            {
                "type": "response_item",
                "turn_id": "turn-1",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "phase": "commentary",
                    "content": [
                        {"type": "output_text", "text": "Running tests now."}
                    ],
                },
            },
            {
                "type": "response_item",
                "turn_id": "turn-1",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "phase": "final_answer",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "The implementation is complete and all tests pass.",
                        }
                    ],
                },
            },
        ]
        transcript.write_text(
            "\n".join(json.dumps(record) for record in records) + "\n",
            encoding="utf-8",
        )
        payload_path = tmp_path / "payload.json"
        payload_path.write_text(
            json.dumps(
                {
                    "session_id": "session-1",
                    "turn_id": "turn-1",
                    "last_assistant_message": "The implementation is complete and all tests pass.",
                    "transcript_path": str(transcript),
                }
            ),
            encoding="utf-8",
        )
        process = Mock(return_value="qwen")
        monkeypatch.setattr(codex_tts, "process_turn", process)

        assert codex_tts.worker_main(str(payload_path)) == 0

        process.assert_called_once()
        turn = process.call_args.args[0]
        assert turn.response_text == "The implementation is complete and all tests pass."

    def test_worker_accepts_completed_payload_without_transcript(
        self, monkeypatch, tmp_path
    ):
        payload_path = tmp_path / "payload.json"
        payload_path.write_text(
            json.dumps(
                {
                    "session_id": "legacy-session",
                    "turn_id": "legacy-turn",
                    "last_assistant_message": "The implementation is complete.",
                }
            ),
            encoding="utf-8",
        )
        process = Mock(return_value="qwen")
        monkeypatch.setattr(codex_tts, "process_turn", process)

        assert codex_tts.worker_main(str(payload_path)) == 0

        process.assert_called_once()
