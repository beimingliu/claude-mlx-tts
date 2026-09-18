"""Deterministic unit coverage for the Codex Stop-hook Luna pipeline."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import sys
from unittest.mock import Mock, patch


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import codex_tts  # noqa: E402
from tts_pipeline import LunaClient, ProviderSettings, build_summary_prompt  # noqa: E402
import tts_pipeline  # noqa: E402


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return json.dumps({"output_text": "La prueba pasó."}).encode("utf-8")


def test_outgoing_responses_request_uses_luna_and_low_reasoning():
    request_details = {}

    def fake_urlopen(request, timeout):
        request_details["body"] = json.loads(request.data.decode("utf-8"))
        request_details["timeout"] = timeout
        return _Response()

    with patch.object(tts_pipeline.urllib.request, "urlopen", fake_urlopen):
        result = LunaClient(
            ProviderSettings("https://provider.example/v1/responses", "token")
        ).summarize("La prueba pasó.")

    assert result == "La prueba pasó."
    assert request_details["body"]["model"] == "gpt-5.6-luna"
    assert request_details["body"]["reasoning"] == {"effort": "low"}


def test_recap_prompt_preserves_language_and_identifiers():
    prompt = build_summary_prompt(
        "La prueba pasó en api/v1/users con ticket AIRML-1234.",
        language_hint="Spanish",
    )

    assert "Use Spanish for the spoken recap." in prompt
    assert "api/v1/users" in prompt
    assert "AIRML-1234" in prompt
    assert "Preserve product names, file names, commands, identifiers, and numbers exactly." in prompt


def test_stop_payload_returns_empty_json_and_launches_detached_worker():
    payload = {
        "hook_event_name": "Stop",
        "session_id": "session-42",
        "turn_id": "turn-7",
        "last_assistant_message": "Done.",
    }
    launch_worker = Mock(return_value=True)

    with (
        patch.dict(os.environ, {"CODEX_TTS_DISABLED": "false"}),
        patch.object(codex_tts, "launch_worker", launch_worker),
        patch.object(codex_tts.sys, "stdin", io.StringIO(json.dumps(payload))),
        patch.object(codex_tts.sys, "stdout", new_callable=io.StringIO) as stdout,
    ):
        assert codex_tts.hook_main() == 0

    assert stdout.getvalue() == "{}\n"
    launch_worker.assert_called_once_with(payload)
