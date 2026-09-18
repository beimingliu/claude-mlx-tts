"""Opt-in live Luna check for the Codex summary provider.

Run with:
    CODEX_TTS_RUN_LIVE_TEST=1 uv run pytest \
        tests/integration/test_luna_summary_itest.py -v
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from tts_pipeline import summarize_with_luna  # noqa: E402


def test_live_luna_summary_uses_low_effort_configuration():
    if os.environ.get("CODEX_TTS_RUN_LIVE_TEST") != "1":
        pytest.skip("set CODEX_TTS_RUN_LIVE_TEST=1 to call the configured Luna provider")

    summary = summarize_with_luna(
        "The Codex Stop hook now launches a detached worker. The worker asks "
        "gpt-5.6-luna for a short spoken recap and then uses the existing TTS "
        "backend. The audio device is not exercised by this test.",
    )

    assert summary
    assert len(summary) < 900
