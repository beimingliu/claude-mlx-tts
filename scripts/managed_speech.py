#!/usr/bin/env python3
"""Background worker for the plugin's managed say commands."""

from __future__ import annotations

import argparse
import importlib.util
import logging
from pathlib import Path

from voice_output import suppression_reason


log = logging.getLogger("managed-speech")


def _tts_notify_module():
    path = Path(__file__).with_name("tts-notify.py")
    spec = importlib.util.spec_from_file_location("tts_notify", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("text", nargs="+")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if suppression_reason("worker-start") is not None:
        return 0

    text = " ".join(args.text).strip()
    if not text:
        return 2
    tts = _tts_notify_module()
    if args.summarize:
        if suppression_reason("before-summary") is not None:
            return 0
        text = f"{tts.ATTENTION_PREFIX_DEFAULT} ... {tts.summarize(text)}"
    tts.speak(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
