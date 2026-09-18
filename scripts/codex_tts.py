#!/usr/bin/env python3
"""Codex Stop-hook adapter for the Luna summary and TTS pipeline.

The hook process is intentionally tiny: it reads Codex's JSON payload, writes
that payload to a private temporary file, starts a detached worker, prints an
empty JSON result, and exits.  The worker owns the potentially slow Luna and
audio operations.

Usage:
    python scripts/codex_tts.py --hook
    python scripts/codex_tts.py --summarize "text to recap"
    python scripts/codex_tts.py --worker /path/to/payload.json
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import argparse
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Iterator


SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from tts_pipeline import (  # noqa: E402
    fallback_excerpt,
    is_muted,
    speak_text,
    summarize_with_luna,
)


log = logging.getLogger("codex-mlx-tts")


@dataclass(frozen=True)
class NormalizedTurn:
    """Turn data shared by the Codex adapter and worker."""

    source: str
    session_id: str
    turn_id: str
    response_text: str
    transcript_path: str = ""
    language_hint: str | None = None

    @property
    def event_key(self) -> str:
        return f"{self.source}:{self.session_id}:{self.turn_id}"


def _string(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _first_string(payload: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = _string(payload.get(key))
        if value:
            return value
    return ""


def _derived_turn_id(session_id: str, transcript_path: str, response_text: str) -> str:
    identity = f"{session_id}\0{transcript_path}\0{response_text}"
    return "derived-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type not in {"output_text", "text"}:
            continue
        text = block.get("text")
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
    return "\n".join(parts).strip()


def normalize_payload(payload: dict[str, Any]) -> NormalizedTurn:
    """Normalize the current Codex Stop payload without reading the transcript."""

    session_id = _first_string(payload, "session_id", "sessionId") or "unknown-session"
    turn_id = _first_string(payload, "turn_id", "turnId")
    response_text = _first_string(
        payload,
        "last_assistant_message",
        "lastAssistantMessage",
        "last_agent_message",
        "lastAgentMessage",
    )
    transcript_path = _first_string(payload, "transcript_path", "transcriptPath")
    language_hint = _first_string(payload, "language_hint", "language") or None

    # Some Codex builds do not include turn_id in every hook payload.  A
    # stable content identity still prevents duplicate playback for that one
    # session, while keeping two different sessions independent.
    if not turn_id:
        turn_id = _derived_turn_id(session_id, transcript_path, response_text)

    return NormalizedTurn(
        source="codex",
        session_id=session_id,
        turn_id=turn_id,
        response_text=response_text,
        transcript_path=transcript_path,
        language_hint=language_hint,
    )


def _record_turn_id(record: dict[str, Any]) -> str:
    value = _first_string(record, "turn_id", "turnId")
    payload = record.get("payload")
    if not value and isinstance(payload, dict):
        value = _first_string(payload, "turn_id", "turnId")
    return value


def _candidate_from_record(record: dict[str, Any]) -> tuple[str, bool] | None:
    """Return (text, is_final_answer) for a Codex JSONL record."""

    record_type = record.get("type")
    if record_type == "response_item":
        payload = record.get("payload")
        if not isinstance(payload, dict) or payload.get("type") != "message":
            return None
        if payload.get("role") != "assistant":
            return None
        phase = _string(payload.get("phase")).lower()
        if phase in {"analysis", "commentary"}:
            return None
        text = _content_text(payload.get("content"))
        if not text:
            return None
        return text, phase in {"final_answer", "final"}

    # Completed-turn records in local Codex sessions may carry the final
    # assistant message directly instead of repeating response_item content.
    if record_type in {"event_msg", "task_complete", "turn_completed"}:
        payload = record.get("payload")
        if isinstance(payload, dict):
            text = _first_string(payload, "last_agent_message", "last_assistant_message")
            if text:
                return text, True
        text = _first_string(record, "last_agent_message", "last_assistant_message")
        if text:
            return text, True

    return None


def extract_transcript_response(transcript_path: str, turn_id: str = "") -> str:
    """Find the final assistant response for one Codex turn in JSONL."""

    path = Path(transcript_path).expanduser()
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            # A completion record is at the end; avoid loading an entire
            # multi-day session into the hook worker.
            if size > 2 * 1024 * 1024:
                handle.seek(size - 2 * 1024 * 1024)
                handle.readline()
            lines = handle.read().decode("utf-8", errors="replace").splitlines()
    except (FileNotFoundError, OSError):
        return ""

    records: list[tuple[dict[str, Any], tuple[str, bool]]] = []
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        candidate = _candidate_from_record(record)
        if candidate:
            records.append((record, candidate))

    identified_records = [
        (record, candidate) for record, candidate in records if _record_turn_id(record)
    ]
    if turn_id and identified_records:
        matching = [
            candidate
            for record, candidate in identified_records
            if _record_turn_id(record) == turn_id
        ]
    else:
        matching = [candidate for _, candidate in records]
    if not matching:
        return ""

    final_answers = [text for text, is_final in matching if is_final]
    return (final_answers[-1] if final_answers else matching[-1][0]).strip()


def _state_dir() -> Path:
    configured = os.environ.get("CODEX_TTS_STATE_DIR", "").strip()
    return Path(configured).expanduser() if configured else Path.home() / ".local/state/claude-mlx-tts/codex"


def _configure_logging() -> None:
    if log.handlers:
        return
    log_path = Path(
        os.environ.get("CODEX_TTS_LOG_PATH", str(_state_dir() / "codex-tts.log"))
    ).expanduser()
    try:
        log_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        handler: logging.Handler = logging.FileHandler(log_path, encoding="utf-8")
    except OSError:
        handler = logging.NullHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(handler)
    log.setLevel(os.environ.get("TTS_LOG_LEVEL", "INFO").upper())


def claim_turn(turn: NormalizedTurn) -> bool:
    """Atomically claim a turn so duplicate Stop events speak only once."""

    claims_dir = _state_dir() / "claims"
    claims_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    digest = hashlib.sha256(turn.event_key.encode("utf-8")).hexdigest()
    claim_path = claims_dir / f"{digest}.json"
    try:
        descriptor = os.open(
            claim_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except FileExistsError:
        return False
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump({"event_key": turn.event_key}, handle)
    except OSError:
        try:
            claim_path.unlink()
        except OSError:
            pass
        raise
    return True


@contextmanager
def playback_lock() -> Iterator[None]:
    """Serialize audio playback across concurrent Codex sessions."""

    lock_path = _state_dir() / "playback.lock"
    lock_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _summary_mode() -> str:
    mode = os.environ.get("CODEX_TTS_SUMMARY_MODE", "always").lower().strip()
    return mode if mode in {"off", "auto", "always"} else "always"


def _should_summarize(text: str) -> bool:
    if len(text) > 420 or "\n" in text or "```" in text:
        return True
    return len(re.findall(r"\b\w+\b", text)) > 65


def _attention_prefix() -> str:
    if os.environ.get("CODEX_TTS_INCLUDE_ATTENTION", "false").lower() != "true":
        return ""
    configured = os.environ.get("CODEX_TTS_ATTENTION_PREFIX", "").strip()
    if configured:
        return configured
    try:
        from tts_config import get_effective_hook_prompt

        return str(get_effective_hook_prompt("stop") or "").strip()
    except (ImportError, OSError, RuntimeError):
        return ""


def process_turn(turn: NormalizedTurn) -> str:
    """Summarize and speak a normalized completed turn."""

    if not turn.response_text.strip():
        return "skipped-empty"
    if os.environ.get("CODEX_TTS_DISABLED", "").lower() == "true":
        return "skipped-disabled"
    if not claim_turn(turn):
        log.info("duplicate turn skipped: %s", turn.event_key)
        return "skipped-duplicate"
    if is_muted():
        log.info("muted turn skipped: %s", turn.event_key)
        return "skipped-muted"

    mode = _summary_mode()
    try:
        if mode == "off" or (mode == "auto" and not _should_summarize(turn.response_text)):
            spoken = fallback_excerpt(turn.response_text, max_chars=900)
        else:
            spoken = summarize_with_luna(
                turn.response_text,
                language_hint=turn.language_hint,
            )
    except Exception as exc:
        log.warning("Luna summary failed; using response excerpt: %s", exc)
        spoken = fallback_excerpt(turn.response_text)

    prefix = _attention_prefix()
    if prefix:
        spoken = f"{prefix} ... {spoken}"
    with playback_lock():
        result = speak_text(spoken)
    log.info("turn spoken: %s backend=%s", turn.event_key, result)
    return result


def _write_payload(payload: dict[str, Any]) -> Path:
    directory = _state_dir() / "pending"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, path = tempfile.mkstemp(prefix="codex-turn-", suffix=".json", dir=directory)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        os.chmod(path, 0o600)
    except BaseException:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise
    return Path(path)


def launch_worker(payload: dict[str, Any]) -> bool:
    """Start a detached worker and return without waiting for Luna or TTS."""

    payload_path = _write_payload(payload)
    log_path = Path(
        os.environ.get("CODEX_TTS_LOG_PATH", str(_state_dir() / "codex-tts.log"))
    ).expanduser()
    log_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    log_handle = log_path.open("a", encoding="utf-8")
    python = os.environ.get("CODEX_TTS_PYTHON", sys.executable)
    command = [python, str(Path(__file__).resolve()), "--worker", str(payload_path)]
    try:
        subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=log_handle,
            close_fds=True,
            start_new_session=True,
        )
    except OSError:
        log_handle.close()
        try:
            payload_path.unlink()
        except OSError:
            pass
        raise
    finally:
        log_handle.close()
    return True


def hook_main() -> int:
    """Handle a Codex hook invocation and always fail open for Codex."""

    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError, OSError):
        payload = {}
    if not isinstance(payload, dict):
        payload = {}

    if payload.get("stop_hook_active") is True:
        print("{}")
        return 0
    if os.environ.get("CODEX_TTS_DISABLED", "").lower() != "true":
        try:
            launch_worker(payload)
        except Exception as exc:
            # Hook failures must not block or alter the completed Codex turn.
            log.warning("could not launch worker: %s", exc)
    print("{}")
    return 0


def worker_main(payload_path: str) -> int:
    path = Path(payload_path).expanduser()
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        log.warning("worker payload could not be read")
        return 0
    finally:
        try:
            path.unlink()
        except OSError:
            pass
    if not isinstance(payload, dict):
        return 0

    turn = normalize_payload(payload)
    if not turn.response_text and turn.transcript_path:
        response = extract_transcript_response(turn.transcript_path, turn.turn_id)
        turn_id = turn.turn_id
        if response and turn_id.startswith("derived-"):
            turn_id = _derived_turn_id(turn.session_id, turn.transcript_path, response)
        turn = NormalizedTurn(
            source=turn.source,
            session_id=turn.session_id,
            turn_id=turn_id,
            response_text=response,
            transcript_path=turn.transcript_path,
            language_hint=turn.language_hint,
        )
    try:
        process_turn(turn)
    except Exception as exc:
        # This is a detached notification worker; report diagnostics and leave
        # Codex's already-completed turn untouched.
        log.exception("Codex TTS worker failed: %s", exc)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--hook", action="store_true", help="read a Codex hook payload from stdin")
    mode.add_argument("--worker", metavar="PAYLOAD", help="process a detached payload file")
    mode.add_argument("--summarize", metavar="TEXT", help="call Luna and print a spoken recap")
    return parser


def main(argv: list[str] | None = None) -> int:
    _configure_logging()
    args = build_parser().parse_args(argv)
    if args.worker:
        return worker_main(args.worker)
    if args.summarize is not None:
        try:
            print(summarize_with_luna(args.summarize))
            return 0
        except Exception as exc:
            print(f"Luna summary failed: {exc}", file=sys.stderr)
            return 1
    return hook_main()


if __name__ == "__main__":
    raise SystemExit(main())
