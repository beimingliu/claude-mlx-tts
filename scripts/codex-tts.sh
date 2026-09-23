#!/bin/bash
# Codex Stop hook launcher.
#
# Keep dependency installation out of the hook path. The default Codex TTS
# backend calls the separate Qwen playground over localhost HTTP.

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PLUGIN_ROOT="$(dirname "$SCRIPT_DIR")"

if [ -n "${CODEX_TTS_PYTHON:-}" ]; then
    PYTHON="$CODEX_TTS_PYTHON"
elif [ -x "$PLUGIN_ROOT/.venv/bin/python" ]; then
    PYTHON="$PLUGIN_ROOT/.venv/bin/python"
else
    PYTHON="python3"
fi

if ! "$PYTHON" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
    printf '%s\n' '{}'
    exit 0
fi

exec "$PYTHON" "$SCRIPT_DIR/codex_tts.py" --hook
