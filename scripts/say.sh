#!/bin/bash
# Speak text directly using the managed TTS worker (fire-and-forget)
# Usage: ./say.sh <text to speak>
#
# This script forks TTS to background and returns immediately.
# This allows callers to continue working while audio plays.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PLUGIN_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PYTHON="$PLUGIN_ROOT/.venv/bin/python"

# Use venv Python if available, otherwise system Python
if [ -x "$VENV_PYTHON" ]; then
    PYTHON="$VENV_PYTHON"
else
    PYTHON="python3"
fi

if [ "$#" -eq 0 ]; then
    echo "Usage: say <text to speak>"
    exit 1
fi

python3 "$SCRIPT_DIR/voice_output.py" --check --stage before-worker || exit 0
"$PYTHON" "$SCRIPT_DIR/managed_speech.py" -- "$@" >/dev/null 2>&1 &

# Disown the background process so it's not tied to this shell
disown 2>/dev/null

echo "TTS started"
