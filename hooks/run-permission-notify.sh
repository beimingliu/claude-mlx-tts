#!/bin/bash
# Wrapper script for permission TTS hook - ensures venv is used for mlx_audio

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PLUGIN_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PLUGIN_ROOT/.venv"

# This standard-library preflight runs before importing the TTS environment.
python3 "$PLUGIN_ROOT/scripts/voice_output.py" --check --stage before-worker || exit 0

# Use venv Python if it exists, otherwise fall back to system python
if [ -x "$VENV_DIR/bin/python" ]; then
    exec "$VENV_DIR/bin/python" "$PLUGIN_ROOT/scripts/claude_hook_worker.py" "$SCRIPT_DIR/permission_notify.py"
else
    exec python3 "$PLUGIN_ROOT/scripts/claude_hook_worker.py" "$SCRIPT_DIR/permission_notify.py"
fi
