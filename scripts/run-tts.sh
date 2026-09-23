#!/bin/bash
# Wrapper script for TTS hook - ensures venv is ready and runs the Python script

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PLUGIN_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PLUGIN_ROOT/.venv"

# Honor HearHear's persistent pause before any environment setup or worker work.
# Suppression is a successful hook outcome; diagnostics are written to stderr.
python3 "$SCRIPT_DIR/voice_output.py" --check --stage before-worker || exit 0

# Create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    uv venv "$VENV_DIR" 2>/dev/null
    uv pip install --python "$VENV_DIR/bin/python" mlx-audio librosa mlx-lm einops sounddevice pedalboard 2>/dev/null
fi

# Detach before summary/playback. The worker keeps the payload private and owns
# the shared lock until playback finishes.
exec "$VENV_DIR/bin/python" "$SCRIPT_DIR/claude_hook_worker.py" "$SCRIPT_DIR/tts-notify.py"
