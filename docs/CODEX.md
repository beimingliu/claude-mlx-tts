# Codex integration

The user-level global Codex `Stop` hook uses the runtime installed at
`~/.codex/tts`. It sends the completed response to `gpt-5.6-luna` with low
reasoning effort, then sends the recap to the local Qwen TTS playground and
plays its WAV response. macOS `say` remains the final fallback.

## One-time setup

From the repository root:

```bash
uv sync --extra dev
```

The Codex hook does not need this repository's MLX/Chatterbox extra. Start the
existing Qwen playground separately:

```bash
cd /Users/bliu/beiming/beiming/tts-playground
uv run python -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

The hook never installs dependencies while Codex is running. Its Qwen client
uses only the Python standard library; the model dependencies stay inside the
playground environment. If `.venv` is present, the launcher uses its Python
interpreter. Otherwise it uses `python3`; the interpreter must be Python 3.10
or newer. An older system interpreter fails open with `{}` so it cannot block
Codex.

The hook is configured in `~/.codex/hooks.json`, so it applies to every Codex
project. The launcher and runtime files live in `~/.codex/tts`, independent of
the current repository. Codex may ask you to review and trust it; use `/hooks`
in the Codex CLI to inspect the global hook definition.

Copy these runtime files together when installing or updating the hook:
`codex-tts.sh`, `codex_tts.py`, `tts_pipeline.py`, and `voice_output.py`. The
last file implements the shared pause-state and audio-lock contract; omitting
it makes the adapter incomplete.

## Provider and model

The default summary model is `gpt-5.6-luna`, with
`reasoning.effort = "low"`. The worker first uses an explicit
`CODEX_TTS_RESPONSES_URL` plus `CODEX_TTS_API_KEY`, then resolves the active
provider and auth command from `~/.codex/config.toml`.

Useful overrides:

```bash
export CODEX_TTS_SUMMARY_MODEL=gpt-5.6-luna
export CODEX_TTS_SUMMARY_REASONING_EFFORT=low
export CODEX_TTS_BACKEND=auto       # auto, qwen, mlx, or say
export CODEX_TTS_QWEN_CLONE_URL=http://127.0.0.1:8001/api/clone
export CODEX_TTS_QWEN_LANGUAGE=English
export CODEX_TTS_QWEN_TIMEOUT=120
```

Luna returns a short narration plus `metadata.summary_language`. Chinese uses
the `陈晓卿-香料与人类进步` reference; English uses `calm-voice`. You can
override those with `CODEX_TTS_QWEN_CHINESE_REFERENCE_ID` and
`CODEX_TTS_QWEN_ENGLISH_REFERENCE_ID`, or use `CODEX_TTS_QWEN_REFERENCE_ID` for
one reference in every language. A language hint supplied in the Stop payload
is only a prompt/fallback hint; valid summary metadata selects the voice.

`auto` tries Qwen first, then the optional legacy MLX backend when it is
installed, and finally macOS `say`. Set `CODEX_TTS_BACKEND=qwen` to keep the
Codex path focused on Qwen (with `say` still available if Qwen is offline).

Verify the Luna connection without playing audio:

```bash
CODEX_TTS_BACKEND=say \
  uv run python scripts/codex_tts.py --summarize \
  "The tests pass, but the deployment is blocked by a missing production secret."

# Inspect the language metadata without playing audio:
CODEX_TTS_DRY_RUN=true uv run python scripts/codex_tts.py \
  --summarize --language Chinese "本轮实现已经完成，测试也已通过。"
```

## Safe controls

```bash
CODEX_TTS_DRY_RUN=true codex          # summarize, but do not play audio
CODEX_TTS_ENABLED=false codex        # disable summary and speech together
CODEX_TTS_SUMMARY_MODE=off codex     # speak a bounded excerpt instead
CODEX_TTS_BACKEND=qwen codex         # use the local Qwen playground
CODEX_TTS_BACKEND=say codex          # bypass local TTS for a system voice
```

## Microphone coordination

The standalone **Voice Integrations** app's **Pause voice output** control
persists shared state at
`~/Library/Application Support/HearHear/voice-output-state.json`. A missing
file means active; an explicit pause, malformed file, or unreadable file
suppresses work. Resume from Voice Integrations to repair invalid state.
The HearHear directory is retained for compatibility with installed hooks;
HearHear itself has no Integrations page or manual pause control.

The Codex hook shares an advisory lock with HearHear at
`~/Library/Application Support/HearHear/audio-turn-taking.lock`. HearHear
holds it while recording; the hook skips the summary before it calls Luna and
checks again immediately before playback. TTS holds it for the duration of
speech, so a new recording cannot start until audio output is finished.

After changing this integration, rebuild and relaunch HearHear so its running
process uses the lock-aware version.

The hook returns `{}` immediately. A detached worker performs the Luna call
and audio playback. Turn IDs are claimed atomically, and a file lock prevents
two Codex sessions from playing audio at the same time.

## Language behavior

The summary prompt is intentionally short: roughly 8–15 words for a trivial
turn, 15–35 for a routine turn, and at most 60 for a complex turn. It asks Luna
to return exactly one JSON object with the narration and primary language. The
CLI test command prints the same metadata contract that the worker consumes.

When Qwen is unavailable, Chinese skips the English-focused MLX fallback and
uses macOS `say -v Tingting`; English uses `say -v Samantha` unless overridden.

The original Chatterbox Turbo MLX server remains available for the plugin's
legacy Claude Code path, but it is not required by this Codex integration.

## Disable or remove

The summary-and-speech pipeline is enabled by default. Set
`CODEX_TTS_ENABLED=false` for a temporary disable, or set it to `true` to
enable it explicitly. To remove the global integration, remove the TTS entry
from `~/.codex/hooks.json` or disable it through Codex `/hooks`.
