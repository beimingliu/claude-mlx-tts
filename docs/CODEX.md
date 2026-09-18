# Codex integration

This checkout includes a project-local Codex `Stop` hook. It sends the
completed response to `gpt-5.6-luna` with low reasoning effort, then speaks the
recap through the existing MLX server or macOS `say`.

## One-time setup

From the repository root:

```bash
uv sync --extra dev
```

For MLX voice cloning on Apple Silicon, install the optional runtime once:

```bash
uv sync --extra mlx --extra dev
```

The hook never installs dependencies while Codex is running. If `.venv` is
present, the launcher uses its Python interpreter. Otherwise it uses `python3`
and uses macOS `say`; the interpreter must be Python 3.10 or newer. An older
system interpreter fails open with `{}` so it cannot block Codex.

Codex must review and trust the project hook before it runs. In the Codex CLI,
use `/hooks` and trust the hook definition for this checkout.

## Provider and model

The default summary model is `gpt-5.6-luna`, with
`reasoning.effort = "low"`. The worker first uses an explicit
`CODEX_TTS_RESPONSES_URL` plus `CODEX_TTS_API_KEY`, then resolves the active
provider and auth command from `~/.codex/config.toml`.

Useful overrides:

```bash
export CODEX_TTS_SUMMARY_MODEL=gpt-5.6-luna
export CODEX_TTS_SUMMARY_REASONING_EFFORT=low
export CODEX_TTS_BACKEND=auto       # auto, mlx, or say
export CODEX_TTS_VOICE=default      # MLX voice name
```

Verify the Luna connection without playing audio:

```bash
CODEX_TTS_BACKEND=say \
  uv run python scripts/codex_tts.py --summarize \
  "The tests pass, but the deployment is blocked by a missing production secret."
```

## Safe controls

```bash
CODEX_TTS_DRY_RUN=true codex          # summarize, but do not play audio
CODEX_TTS_DISABLED=true codex        # leave Codex unchanged
CODEX_TTS_SUMMARY_MODE=off codex     # speak a bounded excerpt instead
CODEX_TTS_BACKEND=say codex          # bypass MLX for a system voice
```

The hook returns `{}` immediately. A detached worker performs the Luna call
and audio playback. Turn IDs are claimed atomically, and a file lock prevents
two Codex sessions from playing audio at the same time.

## Language behavior

The summary prompt asks Luna to preserve the response's natural language and
does not translate identifiers or numbers. The existing Chatterbox Turbo MLX
voice setup is English-focused; multilingual speech should use a matching
macOS voice or a separately validated multilingual MLX model.

## Disable or remove

Set `CODEX_TTS_DISABLED=true` for a temporary disable. To remove this project's
integration, delete `.codex/hooks.json` or disable it through Codex `/hooks`.
