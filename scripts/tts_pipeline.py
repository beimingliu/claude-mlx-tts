#!/usr/bin/env python3
"""Shared summary and speech pipeline used by Codex completion hooks.

The module deliberately has no MLX import at module load time.  A Codex hook
must be cheap to start, and the MLX dependency is optional for installations
that use the macOS ``say`` backend.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Callable
import urllib.error
import urllib.request

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 compatibility
    import tomli as tomllib  # type: ignore[import-not-found]


log = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-5.6-luna"
DEFAULT_REASONING_EFFORT = "low"
DEFAULT_MAX_OUTPUT_TOKENS = 180
DEFAULT_SUMMARY_TIMEOUT = 45
DEFAULT_SAY_VOICE = "Daniel"
DEFAULT_SAY_RATE = 180
DEFAULT_SAY_TIMEOUT = 60


class LunaConfigurationError(RuntimeError):
    """Raised when Codex's active provider cannot be used for Luna."""


class LunaRequestError(RuntimeError):
    """Raised when the Luna Responses request fails or is malformed."""


@dataclass(frozen=True)
class ProviderSettings:
    """Resolved Responses endpoint and bearer token."""

    responses_url: str
    token: str


@dataclass(frozen=True)
class LunaClient:
    """Small stdlib-only client for the configured Responses endpoint."""

    settings: ProviderSettings
    model: str = DEFAULT_MODEL
    reasoning_effort: str = DEFAULT_REASONING_EFFORT
    timeout: float = DEFAULT_SUMMARY_TIMEOUT
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS

    def summarize(self, response_text: str, language_hint: str | None = None) -> str:
        prompt = build_summary_prompt(response_text, language_hint=language_hint)
        body = {
            "model": self.model,
            "input": prompt,
            "reasoning": {"effort": self.reasoning_effort},
            "max_output_tokens": self.max_output_tokens,
        }
        result = _post_responses(
            self.settings.responses_url,
            self.settings.token,
            body,
            timeout=self.timeout,
        )
        summary = extract_response_text(result)
        if not summary:
            raise LunaRequestError("Luna returned no output text")
        return clean_spoken_text(summary)


def _env_first(*names: str) -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def _load_codex_config() -> dict[str, Any]:
    config_root = Path(
        os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))
    ).expanduser()
    config_path = config_root / "config.toml"
    try:
        with config_path.open("rb") as handle:
            value = tomllib.load(handle)
    except (FileNotFoundError, OSError, tomllib.TOMLDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _token_from_provider(provider: dict[str, Any]) -> str:
    auth = provider.get("auth")
    if isinstance(auth, dict):
        command = auth.get("command")
        if isinstance(command, str) and command.strip():
            args = auth.get("args", [])
            command_args = [command]
            if isinstance(args, list):
                command_args.extend(str(arg) for arg in args)
            try:
                result = subprocess.run(
                    command_args,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=float(os.environ.get("CODEX_TTS_AUTH_TIMEOUT", "45")),
                )
            except (OSError, subprocess.SubprocessError) as exc:
                raise LunaConfigurationError("Codex provider auth command failed") from exc
            token = result.stdout.strip()
            if token:
                return token

        env_key = auth.get("env_key")
        if isinstance(env_key, str) and os.environ.get(env_key, "").strip():
            return os.environ[env_key].strip()

    return _env_first("CODEX_TTS_API_KEY", "OPENAI_API_KEY", "LITELLM_API_KEY")


def resolve_provider_settings() -> ProviderSettings:
    """Resolve the active Codex provider without exposing credentials in logs.

    Explicit ``CODEX_TTS_RESPONSES_URL``/``CODEX_TTS_API_KEY`` values win,
    followed by the active provider in ``~/.codex/config.toml``.  This mirrors
    the provider selection Codex itself already uses.
    """

    explicit_url = _env_first("CODEX_TTS_RESPONSES_URL")
    explicit_token = _env_first("CODEX_TTS_API_KEY", "OPENAI_API_KEY", "LITELLM_API_KEY")
    if explicit_url and explicit_token:
        return ProviderSettings(explicit_url.rstrip("/"), explicit_token)

    config = _load_codex_config()
    provider_name = config.get("model_provider")
    providers = config.get("model_providers")
    provider = providers.get(provider_name, {}) if isinstance(providers, dict) else {}
    if not isinstance(provider, dict):
        provider = {}

    base_url = _env_first("CODEX_TTS_BASE_URL")
    if not base_url:
        configured_base = provider.get("base_url")
        if isinstance(configured_base, str):
            base_url = configured_base.strip()
    if not base_url:
        base_url = _env_first("OPENAI_BASE_URL", "LITELLM_ENDPOINT")
    if not base_url and os.environ.get("OPENAI_API_KEY", "").strip():
        base_url = "https://api.openai.com/v1"

    token = explicit_token or _token_from_provider(provider)
    if not base_url or not token:
        raise LunaConfigurationError(
            "No usable Codex Responses provider; set CODEX_TTS_RESPONSES_URL and "
            "CODEX_TTS_API_KEY or configure the active Codex provider"
        )

    responses_url = base_url.rstrip("/")
    if not responses_url.endswith("/responses"):
        responses_url += "/responses"
    return ProviderSettings(responses_url, token)


def build_summary_prompt(response_text: str, language_hint: str | None = None) -> str:
    """Build a spoken recap prompt that preserves the response language."""

    language_instruction = (
        f"Use {language_hint} for the spoken recap."
        if language_hint
        else "Use the same natural language as the response; do not translate it."
    )
    return (
        "You are the voice-recap layer for a coding agent. Convert the completed "
        "agent response below into one or two concise spoken sentences. "
        "State the result first, then mention an important limitation, blocker, "
        "or question only when one exists. Preserve product names, file names, "
        "commands, identifiers, and numbers exactly. Do not use markdown, code "
        "fences, headings, quotes, or an introduction. "
        f"{language_instruction}\n\n"
        "<agent_response>\n"
        f"{clip_for_summary(response_text)}\n"
        "</agent_response>"
    )


def clip_for_summary(text: str, limit: int = 12_000) -> str:
    """Keep both the conclusion and the beginning when a response is large."""

    normalized = str(text or "").strip()
    if len(normalized) <= limit:
        return normalized
    side = (limit - 120) // 2
    return f"{normalized[:side]}\n...[middle omitted]...\n{normalized[-side:]}"


def _post_responses(
    url: str,
    token: str,
    body: dict[str, Any],
    *,
    timeout: float,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = json.load(response)
    except urllib.error.HTTPError as exc:
        # Do not copy provider response bodies into logs; they may contain
        # prompt content or credential-related diagnostics.
        raise LunaRequestError(f"Luna HTTP error {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise LunaRequestError("Luna request could not be completed") from exc
    if not isinstance(value, dict):
        raise LunaRequestError("Luna returned a non-object response")
    return value


def extract_response_text(value: dict[str, Any]) -> str:
    """Extract only model output text from Responses or Chat-style payloads."""

    direct = value.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    pieces: list[str] = []
    output = value.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "output_text" and isinstance(block.get("text"), str):
                    pieces.append(block["text"])

    choices = value.get("choices")
    if not pieces and isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                pieces.append(message["content"])

    return "\n".join(piece.strip() for piece in pieces if piece.strip()).strip()


def clean_spoken_text(text: str, max_chars: int = 900) -> str:
    """Remove formatting that is unhelpful or unsafe for speech playback."""

    value = str(text or "").strip()
    value = re.sub(r"^```[\w-]*\s*|\s*```$", "", value, flags=re.DOTALL).strip()
    value = re.sub(r"^(?:summary|recap|spoken recap)\s*:\s*", "", value, flags=re.IGNORECASE)
    value = value.strip("\"'").strip()
    if len(value) > max_chars:
        value = value[: max_chars - 1].rstrip() + "…"
    return value


def fallback_excerpt(text: str, max_chars: int = 260) -> str:
    """Return a bounded, readable fallback when Luna is unavailable."""

    value = clean_spoken_text(text, max_chars=max_chars)
    sentence = re.split(r"(?<=[.!?。！？])\s+", value, maxsplit=1)[0]
    return sentence or value


def get_luna_client() -> LunaClient:
    settings = resolve_provider_settings()
    return LunaClient(
        settings=settings,
        model=os.environ.get("CODEX_TTS_SUMMARY_MODEL", DEFAULT_MODEL),
        reasoning_effort=os.environ.get(
            "CODEX_TTS_SUMMARY_REASONING_EFFORT", DEFAULT_REASONING_EFFORT
        ),
        timeout=float(os.environ.get("CODEX_TTS_SUMMARY_TIMEOUT", str(DEFAULT_SUMMARY_TIMEOUT))),
        max_output_tokens=int(
            os.environ.get("CODEX_TTS_SUMMARY_MAX_OUTPUT_TOKENS", str(DEFAULT_MAX_OUTPUT_TOKENS))
        ),
    )


def summarize_with_luna(
    response_text: str,
    *,
    language_hint: str | None = None,
    client_factory: Callable[[], LunaClient] = get_luna_client,
) -> str:
    """Summarize with the configured low-effort Luna client."""

    if not response_text.strip():
        return ""
    return client_factory().summarize(response_text, language_hint=language_hint)


def _say_voice() -> str:
    return os.environ.get("CODEX_TTS_SAY_VOICE", DEFAULT_SAY_VOICE)


def speak_with_say(text: str) -> None:
    """Speak synchronously so the worker can serialize playback."""

    voice = _say_voice()
    rate = os.environ.get("CODEX_TTS_SAY_RATE", str(DEFAULT_SAY_RATE))
    timeout = float(os.environ.get("CODEX_TTS_SAY_TIMEOUT", str(DEFAULT_SAY_TIMEOUT)))
    try:
        result = subprocess.run(
            ["say", "-v", voice, "-r", rate, clean_spoken_text(text)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("macOS say timed out") from exc
    if result.returncode != 0:
        raise RuntimeError(f"macOS say exited with status {result.returncode}")


def _mlx_is_available() -> bool:
    try:
        import mlx_audio  # noqa: F401
        from tts_config import discover_voices

        return bool(discover_voices())
    except (ImportError, OSError, RuntimeError):
        return False


def speak_with_mlx(text: str) -> None:
    """Use the existing warm MLX server and let failures reach the fallback."""

    from mlx_server_utils import speak_mlx_http

    voice = os.environ.get("CODEX_TTS_VOICE") or None
    if voice is None:
        try:
            from tts_config import get_effective_hook_voice

            voice = get_effective_hook_voice("stop")
        except (ImportError, OSError, RuntimeError):
            voice = None
    speak_mlx_http(clean_spoken_text(text), voice=voice)


def speak_text(text: str) -> str:
    """Speak through MLX when available, otherwise macOS ``say``."""

    cleaned = clean_spoken_text(text)
    if not cleaned:
        return "skipped"
    if os.environ.get("CODEX_TTS_DRY_RUN", "").lower() == "true":
        log.info("dry-run: would speak %d characters", len(cleaned))
        return "dry_run"

    backend = os.environ.get("CODEX_TTS_BACKEND", "auto").lower()
    if backend in {"auto", "mlx"} and _mlx_is_available():
        try:
            speak_with_mlx(cleaned)
            return "mlx"
        except Exception as exc:  # MLX is optional; system speech remains useful.
            log.warning("MLX speech failed; falling back to macOS say: %s", exc)
            if backend == "mlx":
                # Explicit MLX still falls back so a transient server failure
                # does not make completion notifications disappear.
                pass

    speak_with_say(cleaned)
    return "say"


def is_muted() -> bool:
    """Use the existing mute state when the plugin configuration is present."""

    try:
        from tts_mute import is_muted as existing_is_muted

        return bool(existing_is_muted())
    except (ImportError, OSError, RuntimeError):
        return False
