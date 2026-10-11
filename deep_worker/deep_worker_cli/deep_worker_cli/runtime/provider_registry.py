"""Config-driven API providers — add a provider without touching code.

Owner directive (2026-08-15):

    "I want standard API providers like these to be defined through config,
     not through changes to the deep-worker code."

A *standard API provider* talks plain HTTP with an OpenAI-compatible
dialect (the deepseek / openai / gemini shape). That is the common case:
new vendors keep shipping OpenAI-compatible endpoints, and each one used to
cost us edits in ``SUPPORTED_PROVIDERS``, ``_detect_provider``, the
credential checks, the ``create_model`` if/elif chain, ``Settings`` and the
token-stats provider sets. Now it costs one JSON entry.

CLI/ACP providers (``claude-code-sdk``, ``claude-code``, ``codex``,
``grok``) are NOT in scope: they need a whole adapter layer, which exists
out of necessity, not by choice, and cannot be expressed as config. They stay
built-in and are untouched by this module.

Config location: ``~/.deep_worker/providers.json`` — the product runtime
tree. (``~/.deepagents`` belongs to the upstream ancestor and is never read
for backwards compatibility elsewhere; nothing new goes there.)

    {
      "providers": [
        {
          "name": "minimax",
          "compatible_with": "openai",
          "base_url": "https://api.minimax.io/v1",
          "api_key_env": "MINIMAX_API_KEY",
          "default_model": "MiniMax-M3",
          "default_model_env": "MINIMAX_MODEL",
          "model_markers": ["minimax"],
          "account_group": "minimax-account",
          "extra": {"stream_usage": true}
        }
      ]
    }

SECURITY (project API key security policy): this file names the ENV VAR
that holds the key — it never holds the key. The loader rejects an entry
carrying an inline ``api_key`` (or an ``api_key_env`` that looks like a
secret rather than a variable name), and its error message never echoes
the offending value.

MODEL NAMES (feedback_no_model_hardcode_2026_08_15): no model name is
hardcoded in Python. ``default_model`` lives in the operator's config and
applies ONLY when nothing else was specified; an explicit model always
travels to the backend verbatim, and the backend owns rejection.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from deep_worker_cli.storage_paths import deep_worker_home

REGISTRY_FILE_NAME = "providers.json"

# Dialects the shared client layer can actually speak. Unknown values are
# rejected rather than defaulted: silently running an OpenAI client against
# a non-OpenAI API produces confusing runtime errors instead of a clear
# config error at startup.
SUPPORTED_DIALECTS: frozenset[str] = frozenset({"openai"})

# Names owned by the built-in providers. A config entry may not reuse one:
# agents persist ``provider`` by name, so shadowing "openai" would silently
# change the meaning of an existing agent's saved configuration.
_BUILTIN_NAMES: frozenset[str] = frozenset(
    {
        "deepseek",
        "openai",
        "anthropic",
        "google",
        "gemini",
        "vertexai",
        "claude-code-sdk",
        "claude-code",
        "codex",
        "grok",
    }
)

_REQUIRED_FIELDS = ("name", "compatible_with", "base_url", "api_key_env")

_PROVIDER_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
# A POSIX-ish environment variable name. Deliberately strict so that a
# pasted secret (which is long, mixed-case and usually contains '-' or '.')
# cannot pass as a variable name.
_ENV_VAR_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


class ProviderConfigError(RuntimeError):
    """``providers.json`` is malformed, unsafe, or self-contradictory.

    Raised at load time with an actionable message. Never carries a secret
    value — see :func:`_reject_inline_secret`.
    """


@dataclass(frozen=True)
class ProviderSpec:
    """One config-declared API provider."""

    name: str
    compatible_with: str
    base_url: str
    api_key_env: str
    default_model: str | None = None
    default_model_env: str | None = None
    model_markers: tuple[str, ...] = ()
    account_group: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def api_key(self) -> str | None:
        """The key value from the environment (loaded from env.d), or None."""
        value = os.environ.get(self.api_key_env)
        return value or None

    def has_credentials(self) -> bool:
        return self.api_key() is not None

    def resolve_model(self, explicit: str | None = None) -> str | None:
        """Model precedence, with NO model name hardcoded in Python.

        1. ``explicit`` (the operator's ``--model``) — passed through
           VERBATIM. We never validate it against a local list; the backend
           owns rejection and reports a real error
           (feedback_no_model_hardcode_2026_08_15 rules 2 and 4).
        2. ``$<default_model_env>`` — operator's environment.
        3. ``default_model`` from the operator's config.
        4. ``None`` — the caller decides how to surface "no model chosen".
        """
        if explicit and explicit.strip():
            return explicit.strip()
        if self.default_model_env:
            from_env = os.environ.get(self.default_model_env)
            if from_env and from_env.strip():
                return from_env.strip()
        return self.default_model

    def matches_model(self, model_name: str) -> bool:
        """Whether ``model_name`` looks like it belongs to this provider.

        Used for provider auto-detection from a bare ``--model``. Markers
        are operator-supplied substrings; with none configured the provider
        never claims a model (explicit ``-pp`` is then required), which is
        the safe default — a greedy matcher would steal models from other
        providers.
        """
        if not self.model_markers:
            return False
        lowered = model_name.strip().lower()
        return any(m in lowered for m in self.model_markers)


def registry_path(home: Path | None = None) -> Path:
    """Absolute path of ``providers.json`` in the product runtime tree."""
    return deep_worker_home(home) / REGISTRY_FILE_NAME


def _reject_inline_secret(entry: dict, index: int) -> None:
    """Refuse a config that carries a key value instead of a variable name.

    The message deliberately does NOT include the offending value: an error
    string can reach logs, the web UI and bug reports.
    """
    for forbidden in ("api_key", "apikey", "key", "secret", "token"):
        if forbidden in entry:
            raise ProviderConfigError(
                f"{REGISTRY_FILE_NAME}: provider #{index} declares "
                f"{forbidden!r}. API keys must NEVER be stored in this file. "
                f"Put the key in ~/.deep_worker/env.d/<name>.env (mode 0600) "
                f"and reference it here by variable name via 'api_key_env'. "
                f"See the project API key security policy."
            )


def _parse_entry(entry: Any, index: int) -> ProviderSpec:
    if not isinstance(entry, dict):
        raise ProviderConfigError(
            f"{REGISTRY_FILE_NAME}: provider #{index} must be an object, "
            f"got {type(entry).__name__}"
        )

    _reject_inline_secret(entry, index)

    for required in _REQUIRED_FIELDS:
        value = entry.get(required)
        if not isinstance(value, str) or not value.strip():
            raise ProviderConfigError(
                f"{REGISTRY_FILE_NAME}: provider #{index} is missing required "
                f"field {required!r} (must be a non-empty string). Required "
                f"fields: {', '.join(_REQUIRED_FIELDS)}."
            )

    name = entry["name"].strip().lower()
    if not _PROVIDER_NAME_RE.match(name):
        raise ProviderConfigError(
            f"{REGISTRY_FILE_NAME}: provider name {name!r} is invalid — use "
            f"lowercase letters, digits, '-' and '_' (must start with a "
            f"letter or digit)."
        )
    if name in _BUILTIN_NAMES:
        raise ProviderConfigError(
            f"{REGISTRY_FILE_NAME}: provider name {name!r} is a built-in "
            f"provider and cannot be redefined in config — agents persist "
            f"their provider by name, so redefining it would silently change "
            f"the meaning of existing agents. Choose a different name."
        )

    dialect = entry["compatible_with"].strip().lower()
    if dialect not in SUPPORTED_DIALECTS:
        raise ProviderConfigError(
            f"{REGISTRY_FILE_NAME}: provider {name!r} declares "
            f"compatible_with={dialect!r}, which is not supported. "
            f"Supported dialects: {', '.join(sorted(SUPPORTED_DIALECTS))}."
        )

    base_url = entry["base_url"].strip()
    if not base_url.startswith(("http://", "https://")):
        raise ProviderConfigError(
            f"{REGISTRY_FILE_NAME}: provider {name!r} has base_url "
            f"{base_url!r} — must start with http:// or https://."
        )

    api_key_env = entry["api_key_env"].strip()
    if not _ENV_VAR_RE.match(api_key_env):
        raise ProviderConfigError(
            f"{REGISTRY_FILE_NAME}: provider {name!r} has an api_key_env that "
            f"is not a valid environment variable name (expected e.g. "
            f"MINIMAX_API_KEY: upper-case letters, digits and underscores). "
            f"This field takes the NAME of the variable holding the key, "
            f"never the key itself — see the project API key security policy."
        )

    default_model = entry.get("default_model")
    if default_model is not None and (
        not isinstance(default_model, str) or not default_model.strip()
    ):
        raise ProviderConfigError(
            f"{REGISTRY_FILE_NAME}: provider {name!r} has a non-string "
            f"default_model."
        )

    default_model_env = entry.get("default_model_env")
    if default_model_env is not None:
        if not isinstance(default_model_env, str) or not _ENV_VAR_RE.match(
            default_model_env.strip()
        ):
            raise ProviderConfigError(
                f"{REGISTRY_FILE_NAME}: provider {name!r} has an invalid "
                f"default_model_env (expected a variable name like "
                f"MINIMAX_MODEL)."
            )
        default_model_env = default_model_env.strip()

    raw_markers = entry.get("model_markers", [])
    if not isinstance(raw_markers, list) or not all(
        isinstance(m, str) for m in raw_markers
    ):
        raise ProviderConfigError(
            f"{REGISTRY_FILE_NAME}: provider {name!r} has invalid "
            f"model_markers (expected a list of strings)."
        )
    markers = tuple(m.strip().lower() for m in raw_markers if m.strip())

    account_group = entry.get("account_group")
    if account_group is not None and not isinstance(account_group, str):
        raise ProviderConfigError(
            f"{REGISTRY_FILE_NAME}: provider {name!r} has a non-string "
            f"account_group."
        )

    extra = entry.get("extra", {})
    if not isinstance(extra, dict):
        raise ProviderConfigError(
            f"{REGISTRY_FILE_NAME}: provider {name!r} has a non-object 'extra'."
        )

    return ProviderSpec(
        name=name,
        compatible_with=dialect,
        base_url=base_url,
        api_key_env=api_key_env,
        default_model=default_model.strip() if default_model else None,
        default_model_env=default_model_env,
        model_markers=markers,
        account_group=account_group.strip() if account_group else None,
        extra=dict(extra),
    )


def load_provider_specs(home: Path | None = None) -> tuple[ProviderSpec, ...]:
    """Read and validate ``~/.deep_worker/providers.json``.

    A missing file is NOT an error — it just means no config-declared
    providers. Anything present but malformed IS an error: silently
    ignoring a broken entry would leave the operator staring at "unknown
    provider" with no explanation.
    """
    path = registry_path(home)
    if not path.is_file():
        return ()

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ProviderConfigError(
            f"{REGISTRY_FILE_NAME}: cannot read {path} — {exc}"
        ) from exc

    if not isinstance(raw, dict):
        raise ProviderConfigError(
            f"{REGISTRY_FILE_NAME}: top level must be an object with a "
            f"'providers' array."
        )

    entries = raw.get("providers", [])
    if not isinstance(entries, list):
        raise ProviderConfigError(
            f"{REGISTRY_FILE_NAME}: 'providers' must be an array."
        )

    specs: list[ProviderSpec] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        spec = _parse_entry(entry, index)
        if spec.name in seen:
            raise ProviderConfigError(
                f"{REGISTRY_FILE_NAME}: duplicate provider name {spec.name!r}."
            )
        seen.add(spec.name)
        specs.append(spec)

    return tuple(specs)
