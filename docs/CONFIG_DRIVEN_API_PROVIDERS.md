# Adding an API provider through config (no code changes)

Owner directive (2026-08-15):

> I want standard API providers like these to be defined through config,
> not through changes to the deep-worker code.

New vendors keep shipping OpenAI-compatible HTTP APIs. Adding one used to
require edits in seven places (`SUPPORTED_PROVIDERS`, `_detect_provider`,
two credential-check chains, the `create_model` if/elif, `Settings`, and the
token-stats provider sets). Now it is one JSON entry plus a key file.

**In scope:** "standard API providers" — plain HTTP, OpenAI-compatible
dialect (the deepseek / openai / gemini shape).

**Out of scope:** the CLI/ACP providers (`claude-code-sdk`, `claude-code`,
`codex`, `grok`). They need a whole adapter layer, which exists "out of
necessity" and cannot be expressed as config. They remain built-in and
were not modified.

## 1. Store the key (never in the config file)

Per `API_KEY_SECURITY_POLICY.md`, secrets live in env drop-ins, one file per
concern, mode `0600`:

```bash
mkdir -p ~/.deep_worker/env.d && chmod 0700 ~/.deep_worker/env.d
printf 'MINIMAX_API_KEY=%s\n' "$KEY" > ~/.deep_worker/env.d/minimax.env
chmod 0600 ~/.deep_worker/env.d/minimax.env
```

The loader **skips** any drop-in looser than `0600` and logs a warning, so a
permissions slip fails closed rather than leaking.

## 2. Declare the provider

`~/.deep_worker/providers.json`:

```json
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
      "account_group": "minimax-account"
    }
  ]
}
```

| Field | Required | Meaning |
|---|---|---|
| `name` | yes | Provider id used by `--provider` / `-pp`. Cannot shadow a built-in. |
| `compatible_with` | yes | API dialect. Currently only `openai`. |
| `base_url` | yes | API root; must be `http(s)://`. |
| `api_key_env` | yes | **NAME of the env var** holding the key — never the key. |
| `default_model` | no | Used only when nothing else specifies a model. |
| `default_model_env` | no | Env var checked before `default_model`. |
| `model_markers` | no | Substrings for auto-detecting the provider from a bare `--model`. Empty ⇒ never auto-claims. |
| `account_group` | no | Shared-account tag: providers with the same tag share one usage-limit verdict in `provider_availability.json`. It does **not** by itself place the provider in a failover chain — see below. |
| `extra` | no | Extra kwargs for the client (e.g. `{"temperature": 0}`). |

Then just use it — no code change, no restart of anything but the agent:

```bash
deep-worker myagent -pp minimax              # uses default_model
deep-worker myagent -pp minimax --model MiniMax-M2.1
deep-worker myagent --model MiniMax-M2.5     # auto-detected via model_markers
```

### Usage-limit failover

A provider declared here is a first-class failover target: name it in any
chain and an agent will fail over **to** it, and **off** it, exactly like a
built-in provider. Chains are operator-configured and addressed by name —
see `deep_worker/deep_worker_cli/docs/DESIGN_CONFIGURABLE_FAILOVER_CHAINS.md`.

Two fields interact with limits:

- `account_group` — the identity a limit is recorded against. All nine
  `openrouter` / `or-*` entries above share `openrouter-account`, so one 429
  marks the whole account exhausted. Omit it and the provider is its own
  account.
- The chain a provider appears in — this, and only this, decides where an
  agent goes when the provider is capped. A provider absent from the agent's
  chain is never a failover target.

## 3. Guarantees

**Secrets.** The loader rejects an entry carrying an inline `api_key` /
`secret` / `token`, or an `api_key_env` that looks like a pasted secret
instead of a variable name. Those errors never echo the offending value.

**Model names.** No model name is hardcoded in Python
(a standing project rule). Precedence:

1. explicit `--model` → sent to the vendor **verbatim**, never checked
   against a local list;
2. `$<default_model_env>`;
3. `default_model` from this config (operator's file, not code);
4. otherwise a clear error asking for one of the above.

An unknown model is rejected by the vendor itself with a real error, which
is exactly what we surface.

**Fail loud.** A malformed `providers.json` (bad JSON, missing field,
unknown dialect, duplicate or built-in-shadowing name, non-http `base_url`)
raises at startup with an actionable message. A **missing** file is fine —
it simply means no config-declared providers.

## 4. Where settings live

deep-worker's runtime tree is `~/.deep_worker`, and it is the **only** tree
consulted. The upstream ancestor's config directory is banned for both
reads and writes (POLICIES §16), so there is no legacy fallback: settings
that still live there are invisible and must be moved into `~/.deep_worker`.

Env precedence (first writer wins):

1. `~/.deep_worker/.env`
2. `~/.deep_worker/env.d/*.env`

## 5. MiniMax specifics (live-verified 2026-08-15)

Confirmed OpenAI-compatible: `/v1/models`, `/v1/chat/completions`, tool
calling, and SSE streaming all behave in the standard shape. Verified
end-to-end through `create_model`, including `bind_tools` producing
canonical `tool_calls`, plus usage metadata (LangChain even normalizes the
`cache_read` breakdown).

Two vendor quirks worth knowing:

* **`model` is REQUIRED.** Omitting it, or sending `""`/`"default"`, is a
  400. There is no server-side default and no short-alias resolution
  (`m3` → 400; the canonical id is `MiniMax-M3`). This is why the provider
  needs a `default_model` in config: unlike claude-code, there is no
  backend picker to defer to.
* **Inline `<think>` reasoning.** Models emit chain-of-thought inside the
  regular `content` field rather than a separate `reasoning_content`, so it
  reaches conversation history as visible text. See
  `PROVIDER_MINIMAX_COMPAT_NOTES.md`.
