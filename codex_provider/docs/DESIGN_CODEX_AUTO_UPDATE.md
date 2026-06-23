# Design: Auto-update of the `@openai/codex` CLI

**Status:** ✅ IMPLEMENTED — installed on this host 2026-06-05 (user-systemd
timer + helper script). The script and unit files live on the host (NOT in this
repo), but this document is the **source of truth** for what was put where and
why, so the setup is reproducible on a fresh box.

**Date:** 2026-06-05

## Why

`@openai/codex` is a fast-moving CLI: seven minor releases shipped in three
weeks (`0.130` → `0.137`, May–June 2026). Falling behind has real, recurring
operational consequences — the immediate trigger for writing auto-update was
a live incident on **2026-06-05**:

* An agent was started with `--restart <agent> --provider codex` (no
  `--model`).
* By design (DESIGN_CODEX_PROVIDER.md §15 Q2) we **do not** pass `--model` to
  `codex exec` when the user has not asked for a specific one — we let the
  Codex CLI pick its own current default. The full chain that enforces this is
  audited in detail at the bottom of this doc.
* On `0.130.0` the CLI's own default was `gpt-5.3-codex`. OpenAI had
  silently removed that model from ChatGPT-account access (multiple GitHub
  issues: [openai/codex#14306](https://github.com/openai/codex/issues/14306),
  [#14735](https://github.com/openai/codex/issues/14735),
  [#14190](https://github.com/openai/codex/issues/14190),
  [#14412](https://github.com/openai/codex/issues/14412),
  [#25839](https://github.com/openai/codex/issues/25839)). Result: HTTP 400
  `"The 'gpt-5.3-codex' model is not supported when using Codex with a ChatGPT
  account."` on the first turn.
* Upgrading the host CLI to `0.137.0` made it choose a default the account
  could actually use, and the same `--provider codex` (no `--model`) command
  worked immediately. Empirically verified on a throw-away probe agent before
  this doc was written.

So the bug was NOT in our code — it was the local CLI being three weeks
stale. The right fix is to stop letting that happen.

## What Codex itself offers (and what it does NOT)

Audited against the official OpenAI docs and the CLI itself on 2026-06-05:

| | Available | What it does |
|---|---|---|
| `codex update` subcommand | yes | One-shot, manual. Runs `npm install -g @openai/codex@latest` for you. NOT a daemon. |
| `check_for_update_on_startup` in `~/.codex/config.toml` | yes | Only **checks** at startup and prints a notice. Does NOT install. ([config reference](https://developers.openai.com/codex/config-reference)) |
| `auto_update` / `update_check_interval` / similar | **no** | Not exposed by the CLI. |
| Enterprise auto-deploy flag | **no** | Docs explicitly say `check_for_update_on_startup=false` is for environments where updates are "centrally managed" — implying centralised mechanisms (which is exactly what this doc sets up). |

Conclusion: there is no in-CLI way to auto-install. We must drive it
externally. This is consistent with how the CLI ships (an npm-global binary
with the same update story as any other npm tool).

## What we did

A user-systemd timer that fires the upgrade once a day. Three artefacts on the
host, all owned by the `develop` user — **no `sudo`** needed because the npm
prefix is `~/.npm-global` (verified):

### 1. `~/.local/bin/codex-auto-update.sh` (mode 755)

The actual upgrader. Idempotent and safe:

```bash
#!/bin/bash
# Auto-update @openai/codex globally if a newer version is available on npm.
# Invoked by the user-systemd timer `codex-auto-update.timer` (once a day).
set -u
PATH=$HOME/.npm-global/bin:/usr/local/bin:/usr/bin:/bin

log() { printf '[%s] %s\n' "$(date -Iseconds)" "$*"; }

command -v codex >/dev/null 2>&1 || { log "codex CLI not installed; skipping"; exit 0; }
command -v npm   >/dev/null 2>&1 || { log "npm not installed; cannot update"; exit 0; }

current=$(codex --version 2>/dev/null | awk '{print $NF}')
latest=$(npm view @openai/codex version 2>/dev/null)

if [ -z "$latest" ]; then
    log "could not fetch latest version from npm (network?); current=$current"
    exit 0
fi

log "codex current=$current  latest=$latest"
[ "$current" = "$latest" ] && { log "already on latest; nothing to do"; exit 0; }

log "updating codex $current -> $latest"
if npm install -g "@openai/codex@$latest" 2>&1; then
    log "update OK; codex now $(codex --version 2>/dev/null | awk '{print $NF}')"
else
    rc=$?; log "npm install failed (exit $rc) — staying on $current"; exit "$rc"
fi
```

Design notes:

* `set -u` only (no `set -e`) — we want explicit `exit 0` on no-op paths so a
  network blip never marks the unit as failed.
* `PATH` is pinned because systemd-user does not inherit the login shell PATH.
* The script never touches the npm prefix as root and never runs `npm install`
  unless `npm view` returned a non-empty version (guards against offline
  hosts).
* Output goes to stdout/stderr → captured by systemd-journal (see unit below)
  → auditable with `journalctl --user -u codex-auto-update.service`.

### 2. `~/.config/systemd/user/codex-auto-update.service`

```ini
[Unit]
Description=Auto-update @openai/codex CLI to the latest npm release
ConditionPathExists=%h/.local/bin/codex-auto-update.sh

[Service]
Type=oneshot
ExecStart=%h/.local/bin/codex-auto-update.sh
Nice=10
IOSchedulingClass=idle
TimeoutStartSec=10min
StandardOutput=journal
StandardError=journal
```

* `Type=oneshot` — fits the "run a script, exit" shape.
* `Nice=10` + `IOSchedulingClass=idle` — never compete with real work; a daily
  one-minute install is cheap regardless.
* `TimeoutStartSec=10min` — hard cap so a hung `npm install` cannot sit
  forever.
* `ConditionPathExists` — degrades cleanly if the script ever gets removed.

### 3. `~/.config/systemd/user/codex-auto-update.timer`

```ini
[Unit]
Description=Daily auto-update of @openai/codex CLI
PartOf=default.target

[Timer]
OnCalendar=*-*-* 04:00:00
RandomizedDelaySec=15min
Persistent=true
AccuracySec=10min

[Install]
WantedBy=default.target
```

Choices:

* **Cadence** — daily at 04:00 local. The cadence is tuned to the release
  rhythm (≈7 minors / 3 weeks); weekly would routinely lag behind a behaviour-
  changing CLI release like the one that triggered the incident above.
* **`RandomizedDelaySec=15min`** — be a good citizen of the npm registry: if
  every machine with this timer fires at exactly 04:00, we DDOS ourselves.
* **`Persistent=true`** — if the host was suspended overnight, the missed
  firing runs at boot instead of being dropped.
* **`AccuracySec=10min`** — saves wakeups; we don't need second-precision.
* **`PartOf=default.target`** + `WantedBy=default.target` + the user has
  `loginctl Linger=yes`: the timer survives logout/reboot.

### One-time install

```bash
systemctl --user daemon-reload
systemctl --user enable --now codex-auto-update.timer
```

Confirmed listed by `systemctl --user list-timers codex-auto-update.timer` and
exercised manually via `systemctl --user start codex-auto-update.service`
(journal shows the no-op path on a host already at latest).

## Operations cheat sheet

| Want to… | Run |
|---|---|
| See when it last ran / next will run | `systemctl --user list-timers codex-auto-update.timer` |
| Read history of runs | `journalctl --user -u codex-auto-update.service` |
| Force a check right now | `systemctl --user start codex-auto-update.service` |
| Pause auto-updates | `systemctl --user disable --now codex-auto-update.timer` |
| Change frequency | edit `~/.config/systemd/user/codex-auto-update.timer` (`OnCalendar=`), `systemctl --user daemon-reload`, `systemctl --user restart codex-auto-update.timer` |
| Switch to manual mode | `npm install -g @openai/codex@latest` |

## Why we don't pin a Codex CLI version on our side

The same reason we don't pin a model in `_PROVIDER_DEFAULT_MODELS["codex"]`
(see DESIGN_CODEX_PROVIDER.md §15 Q2): the CLI evolves quickly, and the value
of our provider is that we ride along with it instead of getting stuck on a
snapshot. Auto-update on the host is the logical extension of that same
principle: the CLI is "infrastructure", not a vendored artefact.

## Audit: how we keep "default model" out of our codebase

Recorded here too because this is also the entry point a future engineer will
find when investigating model-default questions. Verified at the strings/code
level on 2026-06-05:

1. The CLI's `config.py:1395` — `"codex": ("CODEX_MODEL", None)`.
   Env var `CODEX_MODEL`, fallback `None`. No literal model name.
2. The CLI's `config.py:1411-1417` — `get_default_model_name("codex")`
   returns `os.environ.get("CODEX_MODEL", None)` → `None` unless the operator
   set the env var.
3. The CLI's `config.py:1481` — `create_model(provider="codex",
   model_name=None)` calls (2), so `model_name` stays `None`.
4. The CLI's `cli/agent_command.py:268-276` (`build_launch_config`,
   `--restart` path) — `if model is None: model = get_default_model_name(
   args.provider)` and the inline comment explicitly forbids coercing to
   `"unknown"` (a related internal incident).
5. `codex_provider/chat_model.py:282` — `self._model_name = model` (None
   propagates).
6. `codex_provider/chat_model.py:624` — `ExecArgs(model=self._model_name, …)`
   → `ExecArgs(model=None)`.
7. `codex_provider/exec_runner.py:478-479` — `if args.model: argv.extend(
   ["--model", args.model])`. With `model=None`, `--model` is NOT added to
   the `codex exec` argv. Codex CLI then picks **its own** current default
   — which is exactly the behaviour this doc keeps current with reality by
   keeping the CLI up to date.

The literal `gpt-5.3-codex` exists in our tree in exactly **two** places, both
in
the CLI's `tests/integration_ai/test_codex_inter_agent_send_and_reply.py`,
and both are descriptive comments stating the CLI's own behaviour at the time
of writing — not code we run.

## Risks + rollback

* **Risk: a broken upstream release lands.** Mitigation: the install runs
  under `develop` only, fails closed (no overwrite of an older binary except
  by `npm install`), and `systemctl --user disable --now
  codex-auto-update.timer` rolls forward to manual mode. Recovery is
  `npm install -g @openai/codex@<known-good>`.
* **Risk: registry outage / DNS failure.** The script returns 0 with a log
  line; nothing else happens. The host stays on the previous version.
* **Risk: `npm install` hangs.** `TimeoutStartSec=10min` kills it and the
  next timer firing tries again.

## Source links

* npm package: <https://registry.npmjs.org/@openai/codex>
* CLI changelog: <https://developers.openai.com/codex/changelog>
* Config reference (`check_for_update_on_startup`):
  <https://developers.openai.com/codex/config-reference>
* GitHub releases: <https://github.com/openai/codex/releases>
