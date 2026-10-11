# MiniMax provider — live compatibility notes (2026-08-15)

Status: **investigation only. No provider code written.** Design questions
1/3/4 (config location, scope, `<think>` handling) and the reformulated Q2
(fail-loud vs operator-config default) are awaiting the owner. This file
exists so the findings survive between sessions.

Related: `API_KEY_SECURITY_POLICY.md` (key storage),
`../deep_worker/deep_worker_cli/deep_worker_cli/process/env_loader.py`
(the `env.d` drop-in loader).

## 1. Credential storage (DONE, per policy)

* Key lives in `~/.deep_worker/env.d/minimax.env` as `MINIMAX_API_KEY=...`,
  created mode `0600`; the `env.d` directory was `0777` and was tightened
  to `0700`.
* The original `~/tmp/mm_k.txt` was shredded (`shred -u`).
* Nothing about the key is in the repo: `git grep -i minimax` and a literal
  value scan over the worktree both return nothing.
* Verified with the SHIPPED loader (`load_env_d`), including a negative
  control: at `0644` the loader SKIPS the file and the variable never
  reaches `os.environ` — so a permissions slip fails closed, loudly.

## 2. OpenAI compatibility — live-verified

Base URL `https://api.minimax.io/v1` (`https://api.minimaxi.chat/v1`
serves the same catalogue).

| Capability | Result |
|---|---|
| `GET /v1/models` | 200, OpenAI-shaped `{object:"list", data:[...]}` |
| `POST /v1/chat/completions` | 200, standard `choices[].message` |
| Tool calling | Works; canonical `tool_calls[].function.arguments` (JSON string) |
| Streaming | `text/event-stream`, standard `data:` frames |

Conclusion: usable via `ChatOpenAI(base_url=..., api_key=...)`.

## 3. Model selection — the constraint that shapes the design

`/v1/models` returns ONLY `{id, object, created, owned_by}` for all 8
entries. There is **no** `recommended` / `default` / `deprecated` / `status`
marker (verified over the union of keys across every entry).

Live probes of the "let the backend choose" pattern:

| Request | Result |
|---|---|
| `model` omitted | **400** `missing required parameter (2013)` |
| `model: ""` | **400** `unknown model ''` |
| `model: "M2.1"` (short alias) | **400** `unknown model 'm2.1'` |
| `model: "abab"` (short alias) | **400** `unknown model 'abab'` |

**This is the key architectural difference from claude-code.** There,
the adapter owns a per-session picker and resolves short class names, so
"send no model and let the backend pick the current default class"
(`feedback_no_model_hardcode_2026_08_15` rule 1) works. MiniMax has no
such mechanism: `model` is REQUIRED and aliases are not resolved, so a
concrete id must come from somewhere.

### Why "poll /v1/models and auto-pick" was rejected

Proposed during the round as a way to honour the no-hardcode rule. Rejected
on evidence:

* No recommended/default marker exists, so "pick the recommended one" is
  not implementable.
* "First in the list" relies on undocumented response ordering.
* "Newest by `created`" would silently migrate every agent onto a brand-new
  untested model the day the vendor ships it — an unreviewed production
  model change as a side effect of a heuristic. `created` is also NOT
  unique (`M2.7` and `M2.7-highspeed` tie), so the rule is ambiguous.

That reintroduces the requested-vs-executed divergence that round 7
(`e9889d4`) existed to remove — and non-deterministically.

### What the no-hardcode rule actually forbids

Re-reading the rule: the banned artifact is a **whitelist that REJECTS user
input** (`normalize_model` + `KNOWN_MODEL_ALIASES` refusing `fable`, plus a
`raw:` escape hatch). A default applied only when the operator specified
nothing rejects nothing.

Proposed precedence (no model-name literal anywhere in Python):

1. explicit `--model` → passed to the backend VERBATIM (no local validation);
2. else `$MINIMAX_MODEL` (`default_model_env`);
3. else `default_model` from the operator's `providers.json` (config, not code);
4. else **fail loud**, listing the models from a LIVE `GET /v1/models`
   so the operator chooses.

The live catalogue is used where it belongs — in the error message
(same shape as the round-7 picker error) — never to guess on the vendor's
behalf. Unknown model names are rejected by MiniMax itself (400 + message),
which satisfies rule 4 (surface the backend's real error).

## 4. Two quirks to handle if/when implementation is approved

* **Inline `<think>` reasoning.** Models emit chain-of-thought inside the
  regular `content` field (`"<think>\nThe user wants...\n</think>\n"`),
  not a separate `reasoning_content`. Left as-is it lands in conversation
  history and token accounting. Needs a decision (owner Q4).
* **Sparse `usage`.** Only `{total_tokens, total_characters, prompt_tokens,
  completion_tokens}` — no `prompt_tokens_details`, so no cache-read
  breakdown. Token stats work but the cached-input column stays empty.
