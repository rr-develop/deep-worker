# dw-hist

Host-level CLI for inspecting deep-worker agent conversation history. Standalone
from `deep-worker` — see [`docs/DESIGN_DW_HIST_INITIAL.md`](docs/DESIGN_DW_HIST_INITIAL.md)
for the design.

## Install

`dw-hist` ships as a source tree + bash launcher (same convention as
`deep-worker` / `dwcli`). To make it globally available, symlink the launcher
into `~/.local/bin/`:

```bash
# Point this at your checked-out dw_hist tree:
ln -sf /path/to/dw_hist/dw-hist ~/.local/bin/dw-hist
```

Ensure `~/.local/bin` is on your `$PATH`. The launcher sets `PYTHONPATH`
internally — no `pip install` is needed (project policy).

## Commands (v1)

### List agents that have a built history index

```bash
dw-hist --list
dw-hist --list --format json
```

Exits `0` even when no agents have an index (prints nothing on stdout), so
shell loops are safe:

```bash
for a in $(dw-hist --list); do
    dw-hist "$a" -q "deadlock"
done
```

### Search one agent's history semantically

```bash
dw-hist <agent_name> -q "<query text>"
dw-hist <agent_name> -q "<query>" -k 10
dw-hist <agent_name> -q "<query>" --format json
```

Output mirrors `history_memory`'s tool output and includes the explicit
similarity `score` for each result. `--format json` emits a flat record list
for scripting.

The search runs strictly offline against the embedding model that is already
in the local HF cache; third-party noise (HF Hub warnings, tqdm progress
bars, transformers `BertModel LOAD REPORT`, llama_index's
`"LLM is explicitly disabled. Using MockLLM."` banner) is suppressed by
default so the output stays clean. Pass `-v` / `--verbose` to surface all
of it on stderr when debugging cold-cache or model-load issues.

### Search one agent's per-message embeddings (`-qm`)

```bash
dw-hist <agent_name> -qm "<query text>"
dw-hist <agent_name> -qm "<query>" -k 10 --format json
dw-hist <agent_name> -qm "<query>" -f SUMMARIZATION
dw-hist <agent_name> -qm "<query>" -f PLAIN_USER -f TASK_MANAGER
```

`-qm` (mutually exclusive with `-q`) ranks each **single message** by the
query's projection (dot product == cosine, both sides are L2-normalised) onto
that message's embedding in the per-message store
(`{summaries}/.message_embeddings/`, built by
`semantic_search.message_embedding_store`). Output mirrors `-q` and additionally
shows each hit's `Flags:` (the `MessageFlag` bits set on that message).

`-f` / `--flag` (repeatable, logical OR) restricts scoring to messages that have
at least one of the named flags. Valid names (case-insensitive): `USER_ROLE`,
`TASK_MANAGER`, `PLAIN_USER`, `AGENT`, `AGENT_TO_AGENT`, `TOOL_RESULT`,
`TOOL_CALL`, `SYSTEM`, `SUMMARIZATION`, `AGENT_FINAL`. `-f` requires `-qm`.

Exit `3` if the agent has no per-message embedding store (distinct from the
`-q` history index); build it with `build_message_embeddings(...)` +
`build_message_metadata(...)` from `semantic_search`.

### Isotropy transforms for `-qm` (`-c` / `-w`)

e5 embeddings are strongly anisotropic (~99.6% of a raw score is a shared "cone"
component, only ~0.4% is topical), which makes **diffuse** queries — abstract
or emotional, without rare anchor terms — surface short noise reliably. Two
opt-in modifiers rank in an isotropic space instead:

```bash
dw-hist <agent> -qm "failure, frustration" -f AGENT -c         # centering
dw-hist <agent> -qm "failure, frustration" -f AGENT -w         # whitening: zca + truncate to 90% variance
dw-hist <agent> -qm "..." -w -wv 0.95                         # keep 95% of variance
dw-hist <agent> -qm "..." -w -wd 128                          # keep exactly 128 axes
dw-hist <agent> -qm "..." -w -wv 1.0                          # full whitening (no truncation)
dw-hist <agent> -qm "..." -c --stats-scope filtered --score-stats
```

- `-c` / `--centering` — subtract corpus μ, renormalise, cosine. Cheap.
- `-w` / `--whitening` — additionally equalise per-axis variance via `(x-μ)·W`.
  `--whitening-method` `pca` | `zca` (default `zca`); the two yield identical
  cosine rankings (cosine is invariant to the orthogonal factor that
  distinguishes them) — the option is kept for future non-cosine experiments.
  `--whitening-eps` regularises the kept eigenvalue block (`λ_r+ε`) for
  numerical safety (default `1e-3`, must be > 0).
- **Truncation (why `-w` truncates by default).** e5's covariance is extremely
  ill-conditioned (condition number ~10¹⁰): hundreds of near-zero-variance axes
  are pure numerical noise, and full whitening amplifies them by `1/√λ` (up to
  ~6·10⁵), so noise dominates the ranking and a long, lexically-rich message can
  be pushed to rank 1. `-w` therefore keeps only the top principal axes and
  **drops the noisy tail**:
  - `--whitening-var` / `-wv` `F` — keep the top axes covering fraction `F` of
    total variance (default **`0.90`**). `-wv 1.0` = full whitening.
  - `--whitening-dims` / `-wd` `N` — keep exactly `N` top axes (overrides `-wv`).
  - `-wv` and `-wd` are mutually exclusive; `0 < F ≤ 1`, `N ≥ 1`.
  - When whitening truncates, a one-line notice on **stderr** reports the axes
    kept, e.g. `whitening: kept 141 of 384 axes (variance target 0.90)` — shown
    even without `--score-stats`; stdout stays JSON-parseable.
- `--stats-scope` `corpus` | `filtered` — rows used to estimate μ (and Σ). Default
  `corpus` (most stable). `filtered` isotropises relative to the `-f` subset.
- `--score-stats` — one-line score distribution (min/median/max/range over all
  M candidates) to **stderr**; stdout remains JSON-parseable under `--format json`.
- `-c` and `-w` are mutually exclusive.
- Transformed scores are the residual cosine and can be **negative**; their
  magnitude is NOT comparable across modes (different spaces).
- Whitening needs at least `dim+1` stats rows (`dim=384`). With a narrow
  filter + `--stats-scope filtered` the CLI exits `4` with a clear message —
  widen the filter, use `--stats-scope corpus`, or drop `-w`.

History that grew after the store was built (summarisation appends new turns)
does NOT break `-qm`: row re-derivation is bounded by the store's manifest
watermark, so newly-appended messages are ignored until the store is next
rebuilt, and analysis of the already-indexed messages is unaffected.

See `docs/DESIGN_DW_HIST_QM_ISOTROPY.md` for the math and the anisotropy
diagnosis.

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | success / `--list` with zero agents |
| 2 | usage error (conflicting flags, invalid `--top-k`, unknown agent) |
| 3 | agent exists but has no history index |
| 4 | search runtime error |

## What v1 deliberately does NOT do

- No reindex / clear / info commands yet (placeholders in the design).
- No agent-runtime integration — this is a read-only host operator's tool.
- No `pip install`, no `pyproject.toml` — source-imports only.
