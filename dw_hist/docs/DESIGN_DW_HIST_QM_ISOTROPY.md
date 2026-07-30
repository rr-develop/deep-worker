# dw-hist `-qm` isotropy transforms: `--centering` / `--whitening`

Status: design (proposed)
Scope: `dw_hist/` (CLI surface) + `semantic_search/message_search.py`
(ranking math). No new deps, no new threads, no store writes, no embedding
recompute. Read-only over the existing per-message embedding store.

## 1. Motivation

`-qm` ranks per-message embeddings by cosine (dot product of unit vectors).
e5 embeddings are strongly **anisotropic** — they sit in a narrow cone, so the
raw cosine is dominated by a shared "common direction" component (~99.6% of the
average score on this corpus) and only ~0.4% is topical. For **diffuse** queries
(abstract / emotional, no rare anchor terms) this makes the ranking noisy: short
filler messages outrank on-topic ones because they align slightly better with
the corpus mean direction. (Measured: AGENT-subset score range was only 0.107;
the weak hit "The log is being written. Wait…" sat at the 99.9th percentile.)

Two standard isotropy transforms fix this by removing the common component:

* **Centering** — subtract the corpus mean direction μ, re-normalise, cosine.
  Cheap; empirically expanded the score range ×4 (0.107 → 0.43) and produced a
  semantically sensible top-k.
* **Whitening** — `x' = (x − μ)·W`, with `W` from the corpus covariance Σ, then
  re-normalise, cosine. Removes the mean AND equalises per-axis variance (full
  isotropy).

This feature exposes both as opt-in `-qm` modifiers so the operator can run
isotropy experiments. Default behaviour (raw cosine) is unchanged.

## 2. CLI surface

All modifiers attach to `-qm` (error if used without it).

| flag | type | meaning |
|------|------|---------|
| `-c` / `--centering` | bool | rank in the mean-centered space (subtract μ, renorm) |
| `-w` / `--whitening` | bool | rank in the whitened space (`(x−μ)·W`, renorm) |
| `--whitening-method` | `pca`\|`zca` | whitening variant (default **`zca`**) |
| `--whitening-eps` | float | covariance regularisation ε (default `1e-3`) |
| `--whitening-var` / `-wv` | float in (0,1] | keep the top principal axes covering this fraction of total variance; drop the noisy tail (default **`0.90`**) |
| `--whitening-dims` / `-wd` | int ≥ 1 | keep exactly this many top axes (absolute override of `-wv`) |
| `--stats-scope` | `corpus`\|`filtered` | rows used to estimate μ/Σ (default **`corpus`**) |
| `--score-stats` | bool | also print the score distribution (min/median/max/range) to stderr for diagnostics |

* `-c` and `-w` are **mutually exclusive** (→ usage error, exit 2).
* `--whitening-method` / `--whitening-eps` / `--whitening-var` / `--whitening-dims`
  require `-w` (ignored-with-warning is worse than an explicit error; → usage
  error if given without `-w`).
* `--whitening-var` and `--whitening-dims` are **mutually exclusive** (both cap
  the kept-axis count; giving both is ambiguous → usage error, exit 2).
* `--whitening-var` must be in `(0, 1]`; `--whitening-dims` must be `≥ 1`
  (→ usage error otherwise). `--whitening-dims` is clamped to `dim` (can't keep
  more axes than exist) and, after the `< dim+1` stats-row guard, to the number
  of numerically-usable axes (see §3.6).
* `--stats-scope` requires `-c` or `-w` (no statistics needed for raw).
* `--score-stats` is valid in any `-qm` mode (incl. raw).

### 2.1 Examples
```bash
dw-hist <agent> -qm "failure, frustration" -f AGENT                 # raw (current)
dw-hist <agent> -qm "failure, frustration" -f AGENT -c             # centering
dw-hist <agent> -qm "failure, frustration" -f AGENT -w             # whitening: zca + truncate to 90% variance (default)
dw-hist <agent> -qm "..." -w -wv 0.95                             # keep 95% of variance
dw-hist <agent> -qm "..." -w -wd 128                              # keep exactly 128 axes
dw-hist <agent> -qm "..." -w --whitening-method pca --whitening-eps 1e-2
dw-hist <agent> -qm "..." -c --stats-scope filtered --score-stats
```

## 3. Ranking math (`message_search.py`)

`search_messages` gains parameters (all defaulting to the current behaviour):

```python
def search_messages(*, query, summaries_dir, top_k, flags_any=0,
                    transform="raw",          # "raw" | "centering" | "whitening"
                    whitening_method="zca",   # "pca" | "zca"
                    whitening_eps=1e-3,
                    whitening_var=0.90,       # keep top axes covering this variance fraction
                    whitening_dims=None,      # int | None — absolute axis count (overrides var)
                    stats_scope="corpus",     # "corpus" | "filtered"
                    stats_out=None,
                    embed_model=None, config=None) -> list[MessageSearchResult]
```

Algorithm (numpy only, no torch; nothing persisted):

1. Load `embs (N,dim)` and `flags (N,)` as today. Embed the query once → `q`.
2. **Statistics rows.** `stats_scope="corpus"` → use all `N` rows for μ/Σ
   (default; most stable). `stats_scope="filtered"` → use only the `-f`
   candidate rows (lets the operator isotropise relative to the subset).
   Guard: whitening needs enough rows to estimate Σ; if the stats set has
   `< dim+1` rows raise a `RuntimeError` (→ exit 4). **RC-3 (actionable
   message):** the exception text names all three remedies —
   *"whitening needs at least {dim+1} stats rows, got {n}: widen the filter,
   use `--stats-scope corpus`, or drop `-w`"*. A unit test exercises this exact
   path (filtered scope, few candidates, `-w` → RuntimeError with each remedy
   substring present) so the message stays actionable across refactors.
3. Compute the transform from the stats rows:
   * `centering`: `μ = mean(stats)`. Map `f(x) = normalize(x − μ)`.
   * `whitening`: `μ = mean(stats)`; `Σ = cov(stats)`; eigh `Σ = U diag(λ) Uᵀ`.
     **Truncate the noisy tail** (§3.6): keep only the top `r` axes (largest λ);
     `r` is chosen from `whitening_dims` (absolute) or `whitening_var` (variance
     fraction). Let `U_r`, `λ_r` be the kept eigenvectors/values. Regularise the
     kept block `λ_r + ε` and form
     `W = U_r diag((λ_r+ε)^{-1/2})` (pca) or `U_r diag((λ_r+ε)^{-1/2}) U_rᵀ` (zca).
     Map `f(x) = normalize((x − μ)·W)`. With the tail dropped, the near-zero
     eigenvalues no longer explode `λ^{-1/2}` and inject noise.
   * `raw`: `f(x) = x` (already unit-norm).
4. Apply `f` to BOTH the candidate rows and the query (same μ/W — they must live
   in the same space). `scores = f(cand) @ f(q)`.
5. Top-k via `argpartition` with `k_eff = min(top_k, M)` (unchanged), tie-break
   by ascending row index. Build `MessageSearchResult`s as today; `score` now
   carries the transformed cosine.
6. `--score-stats` (CLI) prints min/median/max/range of the candidate scores.

### 3.1 PCA vs ZCA — documented equivalence for cosine ranking
PCA and ZCA whitening differ only by an orthogonal rotation (`Uᵀ`). Cosine is
**invariant** under a shared orthogonal map applied to both operands, so for
this cosine ranking PCA and ZCA produce the **same ordering and identical
scores** (verified numerically: top-k scores match to ~1e-12). The option is
kept for future experiments / non-cosine metrics, and **ZCA is the default**
because its `W` is symmetric and keeps the original axes (more interpretable;
`max|W−Wᵀ| ≈ 3e-15` in the prototype). This equivalence is stated in the
docstring and README so the operator is not surprised that the two coincide.

### 3.2 Why subtracting μ then renormalising is "centering"
Anisotropy = a large shared mean component. `normalize(x − μ)` removes exactly
that component and rescales, so cosine then reflects the residual (topical)
direction. Whitening additionally divides each principal axis by its std, so no
single high-variance direction dominates. Centering is the rank-1 special case
of whitening.

### 3.3 Zero-norm guard (RC-1)
`normalize(v)` MUST guard `‖v‖ == 0` and return the zero vector in that case
(same convention as `embed_long_text` on empty text). Otherwise a row that
equals μ (centering) or lands at the origin after whitening produces
`0/0 = NaN`, and NaN silently corrupts `argpartition` / `sort` ordering. Guarded
form:
```python
n = np.linalg.norm(v, axis=-1, keepdims=True)
return np.where(n > 0.0, v / np.maximum(n, 1e-12), 0.0)
```
Score for such a row is exactly `0.0` (dot with a normalised query), not NaN.
Regression tests cover both centering and whitening.

### 3.4 Numerical / efficiency notes
* μ/Σ are computed in float64 for stability; the query vector is **also cast to
  float64 before `(x−μ)·W`** (NIT) so both operands share dtype through the
  transform; results cast back to float32.
* Σ is `dim×dim` (384²); eigh is sub-second once per call. The transform of the
  candidate matrix is one `(M,dim)·(dim,dim)` matmul. Still effectively instant.
* `εI` regularisation keeps tiny eigenvalues from exploding `λ^{-1/2}`.
* No statistics are persisted — recomputed per call from the loaded store. (A
  persisted/cached `W` is a possible future optimisation, explicitly out of
  scope: keeps the store read-only and the math reproducible.)

### 3.5 Append-only history: row re-derivation is watermark-bounded (FIX)

`search_messages` re-derives each row's *source* data (file / physical line /
text / type / timestamp) to display hits, via `_row_messages`. The stored
embedding rows are **frozen** at build time; the history JSONL files, however,
keep growing (summarisation appends new turns after every session). Re-walking
the *current* files therefore yields MORE messages than the frozen store, and
the old 1-to-1 check hard-failed with
`re-derived message count (X) != store rows (N) … rebuild it` — breaking `-qm`
entirely (all three modes) whenever any new message had been appended since the
last store build. That violated the invariant that **later messages must never
break analysis of earlier ones.**

Fix: bound the re-walk by the store's own watermark. The manifest already
records `file_entry_counts` — per source file, the number of *parsed* entries
(same unit as `count_jsonl_lines`; blank/undecodable lines already dropped)
consumed when the store was built. `message_embedding_store.store_file_entry_counts()`
(new underscore-free public accessor, §5-style, avoids a `reportPrivateUsage`
cross-module import) exposes it. `_row_messages` counts parsed entries per file
and STOPS at that file's watermark, reproducing exactly the prefix the rows came
from; appended entries are ignored. Rows are append-only and order-stable
(DESIGN_MESSAGE_EMBEDDING_STORE.md §3), so the bounded prefix maps 1-to-1 onto
the store rows by construction. A file absent from the watermark (legacy store
built before watermark tracking) is walked in full — previous behaviour.

The 1-to-1 `RuntimeError` is **kept** but now fires only after applying the
watermark: i.e. only on a genuine *rewrite* (a file shrank or an earlier entry
was edited to non-embeddable), not on ordinary appends. Physical line numbers
for display (RC-1) are unaffected — the counter that gates the watermark is the
parsed-entry counter, while the emitted `line` remains the physical line.

### 3.6 Truncated whitening: drop the noisy tail (FEATURE)

**Motivation (measured).** e5's covariance is extremely ill-conditioned — on a
real ~7k-message store the eigenvalue spectrum spans `λ_max ≈ 3e-2` down to
`λ_min ≈ 2.5e-12` (condition number `κ = λ_max/λ_min ≈ 1.2e10`). Full whitening
scales every axis by `λ^{-1/2}`, so the hundreds of near-zero-variance tail axes
(pure numerical noise — no real information) get amplified by up to `1/√(2.5e-12)
≈ 6.3e5`. That amplified noise dominated the ranking: a long, lexically-rich
"final report" message with incidental projections onto those tail axes was
pushed to rank 1 for a diffuse query — objectively the *least* relevant hit.
`--whitening-eps` only softens the tail; it does not remove it, and the report
still ranked 1 even at ε=1e-2.

**What truncation does.** Sort the eigen-axes by variance `λ` (descending). Keep
only the top `r` axes and **discard the rest** before building `W`. The dropped
tail is exactly the low-information noise directions, so it can no longer be
amplified. This is standard PCA/ZCA-whitening truncation. Empirically (real
store, diffuse query, `-f AGENT`): keeping ~90% of variance (`r≈141` of 384)
drops the report out of the top-5 and surfaces concrete, on-topic messages;
keeping 100% keeps the report at rank 1; keeping too little (<80%) over-truncates
and drifts off-topic. Hence the **default `--whitening-var 0.90`**.

**Choosing `r`.**
* `whitening_dims` (absolute): `r = clamp(whitening_dims, 1, r_max)`.
* `whitening_var = F` (default `0.90`): sort λ descending, take the cumulative
  sum, and pick the smallest `r` whose cumulative variance ratio
  `Σ_{i≤r} λ_i / Σ λ` ≥ `F`. This is the "keep axes covering fraction F of total
  variance; drop axes carrying the remaining (1−F)" rule — the mathematically
  precise reading of "drop axes with < (1−F) of the information".
* `whitening_dims` overrides `whitening_var` when both are internally set; the
  CLI additionally rejects passing both explicitly (§2, exit 2) so this override
  is only ever reached via the API default, never a conflicting user request.

**Interaction with the stats-row guard and ε.** `r_max` is the number of
numerically-usable axes: at most `dim`, and never more than the stats-row count
supports (Σ estimated from `m` rows has rank ≤ `m−1`). The existing
`m < dim+1` → `RuntimeError` (RC-3) is unchanged and still fires first. ε is
still added to the KEPT block (`λ_r + ε`) for numerical safety, but with the
tail already gone ε is no longer load-bearing; the default `1e-3` is retained.

**Equivalence & scope.** PCA and ZCA truncated identically in cosine ordering
(the §3.1 argument holds on the kept `r`-dim subspace: they differ by the shared
orthogonal `U_rᵀ`). Truncation is applied to BOTH candidates and the query
through the SAME `W` (built once), preserving the core invariant. `centering` and
`raw` are unaffected. The store stays read-only; nothing is persisted.

**Default behaviour change (approved).** Plain `-w` now means "zca + truncate to
90% variance", not full whitening. This is a deliberate change: full whitening
was shown to produce the worst top hit, so the out-of-the-box `-w` should be the
good variant. Full whitening remains reachable via `-wv 1.0` (or `-wd 384`).
Documented in the README and `--help`.

## 4. dw-hist integration

* `_build_parser`: add the flags. `-c`/`-w` as `store_true`;
  `--whitening-method` (choices), `--whitening-eps` (float),
  `--whitening-var`/`-wv` (float), `--whitening-dims`/`-wd` (int),
  `--stats-scope` (choices), `--score-stats` (store_true). `-wv`/`-wd` use
  argparse `default=None` (a sentinel, NOT the numeric default) so "was it
  supplied?" is detectable for validation; the numeric default (`whitening_var=
  0.90`) lives in `search_messages`, applied only when neither was passed.
* `main`: validation (all → exit 2 with a clear message):
  * `-c` + `-w` together → error.
  * `-c`/`-w`/`--stats-scope`/whitening modifiers without `-qm` → error (extend
    the existing "`-f` only with `-qm`" rule to the new modifiers).
  * `--whitening-method`/`--whitening-eps`/`--whitening-var`/`--whitening-dims`
    without `-w` → error.
  * `--whitening-eps <= 0` → error.
  * `--whitening-var` outside `(0, 1]` → error.
  * `--whitening-dims < 1` → error.
  * `--whitening-var` AND `--whitening-dims` both supplied → error (ambiguous).
  * "supplied?" is detected by comparing `argv`/sentinel, matching the existing
    approach for the other modifiers (not by value equality with the default).
* `_cmd_search_messages`: resolve `transform` from the flags, pass the new
  params through to `search_messages`. On `--score-stats`, print the
  distribution to stderr after the search (kept off stdout so `--format json`
  stays parseable).
* `SearchMessagesFunc` Protocol widened with the new keyword params (defaulted).
  **RC-2 (test mechanics):** a Protocol default does NOT make concrete callable
  stubs tolerant of new kwargs — a stub written as
  `def __call__(*, query, summaries_dir, top_k, flags_any): ...` will raise
  `TypeError` when `_cmd_search_messages` unconditionally forwards
  `transform=`/`whitening_*=`. Fix: **update the existing test-only stubs in
  `tests/unit_tests/test_cli.py` to accept the new defaulted kwargs** (e.g.
  `**kwargs` on the shared `fake_search`), not by conditionally forwarding on
  the CLI side (which would leak defaults to the API and prevent explicit
  raw calls). The CLI always forwards all new params by keyword.

## 5. Output

* Result formatting unchanged (the existing `score`/`flags`/`flag_names`
  columns). `score` now reflects the active transform; for centering/whitening
  it is the residual cosine and can be **negative** (orthogonal/opposite to the
  query in the isotropic space) — `format_text` already prints `score=%.3f`, so
  negatives render fine. README notes that transformed scores are not comparable
  in magnitude to raw scores (different spaces).
* `--score-stats` adds a one-line stderr summary over ALL `M` candidate scores
  (NIT — not the top-k slice, which would defeat the diagnostic purpose), e.g.
  `score stats (whitening, M=2492): min=-0.167 median=0.014 max=0.265 range=0.432`.
* **Kept-axis report (user request).** When whitening truncates, the number of
  axes actually kept (`r`) and how it was chosen are surfaced so the operator
  can see the effect of `--whitening-var`. `search_messages` reports it via
  `stats_out` — a new key `whitening_dims_kept: int | None` (`None` when no
  truncation applies, i.e. non-whitening transforms or `r == dim`). The CLI
  prints a dedicated one-line **stderr** notice whenever `-w` truncates,
  *independently of* `--score-stats* (so `-wv` alone shows it):
  `whitening: kept 141 of 384 axes (variance target 0.90)` for `-wv`, or
  `whitening: kept 128 of 384 axes (requested 128)` for `-wd`. Stays on stderr so
  `--format json` stdout remains parseable. `stats_out` also carries
  `whitening_var`/`whitening_dims_requested` for completeness.
* `--help` for `--whitening-eps` states it regularises Σ (`Σ + εI`) before eigh
  so tiny eigenvalues don't blow up `λ^{-1/2}` (NIT). `--help` for
  `--whitening-var`/`--whitening-dims` explains truncation (keep top axes by
  variance; drop the noisy tail; default `-wv 0.90`) and that `-wv 1.0`/`-wd dim`
  restore full whitening.

## 6. Testing

Unit (no model; fake embedder; small hand-built store):
* `transform="centering"`: on a tiny store where one row is the corpus mean and
  others are spread, assert the mean row's centered score is ~0 and a row
  aligned with the (de-meaned) query ranks first. Assert query is transformed
  with the SAME μ (regression: if only rows were centered, ranking breaks).
* `transform="whitening"` (zca & pca): assert identical ordering/scores for the
  two methods (the §3.1 equivalence); assert a known isotropic case ranks by the
  intended residual; assert ε regularisation path runs.
* **Truncation (§3.6):**
  * `whitening_dims=r` keeps exactly `r` axes: `stats_out["whitening_dims_kept"]
    == r`; `r` clamped to `dim` and to the usable rank (`m−1`).
  * `whitening_var=F` picks the smallest `r` whose cumulative variance ≥ `F`;
    verify on a hand-built spectrum with known λ (e.g. two large + several tiny
    eigenvalues → `F=0.9` keeps only the large ones); `stats_out` reports that
    `r`.
  * `whitening_var=1.0` (or `whitening_dims=dim`) == full whitening (regression
    vs the pre-truncation result on the same store); `whitening_dims_kept` is
    `None` (no truncation) in that case.
  * default (neither passed) truncates at `0.90`.
  * `whitening_dims` overrides `whitening_var` when both set at the API level.
* `stats_scope="filtered"`: μ differs from the corpus μ → different ranking than
  `corpus`, verifying the scope actually changes the statistics set.
* Degenerate Σ (stats rows < dim+1 with `-w`) → `RuntimeError`.
* `raw` path byte-identical to today (regression).
* Watermark (§3.5): appending a message after build does NOT raise and does NOT
  change the ranking/scores over the frozen rows; `_row_messages` bounded by the
  watermark truncates the appended tail; the watermark unit is parsed entries
  (blank/bad physical lines don't shift it); a genuine *rewrite* still raises.
* `store_file_entry_counts` (in `test_message_embedding_store.py`): records the
  parsed-entry count; stays frozen after an append; `{}` when no store exists.
CLI (`test_cli.py`, stubbed `search_messages`):
* `-c`+`-w` → exit 2; `--whitening-method` without `-w` → exit 2;
  `-c` without `-qm` → exit 2; `--whitening-eps 0` → exit 2.
* `-wv` and `-wd` together → exit 2; `-wv 0`/`-wv 1.5` → exit 2; `-wd 0` → exit 2;
  `-wv`/`-wd` without `-w` → exit 2; `-wv`/`-wd` without `-qm` → exit 2.
* happy path: `-wv 0.95`/`-wd 128` map to `whitening_var`/`whitening_dims` passed
  to the stub; aliases `-wv`/`-wd` resolve to the same params as the long forms.
* kept-axis notice: when the stub reports `whitening_dims_kept` via `stats_out`,
  `-w` emits the `whitening: kept R of D axes …` stderr line even WITHOUT
  `--score-stats`, and stdout stays JSON-parseable under `--format json`.
* `--score-stats` emits a stderr line and keeps stdout JSON parseable.

Integration (real model + real store, by request): run the diffuse query in all
three modes and report the top-k side by side + score ranges. Executed manually
(needs the model + built store), not a pytest.

## 6.1 Type-checking gate for dw_hist (FIX)

`dw_hist/` already ships `pyrightconfig.json` with `typeCheckingMode: strict`
and the cross-package `extraPaths`, but the pre-commit hook did NOT run pyright
on it (only `semantic_search/` and the providers were gated). Type regressions
in the CLI's hand-rolled numpy index math could land silently. Added a
`dw_hist/` block to `.githooks/pre-commit`, mirroring the `semantic_search/`
block: it triggers only when staged changes touch `dw_hist/`, runs
`cd dw_hist && python3 -m pyright dw_hist/` (so the package's own strict config +
extraPaths apply), and blocks the commit with an actionable message on any
error. Enforcement point for the "type checking is mandatory" requirement.

## 7. Out of scope (future, by request only)
* Persisting/caching μ/W in the store (kept fully recomputed → store read-only).
* Auto-selecting a transform; a tunable shrinkage estimator for Σ.
* Applying isotropy to the LanceDB `-q` path (separate, needs index-side work).
