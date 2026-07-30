"""Exact dot-product search over the per-message embedding store.

A read-only query layer on top of the persisted per-message embedding store
(:mod:`semantic_search.message_embedding_store`). It complements the chunked
LanceDB history search (:func:`semantic_search.history_retriever.search_history`):
here every *single message* has one L2-normalised embedding row, so we can rank
messages directly by the query's projection (dot product) onto each row.

Because both the stored rows and the query embedding are unit-norm, the dot
product equals cosine similarity — the requested "projection of the query onto
each embedding". Ranking is an exact full inner product (no ANN): ``N`` is at
most a few thousand rows per agent, so a single vectorised matmul plus a top-k
``argpartition`` is effectively instant.

This module **never** recomputes the stored embeddings and never writes to the
store. Only ``numpy`` is needed for the ranking (no ``torch``); the query is
embedded once via :func:`semantic_search.long_text_embedding.embed_long_text`,
which loads the same embedder the store was built with.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import IndexConfig
from .long_text_embedding import embed_long_text, message_text
from .message_embedding_store import (
    flag_names as _flag_names,
)
from .message_embedding_store import (
    load_message_embeddings,
    load_message_metadata,
    message_embeddings_exist,
    message_metadata_exist,
    store_dir,
    store_file_entry_counts,
)


@dataclass
class MessageSearchResult:
    """A single per-message search hit (mirrors ``HistorySearchResult`` fields,
    plus the row's :class:`~semantic_search.message_embedding_store.MessageFlag`
    bitmask)."""

    content: str
    """Role-prefixed message text (same extraction used to build the store)."""

    score: float
    """Query·row dot product == cosine similarity (both operands unit-norm).

    In ``[-1, 1]``; for this corpus effectively ``[0, 1]``. This IS the
    requested projection of the (normalised) query onto the (normalised) row.
    """

    thread_id: str
    """Thread ID (stem of the source JSONL filename)."""

    line_range: str
    """PHYSICAL 1-based line number of the message's entry in its JSONL file."""

    timestamp: str
    """ISO-8601 ``summarized_at`` of the message entry (``""`` if absent)."""

    message_types: str
    """The entry's ``message_type`` (``human``/``ai``/``tool``/``system``/…)."""

    file_path: str
    """Absolute path to the source JSONL file."""

    flags: int
    """:class:`MessageFlag` bitmask for this row (from ``flags.npy``)."""

    flag_names: list[str] = field(default_factory=lambda: [])
    """Decoded set-flag names, in canonical bit order."""


@dataclass
class _RowMessage:
    """Positional source data for one embedding row."""

    content: str
    thread_id: str
    line: int
    timestamp: str
    message_type: str
    file_path: str


def _row_messages(
    summaries_dir: Path, watermarks: dict[str, int] | None = None
) -> list[_RowMessage]:
    """Re-derive per-row source-message data, 1-to-1 with the store rows.

    The store's row order is *file-sorted, then embeddable-entry order* — exactly
    what :func:`build_message_metadata` produces. We reproduce that selection and
    order here to map a row index back to its source message for display.

    RATIONALE (do not "simplify" toward a cross-file flatten): the canonical
    metadata build flattens the raw entry stream ACROSS files only so
    ``classify_message`` gets a cross-file look-ahead for ``AGENT_FINAL``. This
    function needs only POSITIONAL data (file / physical line / text / type /
    timestamp) and takes flags from ``flags.npy`` (never recomputed here), so a
    per-file walk is positionally equivalent and is used deliberately.

    RC-1 (physical line numbers): the shared ``_read_jsonl`` silently drops blank
    and undecodable lines and returns a compacted list, so a compacted-list index
    is NOT the physical line number. We therefore walk ``splitlines()`` ourselves
    with a physical 1-based counter, applying the SAME skip rules (blank / JSON
    decode error) so the *selection and order* stay identical to the store, and
    emit the physical counter as the line.

    WATERMARK (append-only history): ``watermarks`` maps each source file's
    absolute path to the number of *parsed* entries consumed when the store was
    built (``manifest["file_entry_counts"]``, same unit as
    :func:`count_jsonl_lines`). History files grow after the build
    (summarisation appends new turns), so re-deriving over the *current* file
    would produce more rows than the frozen store and desynchronise the mapping.
    We therefore count parsed entries per file and STOP at that file's watermark,
    yielding exactly the prefix the store rows came from; later-appended entries
    are ignored. When ``watermarks`` is ``None`` or a file is absent from it
    (e.g. a store built before watermark tracking), that file is walked in full
    — preserving the previous behaviour for legacy stores.
    """
    out: list[_RowMessage] = []
    for fp in sorted(summaries_dir.glob("*.jsonl")):
        try:
            text = fp.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        stem = fp.stem
        abspath = str(fp.resolve())
        # Watermark is keyed by the path the build recorded (str(fp), which is
        # the glob result — not resolve()'d). Match on that, fall back to the
        # resolved path, else no limit (legacy store / untracked file).
        limit = None
        if watermarks is not None:
            wm = watermarks.get(str(fp))
            if wm is None:
                wm = watermarks.get(abspath)
            limit = wm
        physical_line = 0
        parsed_index = 0  # count of parsed entries — the watermark's unit
        for raw_line in text.splitlines():
            physical_line += 1
            s = raw_line.strip()
            if not s:
                continue  # blank — skipped, mirrors _read_jsonl
            try:
                entry: dict[str, Any] = json.loads(s)
            except json.JSONDecodeError:
                continue  # undecodable — skipped, mirrors _read_jsonl
            # This line is a parsed entry. Stop once we've consumed the store's
            # watermark for this file — anything beyond was appended later.
            if limit is not None and parsed_index >= limit:
                break
            parsed_index += 1
            msg_text = message_text(entry)
            if not msg_text:
                continue  # empty content — no embedding row
            out.append(_RowMessage(
                content=msg_text,
                thread_id=stem,
                line=physical_line,
                timestamp=str(entry.get("summarized_at") or ""),
                message_type=str(entry.get("message_type") or "unknown"),
                file_path=abspath,
            ))
    return out


Transform = str  # "raw" | "centering" | "whitening"
WhiteningMethod = str  # "pca" | "zca"
StatsScope = str  # "corpus" | "filtered"

_VALID_TRANSFORMS: tuple[str, ...] = ("raw", "centering", "whitening")
_VALID_WHITENING_METHODS: tuple[str, ...] = ("pca", "zca")
_VALID_STATS_SCOPES: tuple[str, ...] = ("corpus", "filtered")


def _l2_normalise(x: Any) -> Any:
    """L2-normalise the last axis; zero-norm rows/vectors return zero (RC-1).

    A zero-vector row would otherwise produce 0/0 = NaN and silently corrupt
    argpartition / sort ordering downstream. This mirrors the empty-text guard
    in :func:`long_text_embedding.embed_long_text`.
    """
    import numpy as _np  # noqa: PLC0415

    np: Any = _np
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    # np.where broadcasts a 0-scalar over the zero-norm rows so the result stays
    # finite (no /0). float64/32 preserved from the input.
    safe = np.where(norms > 0.0, x / np.maximum(norms, 1e-12), 0.0)
    return safe.astype(x.dtype, copy=False)


def _select_kept_axis_count(
    lam_ascending: Any,
    dim: int,
    whitening_var: float,
    whitening_dims: int | None,
    usable_rank: int,
) -> int:
    """Choose ``r`` = how many top-variance eigen-axes to keep (§3.6).

    ``lam_ascending`` is the eigenvalue vector as ``np.linalg.eigh`` returns it —
    **ascending**, so the LARGEST (highest-variance) eigenvalues are the LAST
    entries. Truncation keeps the ``r`` largest, i.e. the tail of this array
    (RC-4: never the head, which is the near-zero noise floor).

    ``r`` is bounded by ``r_max = min(dim, usable_rank)`` where ``usable_rank``
    reflects how many axes the statistics actually support (a Σ estimated from
    ``m`` rows has rank ≤ ``m-1``). The returned ``r`` is the ACTUAL kept count
    after clamping (RC-5), so callers can report it truthfully.

    * ``whitening_dims`` (absolute) takes precedence when not ``None``:
      ``r = clamp(whitening_dims, 1, r_max)``.
    * else ``whitening_var = F``: smallest ``r`` whose cumulative variance ratio
      (from the largest eigenvalues down) ≥ ``F``; clamped to ``[1, r_max]``.
    """
    import numpy as _np  # noqa: PLC0415

    np: Any = _np
    r_max = max(1, min(dim, usable_rank))
    if whitening_dims is not None:
        return int(max(1, min(int(whitening_dims), r_max)))
    # Variance-fraction selection over the DESCENDING spectrum (largest first).
    lam_desc = np.clip(lam_ascending[::-1].astype(np.float64), 0.0, None)
    total = float(lam_desc.sum())
    if total <= 0.0:  # degenerate: no variance -> keep a single axis.
        return 1
    cum_ratio = np.cumsum(lam_desc) / total
    # smallest r with cum_ratio[r-1] >= F; searchsorted on the monotone cumulative
    # gives the first index reaching F. +1 converts 0-based index to a count.
    r = int(np.searchsorted(cum_ratio, float(whitening_var)) + 1)
    return int(max(1, min(r, r_max)))


def _compute_whitening_matrix(
    stats_rows: Any,
    method: str,
    eps: float,
    dim: int,
    whitening_var: float = 1.0,
    whitening_dims: int | None = None,
) -> tuple[Any, Any, int]:
    """Return ``(mu, W, r_kept)`` for whitening: ``x' = normalize((x - mu) @ W)``.

    ``method='pca'`` gives ``W = U_r diag((λ_r+ε)^{-1/2})`` (rotates into PCA
    axes); ``method='zca'`` gives ``W = U_r diag((λ_r+ε)^{-1/2}) U_r^T``
    (symmetric, keeps original axes; the ZCA default). For cosine ranking the two
    produce the same ordering on the kept ``r``-dim subspace (cosine is invariant
    to the shared orthogonal factor ``U_r^T``) — see §3.1 of
    DESIGN_DW_HIST_QM_ISOTROPY.md.

    TRUNCATION (§3.6): eigh is run on the RAW covariance (no εI beforehand), so
    the eigenvalues are the true per-axis variances and axis selection ranks by
    real variance. Only the top ``r`` axes (largest λ) are kept; the near-zero
    noise tail is discarded rather than amplified by ``λ^{-1/2}``. ε is added to
    the KEPT block (``λ_r + ε``) for numerical safety. ``r`` comes from
    :func:`_select_kept_axis_count` and the actual kept count (after clamping to
    the usable rank, RC-5) is returned as ``r_kept``.
    """
    import numpy as _np  # noqa: PLC0415

    np: Any = _np
    x = stats_rows.astype(np.float64, copy=False)
    m = int(x.shape[0])
    mu = x.mean(axis=0)
    centered = x - mu
    # cov: (dim, dim); rowvar=False treats rows as observations. RAW (no εI):
    # eigenvalues must be the true variances so truncation ranks correctly.
    sigma = np.cov(centered, rowvar=False)
    # eigh returns ASCENDING eigenvalues + orthonormal eigenvectors (Σ is
    # symmetric PSD). The largest λ are therefore the LAST entries.
    lam, u = np.linalg.eigh(sigma)
    # Σ from m observations has rank ≤ m-1; don't keep more axes than supported.
    usable_rank = m - 1
    r = _select_kept_axis_count(lam, dim, whitening_var, whitening_dims, usable_rank)
    # Keep the top-r (largest-variance) axes = the LAST r of the ascending output.
    lam_r = lam[-r:]
    u_r = u[:, -r:]
    inv_sqrt = 1.0 / np.sqrt(lam_r + eps)
    if method == "pca":
        w = u_r * inv_sqrt  # (dim, r): equivalent to U_r @ diag(inv_sqrt)
    elif method == "zca":
        w = (u_r * inv_sqrt) @ u_r.T  # (dim, dim)
    else:  # pragma: no cover — validated by the caller
        raise ValueError(f"unknown whitening method: {method!r}")
    return mu, w, r


def search_messages(
    *,
    query: str,
    summaries_dir: str | Path,
    top_k: int,
    flags_any: int = 0,
    transform: Transform = "raw",
    whitening_method: WhiteningMethod = "zca",
    whitening_eps: float = 1e-3,
    whitening_var: float = 0.90,
    whitening_dims: int | None = None,
    stats_scope: StatsScope = "corpus",
    stats_out: dict[str, Any] | None = None,
    embed_model: Any | None = None,
    config: IndexConfig | None = None,
) -> list[MessageSearchResult]:
    """Rank stored per-message embeddings by the query's projection (dot product).

    Args:
        query: search text. Must be non-empty / non-whitespace — an empty query
            would embed to a zero vector and make every score 0.0 (the caller is
            expected to reject empty queries; we also guard here defensively).
        summaries_dir: the agent's summaries directory (parent of
            ``.message_embeddings``).
        top_k: maximum number of hits to return (clamped to the candidate count).
        flags_any: if non-zero, only rows whose flag bitmask shares at least one
            bit with this mask are scored (logical OR over the requested flags).
        transform: ranking-space transform (``"raw"`` default = current
            behaviour, ``"centering"`` subtracts corpus mean, ``"whitening"``
            additionally equalises per-axis variance). See
            DESIGN_DW_HIST_QM_ISOTROPY.md for the math.
        whitening_method: ``"pca"`` or ``"zca"`` (default ``"zca"``). Only used
            when ``transform="whitening"``. PCA and ZCA yield identical cosine
            rankings; the flag is preserved for future experiments.
        whitening_eps: regularisation ε added to the KEPT eigenvalue block
            (``λ_r + ε``) for numerical safety. Must be > 0. Only used when
            ``transform="whitening"``.
        whitening_var: truncation target — keep the top principal axes covering
            this fraction of total variance (default ``0.90``), dropping the
            noisy low-variance tail (§3.6). Must be in ``(0, 1]``; ``1.0`` keeps
            all usable axes (= full whitening). Ignored when ``whitening_dims``
            is given. Only used when ``transform="whitening"``.
        whitening_dims: absolute number of top axes to keep (overrides
            ``whitening_var`` when not ``None``). Clamped to ``[1, r_max]`` where
            ``r_max`` is the usable rank. Only used when
            ``transform="whitening"``.
        stats_scope: rows used to estimate μ (and Σ for whitening).
            ``"corpus"`` (default) → all N store rows; ``"filtered"`` → the
            ``flags_any`` candidate subset (isotropise relative to the filter).
        stats_out: optional dict; when passed, the function fills it with
            diagnostic info about ALL M candidate scores (not the top-k slice):
            ``{"m": int, "min": float, "median": float, "max": float,
            "range": float, "transform": str}``. For whitening it ALSO records
            truncation info (populated even without ``--score-stats`` so the CLI
            can report the kept-axis count): ``"whitening_dims_kept": int|None``
            (actual axes kept after clamping, RC-5; ``None`` when no truncation
            happened, i.e. ``r == dim``), ``"whitening_dim": int`` (the full
            ``dim``), ``"whitening_var": float|None`` and
            ``"whitening_dims_requested": int|None`` (what the caller asked for).
            Used by ``dw-hist --score-stats`` and the kept-axis notice.
        embed_model / config: optional embedder / config overrides (testing).

    Returns the top hits sorted by descending score, ties broken by ascending
    row index. Returns ``[]`` when the candidate set is empty.

    Raises:
        ValueError: empty / whitespace-only query, or invalid enum-like arg.
        FileNotFoundError: the embedding store or its metadata is absent.
        RuntimeError: store/metadata/history row counts disagree (rebuild), or
            whitening asked for on too few stats rows (message is actionable —
            widen the filter / ``--stats-scope corpus`` / drop ``-w``).
    """
    import numpy as _np  # noqa: PLC0415

    np: Any = _np

    if not query or not query.strip():
        raise ValueError("query must be a non-empty, non-whitespace string")
    if transform not in _VALID_TRANSFORMS:
        raise ValueError(
            f"transform must be one of {_VALID_TRANSFORMS}, got {transform!r}"
        )
    if whitening_method not in _VALID_WHITENING_METHODS:
        raise ValueError(
            f"whitening_method must be one of {_VALID_WHITENING_METHODS}, "
            f"got {whitening_method!r}"
        )
    if stats_scope not in _VALID_STATS_SCOPES:
        raise ValueError(
            f"stats_scope must be one of {_VALID_STATS_SCOPES}, got {stats_scope!r}"
        )
    if transform == "whitening" and not (whitening_eps > 0.0):
        raise ValueError(
            f"whitening_eps must be > 0, got {whitening_eps!r}"
        )
    if transform == "whitening" and not (0.0 < whitening_var <= 1.0):
        raise ValueError(
            f"whitening_var must be in (0, 1], got {whitening_var!r}"
        )
    if (
        transform == "whitening"
        and whitening_dims is not None
        and int(whitening_dims) < 1
    ):
        raise ValueError(
            f"whitening_dims must be >= 1, got {whitening_dims!r}"
        )

    summaries_dir = Path(summaries_dir)
    if not message_embeddings_exist(summaries_dir) or not message_metadata_exist(
        summaries_dir
    ):
        raise FileNotFoundError(
            f"No per-message embedding store with metadata at "
            f"{store_dir(summaries_dir)}. Build it with "
            "build_message_embeddings() and build_message_metadata()."
        )

    embs, _lens = load_message_embeddings(summaries_dir, as_torch=False)
    _idx, flags = load_message_metadata(summaries_dir, as_torch=False)
    n = int(embs.shape[0])
    if int(flags.shape[0]) != n:
        raise RuntimeError(
            f"store row count ({n}) != flags row count ({int(flags.shape[0])}); "
            "rebuild metadata with build_message_metadata()."
        )

    # Candidate selection (flag OR filter). flatnonzero keeps original row order.
    if flags_any:
        cand_rows = np.flatnonzero((flags & flags_any) != 0)
    else:
        cand_rows = np.arange(n)
    if cand_rows.shape[0] == 0:
        return []

    cand_emb = embs[cand_rows]  # (M, dim)

    # Embed the query once (normalised) — the only model work. Same vector space
    # as the store (same embedder, no e5 query: prefix, matching how rows were
    # embedded).
    q = np.asarray(embed_long_text(query, embed_model, config), dtype=np.float32)

    # --- Apply the isotropy transform (if any) to BOTH sides with the SAME
    # μ / W — otherwise candidates and query live in different spaces and the
    # dot product is meaningless. §3.3 zero-norm guard is inside _l2_normalise.
    # Actual truncation outcome for whitening (populated in the whitening branch,
    # read back into stats_out below). None for non-whitening transforms.
    whitening_kept: int | None = None
    dim = int(embs.shape[1])
    if transform == "raw":
        # Both sides are already L2-normalised; nothing to do.
        cand_t = cand_emb.astype(np.float32, copy=False)
        q_t = q
    else:
        stats_rows_np = (
            cand_emb if stats_scope == "filtered" else embs
        ).astype(np.float64, copy=False)
        if transform == "centering":
            mu = stats_rows_np.mean(axis=0)
            cand_t = _l2_normalise(cand_emb.astype(np.float64) - mu).astype(np.float32)
            q_t = _l2_normalise(q.astype(np.float64) - mu).astype(np.float32)
        else:  # transform == "whitening"
            if int(stats_rows_np.shape[0]) < dim + 1:
                raise RuntimeError(
                    f"whitening needs at least {dim + 1} stats rows to estimate "
                    f"the covariance, got {int(stats_rows_np.shape[0])} "
                    f"(stats_scope={stats_scope!r}). Remedies: widen the -f "
                    "filter, use --stats-scope corpus, or drop -w."
                )
            mu, w_mat, whitening_kept = _compute_whitening_matrix(
                stats_rows_np, whitening_method, whitening_eps, dim,
                whitening_var, whitening_dims,
            )
            cand_t = _l2_normalise(
                (cand_emb.astype(np.float64) - mu) @ w_mat
            ).astype(np.float32)
            q_t = _l2_normalise(
                (q.astype(np.float64) - mu) @ w_mat
            ).astype(np.float32)

    scores = cand_t @ q_t  # (M,) dot products == cosine in the transformed space

    if stats_out is not None:
        # Score distribution across ALL M candidates (not the top-k slice).
        # NIT: keeps --score-stats a useful diagnostic of the whole distribution.
        if int(scores.shape[0]) > 0:
            s_min = float(scores.min())
            s_max = float(scores.max())
            s_med = float(np.median(scores))
        else:  # pragma: no cover — cand_rows.shape[0] == 0 returned earlier
            s_min = s_max = s_med = 0.0
        stats_out.update({
            "m": int(scores.shape[0]),
            "min": s_min,
            "median": s_med,
            "max": s_max,
            "range": s_max - s_min,
            "transform": transform,
        })

    # Truncation report (RC-5): record the ACTUAL kept-axis count so the CLI can
    # print the kept-axis notice even without --score-stats. Populated only for
    # whitening; "whitening_dims_kept" is None when no truncation happened
    # (r == dim), so the CLI can suppress the notice for full whitening.
    if stats_out is not None and transform == "whitening":
        stats_out.update({
            "whitening_dim": dim,
            "whitening_dims_kept": (
                whitening_kept if (whitening_kept is not None and whitening_kept < dim) else None
            ),
            "whitening_var": (None if whitening_dims is not None else float(whitening_var)),
            "whitening_dims_requested": (
                int(whitening_dims) if whitening_dims is not None else None
            ),
        })

    # Top-k without a full sort. k_eff guards argpartition against k > M.
    m = int(scores.shape[0])
    k_eff = min(top_k, m)
    if k_eff <= 0:
        return []
    part = np.argpartition(scores, -k_eff)[-k_eff:]
    # Sort the k_eff candidates by score desc, tie-break by ascending row index.
    sel_rows = cand_rows[part]
    order = sorted(
        range(k_eff),
        key=lambda j: (-float(scores[part[j]]), int(sel_rows[j])),
    )

    # Re-derive row source data over the SAME prefix the store was built from,
    # using the manifest watermark. Without this, history appended after the
    # build (summarisation adds turns) would make the re-derived count exceed
    # the frozen store rows and spuriously fail the 1-to-1 check below —
    # later messages must never break analysis of earlier ones.
    watermarks = store_file_entry_counts(summaries_dir)
    rows = _row_messages(summaries_dir, watermarks or None)
    if len(rows) != n:
        raise RuntimeError(
            f"re-derived message count ({len(rows)}) != store rows ({n}) even "
            "after applying the store watermark; the history files were "
            "rewritten (not merely appended) since the store was built — "
            "rebuild it with build_message_embeddings(reset=True)."
        )

    results: list[MessageSearchResult] = []
    for j in order:
        row = int(sel_rows[j])
        rm = rows[row]
        fl = int(flags[row])
        results.append(MessageSearchResult(
            content=rm.content,
            score=float(scores[part[j]]),
            thread_id=rm.thread_id,
            line_range=str(rm.line),
            timestamp=rm.timestamp,
            message_types=rm.message_type,
            file_path=rm.file_path,
            flags=fl,
            flag_names=_flag_names(fl),
        ))
    return results
