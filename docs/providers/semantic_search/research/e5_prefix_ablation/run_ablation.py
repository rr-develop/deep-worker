"""E5 prefix ablation runner.

Builds two parallel indices over the same corpus:
  A: bare text on both query and document sides.
  B: "passage: " prefix on documents, "query: " prefix on queries.

Computes per-query rank of the gold passage, then Recall@1/5/10, MRR@10, and
Wilcoxon signed-rank test on per-query reciprocal-rank (capped at 10).

Embedding model is loaded once; encoding for A vs B differs only in input.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from transformers import AutoModel, AutoTokenizer


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def encode_batched(texts: list[str], tokenizer, model, *, batch_size: int, device: str, tag: str = "") -> np.ndarray:
    import time
    out = np.empty((len(texts), model.config.hidden_size), dtype=np.float32)
    model.eval()
    n_batches = (len(texts) + batch_size - 1) // batch_size
    t0 = time.time()
    last_print = t0
    with torch.inference_mode():
        for bi, start in enumerate(range(0, len(texts), batch_size)):
            chunk = texts[start:start + batch_size]
            batch = tokenizer(chunk, max_length=512, padding=True, truncation=True,
                              return_tensors="pt").to(device)
            outputs = model(**batch)
            emb = average_pool(outputs.last_hidden_state, batch["attention_mask"])
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            out[start:start + len(chunk)] = emb.cpu().numpy()
            now = time.time()
            if now - last_print >= 5.0 or bi == n_batches - 1:
                elapsed = now - t0
                rate = (start + len(chunk)) / elapsed if elapsed > 0 else 0
                eta = (len(texts) - (start + len(chunk))) / rate if rate > 0 else 0
                print(f"  [{tag}] batch {bi + 1}/{n_batches}  done={start + len(chunk)}/{len(texts)}  "
                      f"elapsed={elapsed:.0f}s  rate={rate:.1f}/s  eta={eta:.0f}s", flush=True)
                last_print = now
    return out


def metrics_from_ranks(ranks: list[int]) -> dict:
    n = len(ranks)
    if n == 0:
        return {"n": 0}
    r1 = sum(1 for r in ranks if r is not None and r <= 1) / n
    r5 = sum(1 for r in ranks if r is not None and r <= 5) / n
    r10 = sum(1 for r in ranks if r is not None and r <= 10) / n
    mrr10 = sum((1.0 / r) if (r is not None and r <= 10) else 0.0 for r in ranks) / n
    return {
        "n": n,
        "recall@1": r1,
        "recall@5": r5,
        "recall@10": r10,
        "mrr@10": mrr10,
    }


def wilcoxon_paired(rr_a: list[float], rr_b: list[float]) -> tuple[float, float, int, int, int]:
    """Two-sided Wilcoxon signed-rank on (b - a) reciprocal ranks.
    Returns: (W_statistic, p_value, n_b_better, n_tied, n_a_better).
    Uses scipy if available; otherwise falls back to a normal-approximation."""
    diffs = [b - a for a, b in zip(rr_a, rr_b)]
    n_b_better = sum(1 for d in diffs if d > 0)
    n_a_better = sum(1 for d in diffs if d < 0)
    n_tied = sum(1 for d in diffs if d == 0)
    try:
        from scipy.stats import wilcoxon
        nz = [d for d in diffs if d != 0]
        if not nz:
            return 0.0, 1.0, n_b_better, n_tied, n_a_better
        res = wilcoxon(nz, zero_method="wilcox", alternative="two-sided",
                       mode="auto")
        return float(res.statistic), float(res.pvalue), n_b_better, n_tied, n_a_better
    except Exception:
        # Normal approximation fallback (Mann 1947 style).
        nz = [d for d in diffs if d != 0]
        if not nz:
            return 0.0, 1.0, n_b_better, n_tied, n_a_better
        abs_d = sorted([(abs(d), 1 if d > 0 else -1) for d in nz])
        ranks = list(range(1, len(abs_d) + 1))
        # Average ranks for ties in |d|
        i = 0
        while i < len(abs_d):
            j = i + 1
            while j < len(abs_d) and abs_d[j][0] == abs_d[i][0]:
                j += 1
            avg = sum(ranks[i:j]) / (j - i)
            for k in range(i, j):
                ranks[k] = avg
            i = j
        w_plus = sum(r for r, (_, s) in zip(ranks, abs_d) if s > 0)
        n = len(nz)
        mean = n * (n + 1) / 4
        sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        z = (w_plus - mean) / sd if sd > 0 else 0.0
        # two-sided p
        from math import erf
        p = 2 * (1 - 0.5 * (1 + erf(abs(z) / math.sqrt(2))))
        return float(w_plus), float(p), n_b_better, n_tied, n_a_better


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="pairs.jsonl", type=Path)
    ap.add_argument("--corpus", default="corpus.jsonl", type=Path)
    ap.add_argument("--out-dir", default=".", type=Path)
    ap.add_argument("--model", default="intfloat/multilingual-e5-small")
    ap.add_argument("--batch-size", default=16, type=int)
    args = ap.parse_args()

    pairs = [json.loads(l) for l in args.pairs.read_text(encoding="utf-8").splitlines() if l.strip()]
    corpus = [json.loads(l) for l in args.corpus.read_text(encoding="utf-8").splitlines() if l.strip()]
    doc_ids = [c["doc_id"] for c in corpus]
    doc_id_to_idx = {d: i for i, d in enumerate(doc_ids)}

    # Sanity: every gold pair must be findable in the haystack.
    missing = [p for p in pairs if p["passage_doc_id"] not in doc_id_to_idx]
    if missing:
        raise SystemExit(f"FATAL: {len(missing)} gold pair(s) not in corpus")

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(args.model).to(device)
    print(f"Device: {device}  hidden_size={model.config.hidden_size}")

    queries = [p["query"] for p in pairs]
    docs = [c["text"] for c in corpus]
    print(f"Pairs: {len(queries)}    Corpus: {len(docs)}")

    cache_dir = args.out_dir / "_emb_cache"
    cache_dir.mkdir(exist_ok=True)

    def cache_or_encode(name: str, texts: list[str]) -> np.ndarray:
        path = cache_dir / f"{name}.npy"
        if path.exists():
            arr = np.load(path)
            if arr.shape[0] == len(texts):
                print(f"[cache] {name}: loaded {arr.shape}", flush=True)
                return arr
        print(f"[encode] {name}: {len(texts)} texts", flush=True)
        arr = encode_batched(texts, tokenizer, model, batch_size=args.batch_size,
                             device=device, tag=name)
        np.save(path, arr)
        return arr

    DA = cache_or_encode("docs_bare", docs)
    QA = cache_or_encode("queries_bare", queries)
    DB = cache_or_encode("docs_passage", [f"passage: {t}" for t in docs])
    QB = cache_or_encode("queries_query", [f"query: {t}" for t in queries])

    def rank_for(Q: np.ndarray, D: np.ndarray) -> list[int]:
        # similarity matrix
        S = Q @ D.T  # (n_queries, n_docs)
        ranks: list[int] = []
        for i, p in enumerate(pairs):
            gold_idx = doc_id_to_idx[p["passage_doc_id"]]
            scores = S[i]
            # rank = 1 + number of docs strictly better than gold
            gold_score = scores[gold_idx]
            better = int(np.sum(scores > gold_score))
            ranks.append(better + 1)
        return ranks

    print("Computing ranks A...")
    ranks_a = rank_for(QA, DA)
    print("Computing ranks B...")
    ranks_b = rank_for(QB, DB)

    metr_a = metrics_from_ranks(ranks_a)
    metr_b = metrics_from_ranks(ranks_b)

    rr_a = [1.0 / r if r <= 10 else 0.0 for r in ranks_a]
    rr_b = [1.0 / r if r <= 10 else 0.0 for r in ranks_b]
    W, p_value, n_b_better, n_tied, n_a_better = wilcoxon_paired(rr_a, rr_b)

    # Per-type breakdown
    def by_type(t: str) -> dict:
        idx = [i for i, pr in enumerate(pairs) if pr["type"] == t]
        return {
            "n": len(idx),
            "A": metrics_from_ranks([ranks_a[i] for i in idx]),
            "B": metrics_from_ranks([ranks_b[i] for i in idx]),
        }

    results = {
        "model": args.model,
        "pairs_total": len(pairs),
        "corpus_total": len(corpus),
        "A_baseline_bare": metr_a,
        "B_prefixed": metr_b,
        "wilcoxon_two_sided_on_reciprocal_rank": {
            "W": W,
            "p_value": p_value,
            "n_B_better": n_b_better,
            "n_tied": n_tied,
            "n_A_better": n_a_better,
        },
        "by_pair_type": {
            "T1": by_type("T1"),
            "T2": by_type("T2"),
        },
    }
    out_path = args.out_dir / "results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))

    # Also dump per-query ranks for follow-up analysis.
    per_q = []
    for i, p in enumerate(pairs):
        per_q.append({
            "agent": p.get("agent"),
            "type": p["type"],
            "rank_A": ranks_a[i],
            "rank_B": ranks_b[i],
            "query_idx": p.get("query_idx"),
        })
    (args.out_dir / "per_query_ranks.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in per_q) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
