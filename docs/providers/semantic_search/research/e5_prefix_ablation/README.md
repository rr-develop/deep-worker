# e5_prefix_ablation

Empirical study of whether adopting the official E5 `query:` / `passage:`
prefix contract measurably improves agent-history retrieval quality on our
own data.

- **Question, method, results, conclusion, sources:** see [REPORT.md](./REPORT.md).
- **Tools:**
  - `extract_pairs.py` — build (query, gold-passage) pairs from one or more
    `deepagents.summarization_history` JSONLs (user/TaskManager → final agent
    answer; inter-agent send → peer reply).
  - `run_ablation.py` — encode the corpus and queries twice (bare vs prefixed)
    with `intfloat/multilingual-e5-small`, compute Recall@1/5/10, MRR@10, and
    a paired Wilcoxon signed-rank test on per-query reciprocal rank.

Inputs and outputs are **not** committed: they are agent-specific data and
belong under `~/.deep_worker/<agent>/data/e5_prefix_ablation/`. The repo holds
only the protocol, tools, and the final report.

## Conclusion (in one line)

On our data with `multilingual-e5-small`, adding the prefixes yields no
measurable retrieval improvement (Recall@10 +0.77 pp, MRR@10 +0.11 pp,
Wilcoxon p = 0.86 over 518 pairs). Do **not** prioritize a re-index just for
this; see REPORT.md §7 for alternatives if retrieval quality is on the
critical path later.
