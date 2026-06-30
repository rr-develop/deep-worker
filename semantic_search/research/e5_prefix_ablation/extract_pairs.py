"""Extract (query, gold-passage) pairs from a ``deepagents.summarization_history``
JSONL agent summary file.

Two pair types (see REPORT.md §2.2):
  T1 — user/TaskManager prompt -> final ai answer (excludes send_message_to_agent terminals).
  T2 — outbound send_message_to_agent -> next incoming "<peer> -> <self> :>" reply.

Outputs JSONL: pairs.jsonl, corpus.jsonl, manifest.json.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path


SEND_MSG_TOOL_NAMES = {
    "send_message_to_agent",
    "mcp__deepworker__send_message_to_agent",
}

MIN_LEN = 50
MAX_LEN = 8000


def _content_of(msg_obj: dict) -> str:
    data = msg_obj.get("message", {}).get("data", {})
    content = data.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # langchain occasionally emits list-of-blocks; pull text fields
        parts = []
        for blk in content:
            if isinstance(blk, dict):
                t = blk.get("text") or blk.get("content")
                if isinstance(t, str):
                    parts.append(t)
        return "\n".join(parts)
    return ""


def _tool_calls_of(msg_obj: dict) -> list[dict]:
    data = msg_obj.get("message", {}).get("data", {})
    tc = data.get("tool_calls")
    return tc if isinstance(tc, list) else []


def _is_send_to_agent_terminal(msg_obj: dict) -> bool:
    for tc in _tool_calls_of(msg_obj):
        if tc.get("name") in SEND_MSG_TOOL_NAMES:
            return True
    return False


def _accept_text(s: str) -> str | None:
    s = (s or "").strip()
    if len(s) < MIN_LEN:
        return None
    if len(s) > MAX_LEN:
        s = s[:MAX_LEN]
    return s


def _self_name_from_inter_agent_lines(messages: list[dict]) -> str | None:
    """Detect the self-name by scanning incoming inter-agent messages of form
    '<peer> -> <self> :>'. We look at the first few human messages that match.
    """
    for m in messages:
        if m.get("message_type") != "human":
            continue
        c = _content_of(m)
        # pattern: "<peer> -> <self> :>"
        idx = c.find(" -> ")
        if idx <= 0:
            continue
        rest = c[idx + 4 :]
        colon = rest.find(" :>")
        if colon <= 0:
            continue
        self_candidate = rest[:colon].strip()
        # plausible agent name: short, no whitespace
        if 1 <= len(self_candidate) <= 64 and " " not in self_candidate:
            return self_candidate
    return None


def extract(jsonl_path: Path) -> tuple[list[dict], list[dict], dict]:
    messages: list[dict] = []
    with jsonl_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("kind") != "deepagents.summarization_history.message":
                continue
            messages.append(obj)

    # Keep ordering by message_index
    messages.sort(key=lambda o: o.get("message_index", 0))

    self_name = _self_name_from_inter_agent_lines(messages) or "self"

    pairs: list[dict] = []
    pair_keys_seen: set[str] = set()

    # ---------- T1: user/TaskManager prompt -> final ai answer ----------
    n = len(messages)
    i = 0
    while i < n:
        m = messages[i]
        if m.get("message_type") != "human":
            i += 1
            continue
        # find the run [i+1 .. j-1] of non-human messages
        j = i + 1
        while j < n and messages[j].get("message_type") != "human":
            j += 1
        # last ai message in [i+1, j-1]
        last_ai = None
        for k in range(j - 1, i, -1):
            if messages[k].get("message_type") == "ai":
                last_ai = messages[k]
                break
        if last_ai is not None and not _is_send_to_agent_terminal(last_ai):
            q = _accept_text(_content_of(m))
            p = _accept_text(_content_of(last_ai))
            if q and p:
                key = "T1:" + q
                if key not in pair_keys_seen:
                    pair_keys_seen.add(key)
                    pairs.append({
                        "type": "T1",
                        "query": q,
                        "passage": p,
                        "query_idx": m.get("message_index"),
                        "passage_idx": last_ai.get("message_index"),
                        "meta": {
                            "is_task_manager": q.lstrip().startswith("TaskManager:>"),
                        },
                    })
        i = j

    # ---------- T2: outbound send_message_to_agent -> peer reply ----------
    # Build index of incoming peer messages: list of (msg_index, peer, text)
    incoming: list[tuple[int, str, str]] = []
    self_arrow = f" -> {self_name} :>"
    for m in messages:
        if m.get("message_type") != "human":
            continue
        c = _content_of(m)
        idx = c.find(self_arrow)
        if idx <= 0:
            continue
        peer = c[:idx].strip()
        if not peer or " " in peer:
            continue
        body = c[idx + len(self_arrow) :].strip()
        incoming.append((m.get("message_index", 0), peer, body))

    for m in messages:
        if m.get("message_type") != "ai":
            continue
        for tc in _tool_calls_of(m):
            if tc.get("name") not in SEND_MSG_TOOL_NAMES:
                continue
            args = tc.get("args") or {}
            target = args.get("agent_name")
            outbound = args.get("message")
            if not isinstance(target, str) or not isinstance(outbound, str):
                continue
            q = _accept_text(outbound)
            if not q:
                continue
            send_idx = m.get("message_index", 0)
            # find next incoming from this target
            reply = next(
                ((ridx, body) for (ridx, peer, body) in incoming
                 if peer == target and ridx > send_idx),
                None,
            )
            if reply is None:
                continue
            ridx, body = reply
            p = _accept_text(body)
            if not p:
                continue
            key = "T2:" + q
            if key in pair_keys_seen:
                continue
            pair_keys_seen.add(key)
            pairs.append({
                "type": "T2",
                "query": q,
                "passage": p,
                "query_idx": send_idx,
                "passage_idx": ridx,
                "meta": {"peer": target},
            })

    # ---------- corpus: every eligible ai content message (>=MIN_LEN) ----------
    corpus: list[dict] = []
    corpus_keys_seen: set[str] = set()
    # First, add every gold passage (guarantees they're in the haystack).
    for p in pairs:
        key = f"{p['type']}:{p['passage_idx']}"
        if key in corpus_keys_seen:
            continue
        corpus_keys_seen.add(key)
        corpus.append({
            "idx": p["passage_idx"],
            "text": p["passage"],
            "source": f"gold-{p['type']}",
        })
    # Then add all other ai-content distractors.
    for m in messages:
        if m.get("message_type") != "ai":
            continue
        t = _accept_text(_content_of(m))
        if not t:
            continue
        midx = m.get("message_index")
        key = f"ai:{midx}"
        if key in corpus_keys_seen:
            continue
        corpus_keys_seen.add(key)
        corpus.append({"idx": midx, "text": t, "source": "ai"})
    # Also add inter-agent incoming text (T2 gold passages already in via gold pass).
    for (ridx, peer, body) in incoming:
        t = _accept_text(body)
        if not t:
            continue
        key = f"inc:{ridx}"
        if key in corpus_keys_seen:
            continue
        corpus_keys_seen.add(key)
        corpus.append({"idx": ridx, "text": t, "source": f"incoming:{peer}"})

    manifest = {
        "source_file": str(jsonl_path),
        "messages_total": n,
        "self_name_detected": self_name,
        "pairs_total": len(pairs),
        "pairs_T1": sum(1 for p in pairs if p["type"] == "T1"),
        "pairs_T2": sum(1 for p in pairs if p["type"] == "T2"),
        "corpus_total": len(corpus),
        "min_len": MIN_LEN,
        "max_len": MAX_LEN,
    }
    return pairs, corpus, manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", action="append", required=True, type=Path,
                    help="Source JSONL file(s). May be passed multiple times.")
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    all_pairs: list[dict] = []
    all_corpus: list[dict] = []
    submanifests: list[dict] = []
    seen_keys: set[str] = set()
    corpus_seen: set[str] = set()

    for src in args.jsonl:
        pairs, corpus, manifest = extract(src)
        manifest["source_file_basename"] = src.name
        submanifests.append(manifest)
        # Namespace pair/corpus ids by agent to avoid idx-collision across files.
        agent = src.stem
        for p in pairs:
            key = f"{agent}::{p['type']}::{p['query'][:200]}"
            if key in seen_keys:
                continue
            seen_keys.add(key)
            p["agent"] = agent
            all_pairs.append(p)
        for c in corpus:
            key = f"{agent}::{c['idx']}"
            if key in corpus_seen:
                continue
            corpus_seen.add(key)
            c["agent"] = agent
            c["doc_id"] = f"{agent}::{c['idx']}"
            all_corpus.append(c)
        # also re-key gold passage references inside pairs to doc_id form
        for p in all_pairs:
            if p.get("agent") == agent and "passage_doc_id" not in p:
                p["passage_doc_id"] = f"{agent}::{p['passage_idx']}"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "pairs.jsonl").open("w", encoding="utf-8") as fh:
        for p in all_pairs:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    with (args.out_dir / "corpus.jsonl").open("w", encoding="utf-8") as fh:
        for c in all_corpus:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    overall = {
        "pairs_total": len(all_pairs),
        "pairs_T1": sum(1 for p in all_pairs if p["type"] == "T1"),
        "pairs_T2": sum(1 for p in all_pairs if p["type"] == "T2"),
        "corpus_total": len(all_corpus),
        "sources": submanifests,
    }
    with (args.out_dir / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(overall, fh, ensure_ascii=False, indent=2)

    print(json.dumps(overall, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
