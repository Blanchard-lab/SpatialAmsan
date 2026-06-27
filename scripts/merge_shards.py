#!/usr/bin/env python3
"""
Merge multiple shard JSON outputs from a parallel eval run into one file.

Usage:
    python scripts/merge_shards.py \
        results/local/task1__videochat_r1_7b_s0.json \
        results/local/task1__videochat_r1_7b_s1.json \
        results/local/task1__videochat_r1_7b_s2.json \
        -o results/local/task1__videochat_r1_7b.json

Or with a glob (let the shell expand):
    python scripts/merge_shards.py results/local/task1__videochat_r1_7b_s*.json \
        -o results/local/task1__videochat_r1_7b.json

The shards must share the same meta (model_tag, hf_id). Results are concatenated
in the order shards are passed on the command line. Summary is recomputed across
all merged results.
"""

import argparse
import json
import sys
from pathlib import Path


def summarize(results):
    """Reproduces the summarize() logic from the eval scripts."""
    total_turns = 0
    scored_turns = 0
    q_totals = {}
    q_correct = {}

    for r in results:
        for t in r.get("turns", []):
            total_turns += 1
            s = t.get("score", {})
            if not s.get("scored"):
                continue
            scored_turns += 1
            for k in ["q1_correct", "q2_correct", "q3_correct", "q4_correct", "q4_2_correct"]:
                if k in s:
                    q_totals[k] = q_totals.get(k, 0) + 1
                    q_correct[k] = q_correct.get(k, 0) + (1 if s[k] else 0)

    per_q = {}
    for k in ["q1_correct", "q2_correct", "q3_correct", "q4_correct", "q4_2_correct"]:
        if k in q_totals and q_totals[k] > 0:
            per_q[k] = {
                "n": q_totals[k],
                "correct": q_correct[k],
                "accuracy": q_correct[k] / q_totals[k],
            }

    all_correct = sum(q_correct.get(k, 0) for k in q_totals)
    all_total = sum(q_totals.get(k, 0) for k in q_totals)

    return {
        "n_problems": len(results),
        "total_turns": total_turns,
        "scored_turns": scored_turns,
        "overall_accuracy": (all_correct / all_total) if all_total > 0 else None,
        "per_question": per_q,
    }


def main():
    ap = argparse.ArgumentParser(description="Merge parallel shard outputs.")
    ap.add_argument("shards", nargs="+",
                    help="Shard JSON files (pass in order of problem index)")
    ap.add_argument("-o", "--output", required=True,
                    help="Output JSON path")
    ap.add_argument("--allow_dup_pids", action="store_true",
                    help="By default, fail if any problem_id appears in more than one shard.")
    args = ap.parse_args()

    merged_results = []
    meta = None
    seen_pids = {}

    for shard_path in args.shards:
        p = Path(shard_path)
        if not p.exists():
            sys.exit(f"ERROR: shard not found: {p}")
        d = json.loads(p.read_text(encoding="utf-8"))
        shard_meta = d.get("meta", {})
        shard_results = d.get("results", [])

        if meta is None:
            meta = shard_meta
        else:
            if meta.get("model_tag") != shard_meta.get("model_tag"):
                sys.exit(f"ERROR: model_tag mismatch in {p} "
                         f"({shard_meta.get('model_tag')} vs {meta.get('model_tag')})")
            if meta.get("hf_id") != shard_meta.get("hf_id"):
                sys.exit(f"ERROR: hf_id mismatch in {p}")

        for r in shard_results:
            pid = r.get("problem_id")
            if pid in seen_pids and not args.allow_dup_pids:
                sys.exit(f"ERROR: duplicate problem_id '{pid}' in {p} "
                         f"(also in {seen_pids[pid]}). "
                         f"Pass --allow_dup_pids to merge anyway.")
            seen_pids[pid] = str(p)
            merged_results.append(r)

        print(f"  + {p.name}: {len(shard_results)} problems")

    summary = summarize(merged_results)
    out = {
        "meta": meta or {},
        "summary": summary,
        "results": merged_results,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nMerged {len(args.shards)} shards into {out_path}")
    print(f"  n_problems:     {summary['n_problems']}")
    print(f"  total_turns:    {summary['total_turns']}")
    print(f"  scored_turns:   {summary['scored_turns']}")
    print(f"  overall acc:    {summary['overall_accuracy']}")
    for k, v in summary["per_question"].items():
        print(f"  {k}: {v['correct']}/{v['n']} = {v['accuracy']:.4f}")


if __name__ == "__main__":
    main()
