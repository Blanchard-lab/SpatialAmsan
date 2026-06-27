#!/usr/bin/env python3
"""
Task 1 (Tangram) multi-turn evaluation via local HuggingFace models.
Turn 0: header + goal image + block reference image (no questions).
Turn 1+: one manipulation image + questions, expecting JSON response.
"""

import argparse
import json
import re
import time
import gc
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List

import torch
from PIL import Image

from transformers import AutoProcessor, AutoModelForImageTextToText


LOCAL_MODELS = [
    ("qwen2p5_vl_7b", "Qwen/Qwen2.5-VL-7B-Instruct"),
    ("qwen3_vl_8b", "Qwen/Qwen3-VL-8B-Instruct"),
    ("videochat_r1_7b", "OpenGVLab/VideoChat-R1_7B"),
    ("internvl3p5_8b_hf", "OpenGVLab/InternVL3_5-8B-hf"),
    ("llama4_scout_17b16e", "meta-llama/Llama-4-Scout-17B-16E-Instruct"),
    ("internvl3_8b_hf", "OpenGVLab/InternVL3-8B-hf"),
    ("llava_onevision_qwen2_7b_ov_hf", "llava-hf/llava-onevision-qwen2-7b-ov-hf"),
]


def try_parse_json(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    try:
        return {"ok": True, "data": json.loads(raw), "raw": raw}
    except Exception:
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if m:
            try:
                return {"ok": True, "data": json.loads(m.group(0)), "raw": raw}
            except Exception:
                pass
    return {"ok": False, "data": None, "raw": raw}


def clear_mem():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def fmt_hms(seconds: float) -> str:
    """Format a duration in seconds as HH:MM:SS (zero-padded)."""
    s = max(0, int(seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def get_model_entry(tag: str):
    if tag == "all":
        return LOCAL_MODELS
    for t, hf in LOCAL_MODELS:
        if t == tag:
            return [(t, hf)]
    raise ValueError(f"Unknown model tag: {tag}")


# ── prompt builders ──────────────────────────────────────────────────────────

def build_turn0_prompt(header: Dict[str, Any]) -> str:
    return (
        header["problem_setup"]
        + "\n\nThe first image is the Goal State. "
        "The second image is the Block Reference showing all 7 tangram blocks with their colors.\n"
        "In each following turn, you will receive one image showing the current state "
        "after a manipulation. Answer the questions for each turn in JSON format."
    )


def build_turn_prompt(turn_num: int, is_turn1: bool,
                      q1_opts: List[Dict], q2_opts: List[Dict],
                      q3_opts: List[Dict], q4_opts: List[Dict]) -> str:
    q1_str = ", ".join(f"{o['id']}={o['text']}" for o in q1_opts)
    q2_str = ", ".join(f"{o['id']}={o['text']}" for o in q2_opts)
    q3_str = ", ".join(f"{o['id']}={o['text']}" for o in q3_opts)

    lines = [f"Here is Turn {turn_num}. Answer the following questions in JSON format.\n"]

    lines.append("Q1. Manipulated Block: Which block was manipulated in this turn? (Select one)")
    lines.append(f"Options: {q1_str}\n")

    if not is_turn1:
        lines.append("Q2. Latest Manipulated Block: Which block was the latest one manipulated before this turn? (Select one)")
        lines.append(f"Options: {q2_str}\n")

    lines.append("Q3. Contact Blocks: Which block(s) are in contact with the manipulated block? "
                 "Only count side-face-to-side-face contact. Edge-to-edge and edge-to-face contact do not count. (Select all that apply)")
    lines.append(f"Options: {q3_str}\n")

    lines.append("Q4. Evaluation of Current Manipulation: (Select all that apply)")
    for o in q4_opts:
        lines.append(f"{o['id']}. {o['text']}")
    lines.append("")

    lines.append("Q4-1. Reasoning: Briefly explain the reason for your answer in Q4.\n")
    lines.append("Q4-2. Undo Feasibility (answer only if Q4 includes 4, 5, 6, or 7): "
                 "Can the goal state still be reached by undoing the current manipulation? (\"Yes\" or \"No\")\n")
    lines.append("Q4-2-1. Reasoning (answer only if Q4-2 is answered): "
                 "Explain the reason for your answer in Q4-2.\n")

    if is_turn1:
        lines.append("Respond with JSON only:")
        lines.append('{"q1": <int>, "q3": [<int>, ...], "q4": [<int>, ...], '
                     '"q4_1": "<string>", "q4_2": "<string or null>", "q4_2_1": "<string or null>"}')
    else:
        lines.append("Respond with JSON only:")
        lines.append('{"q1": <int>, "q2": <int>, "q3": [<int>, ...], "q4": [<int>, ...], '
                     '"q4_1": "<string>", "q4_2": "<string or null>", "q4_2_1": "<string or null>"}')

    return "\n".join(lines)


# ── scoring ──────────────────────────────────────────────────────────────────

def score_turn(parsed, gt, is_turn1):
    if not parsed.get("ok") or not isinstance(parsed.get("data"), dict):
        return {"scored": False}

    data = parsed["data"]
    scores = {}

    gt_q1 = gt.get("q1")
    pred_q1 = data.get("q1")
    if gt_q1 is not None and pred_q1 is not None:
        scores["q1_correct"] = (int(pred_q1) == int(gt_q1))

    if not is_turn1:
        gt_q2 = gt.get("q2")
        pred_q2 = data.get("q2")
        if gt_q2 is not None and pred_q2 is not None:
            scores["q2_correct"] = (int(pred_q2) == int(gt_q2))

    gt_q3 = gt.get("q3")
    pred_q3 = data.get("q3")
    if gt_q3 is not None and pred_q3 is not None:
        try:
            gt_set = sorted(set(int(v) for v in gt_q3))
            pred_set = sorted(set(int(v) for v in (pred_q3 if isinstance(pred_q3, list) else [])))
            scores["q3_correct"] = (gt_set == pred_set)
        except Exception:
            scores["q3_correct"] = False

    gt_q4 = gt.get("q4")
    pred_q4 = data.get("q4")
    if gt_q4 is not None and pred_q4 is not None:
        try:
            gt_set = sorted(set(int(v) for v in gt_q4))
            pred_set = sorted(set(int(v) for v in (pred_q4 if isinstance(pred_q4, list) else [])))
            scores["q4_correct"] = (gt_set == pred_set)
        except Exception:
            scores["q4_correct"] = False

    gt_q4_2 = gt.get("q4_2")
    pred_q4_2 = data.get("q4_2")
    if gt_q4_2 is not None and pred_q4_2 is not None:
        scores["q4_2_correct"] = (str(pred_q4_2).strip().lower() == str(gt_q4_2).strip().lower())

    scored_vals = [v for v in scores.values() if isinstance(v, bool)]
    scores["n_scored"] = len(scored_vals)
    scores["n_correct"] = sum(scored_vals)
    scores["accuracy"] = (scores["n_correct"] / scores["n_scored"]) if scores["n_scored"] > 0 else None
    scores["scored"] = True
    return scores


def summarize(results):
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
            per_q[k] = {"n": q_totals[k], "correct": q_correct[k],
                        "accuracy": q_correct[k] / q_totals[k]}

    all_correct = sum(q_correct.get(k, 0) for k in q_totals)
    all_total = sum(q_totals.get(k, 0) for k in q_totals)

    return {
        "n_problems": len(results),
        "total_turns": total_turns,
        "scored_turns": scored_turns,
        "overall_accuracy": (all_correct / all_total) if all_total > 0 else None,
        "per_question": per_q,
    }


# ── local model inference ────────────────────────────────────────────────────

@torch.inference_mode()
def load_model(hf_id, device_map, dtype):
    """Load model and processor once, reuse across problems."""
    proc = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        hf_id,
        trust_remote_code=True,
        device_map=device_map,
        torch_dtype=(torch.bfloat16 if dtype == "bf16" else torch.float16),
        low_cpu_mem_usage=True,
    )
    return proc, model


@torch.inference_mode()
def run_turn(proc, model, images, prompt_text, max_new_tokens):
    """Run a single turn through the model with given images."""
    content = []
    for _ in images:
        content.append({"type": "image"})
    content.append({"type": "text", "text": prompt_text})

    messages = [{"role": "user", "content": content}]
    prompt = proc.apply_chat_template(messages, add_generation_prompt=True)

    inputs = proc(text=prompt, images=images, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    # Strip input tokens to get only generated output
    gen_ids = out_ids[:, inputs["input_ids"].shape[-1]:]
    text = proc.batch_decode(gen_ids, skip_special_tokens=True)[0]
    return text


def run_problem(proc, model, frames_dir, problem, header, max_new_tokens):
    """Run a full problem through the loaded model."""
    qh = header

    def load_img(path_str):
        return Image.open(frames_dir / path_str.replace("./frames/", "")).convert("RGB")

    goal_img = load_img(problem["goal_image"])
    ref_img = load_img(problem["blocks_reference"])

    turn0_prompt = build_turn0_prompt(qh)

    # For local models, we process each turn independently with all context images
    # (goal + ref + turn image) since multi-turn chat is model-dependent
    turns_out = []

    for ti, turn in enumerate(problem["turns"]):
        is_turn1 = (ti == 0)
        turn_img = load_img(turn["image"])

        prompt_text = build_turn_prompt(
            turn["turn"], is_turn1,
            qh["q1_options"], qh["q2_options"],
            qh["q3_options"], qh["q4_options"],
        )

        full_prompt = turn0_prompt + "\n\n" + prompt_text

        text = run_turn(proc, model, [goal_img, ref_img, turn_img],
                        full_prompt, max_new_tokens)

        parsed = try_parse_json(text)
        gt = {k: turn[k] for k in ["q1", "q2", "q3", "q4", "q4_1", "q4_2", "q4_2_1"]}
        sc = score_turn(parsed, gt, is_turn1)

        turns_out.append({
            "turn": turn["turn"],
            "image": turn["image"],
            "prompt_used": prompt_text,
            "response_raw": text,
            "parsed": parsed,
            "ground_truth": gt,
            "score": sc,
            "timestamp": datetime.now().isoformat(),
        })

    return turns_out


def main():
    ap = argparse.ArgumentParser(description="Task 1 Tangram evaluation (local)")
    ap.add_argument("--problems", default="./problems/tangram.json")
    ap.add_argument("--frames_dir", default="./frames")
    ap.add_argument("--output_dir", default="./results/local")
    ap.add_argument("--output_suffix", default="",
                    help="Append to output filename, e.g. '_s0'. Used for parallel shards.")
    ap.add_argument("--models", default="all")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    args = ap.parse_args()

    data = json.loads(Path(args.problems).read_text(encoding="utf-8"))
    header = data["question_header"]
    problems = data["problems"]
    frames_dir = Path(args.frames_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.end is None:
        args.end = len(problems)

    model_entries = get_model_entry(args.models)

    for tag, hf_id in model_entries:
        print("=" * 80)
        print(f"[TASK1] MODEL: {tag}  |  {hf_id}")
        print("=" * 80)

        # Load model once per tag
        proc, model = load_model(hf_id, args.device_map, args.dtype)

        results = []
        t0 = time.time()
        total_problems = args.end - args.start

        for i in range(total_problems):
            idx = args.start + i
            p = problems[idx]
            pid = p["problem_id"]
            print(f"[{i + 1}/{total_problems}] {pid} ({p['num_turns']} turns)", flush=True)

            p_start = time.time()
            turns_out = run_problem(
                proc, model, frames_dir,
                p, header, args.max_new_tokens,
            )
            p_elapsed = time.time() - p_start

            results.append({
                "problem_id": pid,
                "scenario": p["scenario"],
                "api": "local",
                "model_tag": tag,
                "hf_id": hf_id,
                "num_turns": p["num_turns"],
                "turns": turns_out,
            })

            # Incremental save
            out_path = out_dir / f"task1__{tag}{args.output_suffix}.json"
            out_path.write_text(json.dumps(
                {"meta": {"model_tag": tag, "hf_id": hf_id}, "results": results},
                indent=2, ensure_ascii=False,
            ), encoding="utf-8")

            # ETA based on average time per completed problem
            completed = i + 1
            elapsed_total = time.time() - t0
            avg_per_problem = elapsed_total / completed
            remaining_problems = total_problems - completed
            eta = avg_per_problem * remaining_problems
            print(
                f"    done in {fmt_hms(p_elapsed)}  |  "
                f"elapsed {fmt_hms(elapsed_total)}  |  "
                f"ETA {fmt_hms(eta)}  "
                f"({completed}/{total_problems})",
                flush=True,
            )

        summary = summarize(results)
        out_path = out_dir / f"task1__{tag}.json"
        out_path.write_text(json.dumps(
            {"meta": {"model_tag": tag, "hf_id": hf_id}, "summary": summary, "results": results},
            indent=2, ensure_ascii=False,
        ), encoding="utf-8")
        print(f"Saved: {out_path}")
        print(f"Model time: {time.time() - t0:.1f}s")

        del model
        del proc
        clear_mem()


if __name__ == "__main__":
    main()
