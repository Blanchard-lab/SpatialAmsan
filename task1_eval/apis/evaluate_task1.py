#!/usr/bin/env python3
"""
Task 1 (Tangram) multi-turn evaluation via API models.
Turn 0: header + goal image + block reference image (no questions).
Turn 1+: one manipulation image + questions, expecting JSON response.
"""

import json
import mimetypes
import os
import re
import time
import base64
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    from google import genai as google_genai
    from google.genai import types as genai_types
except ImportError:
    google_genai = None
    genai_types = None

from PIL import Image


def _mime(p: Path) -> str:
    """Return MIME type by inspecting file header (magic bytes)."""
    header = p.read_bytes()[:12]
    if header[:8] == b'\x89PNG\r\n\x1a\n':
        return "image/png"
    if header[:2] == b'\xff\xd8':
        return "image/jpeg"
    if header[:4] == b'RIFF' and header[8:12] == b'WEBP':
        return "image/webp"
    mt, _ = mimetypes.guess_type(str(p))
    return mt or "image/jpeg"


def encode_b64_image(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode("utf-8")


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


def openai_usage_dict(resp) -> Optional[Dict[str, Any]]:
    u = getattr(resp, "usage", None)
    if u is None:
        return None
    out: Dict[str, Any] = {}
    for k in ["prompt_tokens", "completion_tokens", "total_tokens",
              "input_tokens", "output_tokens"]:
        if hasattr(u, k):
            v = getattr(u, k)
            if isinstance(v, (int, float, str)) or v is None:
                out[k] = v
    return out


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

def score_turn(parsed: Dict[str, Any], gt: Dict[str, Any], is_turn1: bool) -> Dict[str, Any]:
    if not parsed.get("ok") or not isinstance(parsed.get("data"), dict):
        return {"scored": False}

    data = parsed["data"]
    scores = {}

    # Q1 - single select
    gt_q1 = gt.get("q1")
    pred_q1 = data.get("q1")
    if gt_q1 is not None and pred_q1 is not None:
        scores["q1_correct"] = (int(pred_q1) == int(gt_q1))

    # Q2 - single select (skip Turn 1)
    if not is_turn1:
        gt_q2 = gt.get("q2")
        pred_q2 = data.get("q2")
        if gt_q2 is not None and pred_q2 is not None:
            scores["q2_correct"] = (int(pred_q2) == int(gt_q2))

    # Q3 - multi select (set comparison)
    gt_q3 = gt.get("q3")
    pred_q3 = data.get("q3")
    if gt_q3 is not None and pred_q3 is not None:
        try:
            gt_set = sorted(set(int(v) for v in gt_q3))
            pred_set = sorted(set(int(v) for v in (pred_q3 if isinstance(pred_q3, list) else [])))
            scores["q3_correct"] = (gt_set == pred_set)
        except Exception:
            scores["q3_correct"] = False

    # Q4 - multi select (set comparison)
    gt_q4 = gt.get("q4")
    pred_q4 = data.get("q4")
    if gt_q4 is not None and pred_q4 is not None:
        try:
            gt_set = sorted(set(int(v) for v in gt_q4))
            pred_set = sorted(set(int(v) for v in (pred_q4 if isinstance(pred_q4, list) else [])))
            scores["q4_correct"] = (gt_set == pred_set)
        except Exception:
            scores["q4_correct"] = False

    # Q4-2 - binary (conditional)
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


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
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


# ── API callers ──────────────────────────────────────────────────────────────

class Task1Evaluator:
    def __init__(self, openai_api_key, anthropic_api_key, google_api_key,
                 frames_dir, openai_model, claude_model, gemini_model):
        self.frames_dir = Path(frames_dir)
        self.openai_client = openai.OpenAI(api_key=openai_api_key) if (openai and openai_api_key) else None
        self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key) if (anthropic and anthropic_api_key) else None
        self.gemini_client = None
        self.gemini_model_name = gemini_model
        if google_genai and google_api_key:
            self.gemini_client = google_genai.Client(api_key=google_api_key)
        self.openai_model = openai_model
        self.claude_model = claude_model

    def _img(self, path_str: str) -> Path:
        p = self.frames_dir / path_str.replace("./frames/", "")
        if not p.exists():
            raise FileNotFoundError(f"Missing image: {p}")
        return p

    def _openai_use_responses_api(self):
        """Check if the model requires the Responses API (pro models)."""
        return "pro" in str(self.openai_model).lower()

    def _openai_b64_url(self, img_path: Path) -> str:
        return f"data:{_mime(img_path)};base64,{encode_b64_image(img_path)}"

    def run_problem_openai(self, problem, header, delay, max_tokens):
        if not self.openai_client:
            return {"error": "OpenAI client not initialized"}

        if self._openai_use_responses_api():
            return self._run_problem_openai_responses(problem, header, delay, max_tokens)
        return self._run_problem_openai_chat(problem, header, delay, max_tokens)

    def _run_problem_openai_responses(self, problem, header, delay, max_tokens):
        """Use the Responses API for pro models (gpt-5.2-pro, etc.)."""
        qh = header
        goal_img = self._img(problem["goal_image"])
        ref_img = self._img(problem["blocks_reference"])

        turn0_prompt = build_turn0_prompt(qh)

        # Turn 0: header + goal + reference
        resp0 = self.openai_client.responses.create(
            model=self.openai_model,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": turn0_prompt},
                    {"type": "input_image", "image_url": self._openai_b64_url(goal_img)},
                    {"type": "input_image", "image_url": self._openai_b64_url(ref_img)},
                ],
            }],
            max_output_tokens=100,
            store=True,
            text={"format": {"type": "text"}},
        )
        prev_id = resp0.id

        turns_out = []
        for ti, turn in enumerate(problem["turns"]):
            is_turn1 = (ti == 0)
            turn_img = self._img(turn["image"])
            prompt = build_turn_prompt(
                turn["turn"], is_turn1,
                qh["q1_options"], qh["q2_options"],
                qh["q3_options"], qh["q4_options"],
            )

            resp = self.openai_client.responses.create(
                model=self.openai_model,
                input=[{
                    "role": "user",
                    "content": [
                        {"type": "input_image", "image_url": self._openai_b64_url(turn_img)},
                        {"type": "input_text", "text": prompt},
                    ],
                }],
                max_output_tokens=max_tokens,
                previous_response_id=prev_id,
                store=True,
                text={"format": {"type": "text"}},
            )
            text = resp.output_text or ""
            # Fallback: extract text from output items if output_text is empty
            if not text:
                for item in (resp.output or []):
                    for c in (getattr(item, "content", None) or []):
                        if hasattr(c, "text") and c.text:
                            text += c.text
            prev_id = resp.id

            usage = None
            u = getattr(resp, "usage", None)
            if u:
                usage = {k: getattr(u, k, None) for k in
                         ["input_tokens", "output_tokens", "total_tokens"]}

            parsed = try_parse_json(text)
            gt = {k: turn[k] for k in ["q1", "q2", "q3", "q4", "q4_1", "q4_2", "q4_2_1"]}
            sc = score_turn(parsed, gt, is_turn1)

            turns_out.append({
                "turn": turn["turn"],
                "image": turn["image"],
                "prompt_used": prompt,
                "response_raw": text,
                "parsed": parsed,
                "ground_truth": gt,
                "score": sc,
                "usage": usage,
                "timestamp": datetime.now().isoformat(),
            })

            if delay > 0:
                time.sleep(delay)

        return {
            "problem_id": problem["problem_id"],
            "scenario": problem["scenario"],
            "api": "openai",
            "model": self.openai_model,
            "num_turns": problem["num_turns"],
            "turns": turns_out,
        }

    def _run_problem_openai_chat(self, problem, header, delay, max_tokens):
        """Use the Chat Completions API for standard models."""
        qh = header
        goal_img = self._img(problem["goal_image"])
        ref_img = self._img(problem["blocks_reference"])

        # Turn 0: header + goal + reference
        turn0_prompt = build_turn0_prompt(qh)
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": turn0_prompt},
                {"type": "image_url", "image_url": {"url": self._openai_b64_url(goal_img)}},
                {"type": "image_url", "image_url": {"url": self._openai_b64_url(ref_img)}},
            ],
        }]

        # Get acknowledgment for Turn 0
        params = {"model": self.openai_model, "messages": messages, "temperature": 0.0}
        if str(self.openai_model).startswith("gpt-5"):
            params["max_completion_tokens"] = 100
        else:
            params["max_tokens"] = 100
        resp0 = self.openai_client.chat.completions.create(**params)
        ack = resp0.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": ack})

        turns_out = []
        for ti, turn in enumerate(problem["turns"]):
            is_turn1 = (ti == 0)
            turn_img = self._img(turn["image"])
            prompt = build_turn_prompt(
                turn["turn"], is_turn1,
                qh["q1_options"], qh["q2_options"],
                qh["q3_options"], qh["q4_options"],
            )

            messages.append({
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": self._openai_b64_url(turn_img)}},
                    {"type": "text", "text": prompt},
                ],
            })

            params = {"model": self.openai_model, "messages": messages, "temperature": 0.0}
            if str(self.openai_model).startswith("gpt-5"):
                params["max_completion_tokens"] = max_tokens
            else:
                params["max_tokens"] = max_tokens

            resp = self.openai_client.chat.completions.create(**params)
            text = resp.choices[0].message.content or ""
            messages.append({"role": "assistant", "content": text})

            parsed = try_parse_json(text)
            gt = {k: turn[k] for k in ["q1", "q2", "q3", "q4", "q4_1", "q4_2", "q4_2_1"]}
            sc = score_turn(parsed, gt, is_turn1)

            turns_out.append({
                "turn": turn["turn"],
                "image": turn["image"],
                "prompt_used": prompt,
                "response_raw": text,
                "parsed": parsed,
                "ground_truth": gt,
                "score": sc,
                "usage": openai_usage_dict(resp),
                "timestamp": datetime.now().isoformat(),
            })

            if delay > 0:
                time.sleep(delay)

        return {
            "problem_id": problem["problem_id"],
            "scenario": problem["scenario"],
            "api": "openai",
            "model": self.openai_model,
            "num_turns": problem["num_turns"],
            "turns": turns_out,
        }

    def run_problem_claude(self, problem, header, delay, max_tokens):
        if not self.anthropic_client:
            return {"error": "Anthropic client not initialized"}

        qh = header
        goal_img = self._img(problem["goal_image"])
        ref_img = self._img(problem["blocks_reference"])

        turn0_prompt = build_turn0_prompt(qh)
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": turn0_prompt},
                {"type": "image", "source": {"type": "base64", "media_type": _mime(goal_img), "data": encode_b64_image(goal_img)}},
                {"type": "image", "source": {"type": "base64", "media_type": _mime(ref_img), "data": encode_b64_image(ref_img)}},
            ],
        }]

        resp0 = self.anthropic_client.messages.create(
            model=self.claude_model, max_tokens=100, temperature=0.0, messages=messages,
        )
        ack = resp0.content[0].text if resp0.content else ""
        messages.append({"role": "assistant", "content": [{"type": "text", "text": ack}]})

        turns_out = []
        for ti, turn in enumerate(problem["turns"]):
            is_turn1 = (ti == 0)
            turn_img = self._img(turn["image"])
            prompt = build_turn_prompt(
                turn["turn"], is_turn1,
                qh["q1_options"], qh["q2_options"],
                qh["q3_options"], qh["q4_options"],
            )

            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": _mime(turn_img), "data": encode_b64_image(turn_img)}},
                    {"type": "text", "text": prompt},
                ],
            })

            resp = self.anthropic_client.messages.create(
                model=self.claude_model, max_tokens=max_tokens, temperature=0.0, messages=messages,
            )
            text = resp.content[0].text if resp.content else ""
            messages.append({"role": "assistant", "content": [{"type": "text", "text": text}]})

            parsed = try_parse_json(text)
            gt = {k: turn[k] for k in ["q1", "q2", "q3", "q4", "q4_1", "q4_2", "q4_2_1"]}
            sc = score_turn(parsed, gt, is_turn1)

            usage = {"input_tokens": getattr(resp.usage, "input_tokens", None),
                     "output_tokens": getattr(resp.usage, "output_tokens", None)} if getattr(resp, "usage", None) else None

            turns_out.append({
                "turn": turn["turn"],
                "image": turn["image"],
                "prompt_used": prompt,
                "response_raw": text,
                "parsed": parsed,
                "ground_truth": gt,
                "score": sc,
                "usage": usage,
                "timestamp": datetime.now().isoformat(),
            })

            if delay > 0:
                time.sleep(delay)

        return {
            "problem_id": problem["problem_id"],
            "scenario": problem["scenario"],
            "api": "claude",
            "model": self.claude_model,
            "num_turns": problem["num_turns"],
            "turns": turns_out,
        }

    def run_problem_gemini(self, problem, header, delay, max_tokens):
        if not self.gemini_client:
            return {"error": "Gemini client not initialized"}

        qh = header
        goal_img = Image.open(self._img(problem["goal_image"]))
        ref_img = Image.open(self._img(problem["blocks_reference"]))

        turn0_prompt = build_turn0_prompt(qh)
        # Gemini needs a higher output token budget (image tokens counted differently)
        gemini_max_tokens = max(max_tokens, 8192)
        gen_config = genai_types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=gemini_max_tokens,
        )

        def _img_part(pil_img):
            """Convert PIL image to Gemini Part via inline_data."""
            buf = __import__("io").BytesIO()
            fmt = "PNG" if pil_img.mode == "RGBA" else "JPEG"
            pil_img.save(buf, format=fmt)
            mime = f"image/{'png' if fmt == 'PNG' else 'jpeg'}"
            return genai_types.Part(inline_data=genai_types.Blob(
                mime_type=mime, data=buf.getvalue(),
            ))

        # Use generate_content directly with manual history (chat API truncates)
        contents = [
            genai_types.Content(role="user", parts=[
                genai_types.Part.from_text(text=turn0_prompt),
                _img_part(goal_img),
                _img_part(ref_img),
            ]),
        ]

        # Turn 0: get acknowledgment
        resp0 = self.gemini_client.models.generate_content(
            model=self.gemini_model_name,
            contents=contents,
            config=genai_types.GenerateContentConfig(
                temperature=0.0, max_output_tokens=gemini_max_tokens,
            ),
        )
        ack = (resp0.text or "") if hasattr(resp0, "text") else ""
        contents.append(genai_types.Content(role="model", parts=[
            genai_types.Part.from_text(text=ack),
        ]))

        turns_out = []
        for ti, turn in enumerate(problem["turns"]):
            is_turn1 = (ti == 0)
            turn_img = Image.open(self._img(turn["image"]))
            prompt = build_turn_prompt(
                turn["turn"], is_turn1,
                qh["q1_options"], qh["q2_options"],
                qh["q3_options"], qh["q4_options"],
            )

            contents.append(genai_types.Content(role="user", parts=[
                _img_part(turn_img),
                genai_types.Part.from_text(text=prompt),
            ]))

            resp = self.gemini_client.models.generate_content(
                model=self.gemini_model_name,
                contents=contents,
                config=gen_config,
            )
            text = (resp.text or "") if hasattr(resp, "text") else ""

            contents.append(genai_types.Content(role="model", parts=[
                genai_types.Part.from_text(text=text or " "),
            ]))

            usage = None
            finish_reason = None
            try:
                um = resp.usage_metadata
                usage = {
                    "prompt_tokens": getattr(um, "prompt_token_count", None),
                    "candidates_tokens": getattr(um, "candidates_token_count", None),
                    "total_tokens": getattr(um, "total_token_count", None),
                }
            except Exception:
                pass
            try:
                if resp.candidates:
                    finish_reason = str(resp.candidates[0].finish_reason)
            except Exception:
                pass

            parsed = try_parse_json(text)
            gt = {k: turn[k] for k in ["q1", "q2", "q3", "q4", "q4_1", "q4_2", "q4_2_1"]}
            sc = score_turn(parsed, gt, is_turn1)

            turns_out.append({
                "turn": turn["turn"],
                "image": turn["image"],
                "prompt_used": prompt,
                "response_raw": text,
                "parsed": parsed,
                "ground_truth": gt,
                "score": sc,
                "usage": usage,
                "finish_reason": finish_reason,
                "timestamp": datetime.now().isoformat(),
            })

            if delay > 0:
                time.sleep(delay)

        return {
            "problem_id": problem["problem_id"],
            "scenario": problem["scenario"],
            "api": "gemini",
            "model": self.gemini_model_name,
            "num_turns": problem["num_turns"],
            "turns": turns_out,
        }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Task 1 Tangram evaluation (API)")
    parser.add_argument("--problems", default="./problems/tangram.json")
    parser.add_argument("--frames_dir", default="./frames")
    parser.add_argument("--api", choices=["openai", "claude", "gemini"], required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--openai_model", default="gpt-5.2")
    parser.add_argument("--claude_model", default="claude-sonnet-4-6")
    parser.add_argument("--gemini_model", default="gemini-3-flash-preview")
    args = parser.parse_args()

    data = json.loads(Path(args.problems).read_text(encoding="utf-8"))
    header = data["question_header"]
    problems = data["problems"]

    if args.end is None:
        args.end = len(problems)

    ev = Task1Evaluator(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        frames_dir=args.frames_dir,
        openai_model=args.openai_model,
        claude_model=args.claude_model,
        gemini_model=args.gemini_model,
    )

    model_name = {"openai": args.openai_model, "claude": args.claude_model, "gemini": args.gemini_model}[args.api]
    out_path = Path(args.output) if args.output else Path(f"./results/task1_{args.api}_{model_name.replace('/', '_')}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for i in range(args.start, args.end):
        p = problems[i]
        print(f"[{i + 1}/{args.end}] {p['problem_id']} ({p['num_turns']} turns)")

        if args.api == "openai":
            r = ev.run_problem_openai(p, header, args.delay, args.max_new_tokens)
        elif args.api == "claude":
            r = ev.run_problem_claude(p, header, args.delay, args.max_new_tokens)
        else:
            r = ev.run_problem_gemini(p, header, args.delay, args.max_new_tokens)

        results.append(r)

        # Incremental save
        out_path.write_text(json.dumps(
            {"meta": {"api": args.api, "model": model_name}, "results": results},
            indent=2, ensure_ascii=False,
        ), encoding="utf-8")

    summary = summarize(results)
    out_path.write_text(json.dumps(
        {"meta": {"api": args.api, "model": model_name}, "summary": summary, "results": results},
        indent=2, ensure_ascii=False,
    ), encoding="utf-8")
    print(f"Saved: {out_path}")
    print(f"Summary: {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
