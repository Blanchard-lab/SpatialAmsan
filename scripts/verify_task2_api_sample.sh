#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────────────────────────────────────
# Verify Task 2 API evaluation on a small sample (1 problem, 2-3 turns).
# Runs all 3 API models, prints per-turn scores side by side.
# Usage:
#   ./scripts/verify_task2_api_sample.sh            # default: problem idx 0
#   SAMPLE_IDX=1 ./scripts/verify_task2_api_sample.sh  # pick another problem
# ──────────────────────────────────────────────────────────────────────────────

. ./scripts/load_env.sh

PROBLEMS="${PROBLEMS:-./problems/wooden_puzzle.json}"
FRAMES_DIR="${FRAMES_DIR:-./frames}"
SAMPLE_IDX="${SAMPLE_IDX:-0}"

OPENAI_MODEL="${OPENAI_MODEL:-gpt-5.2}"
CLAUDE_MODEL="${CLAUDE_MODEL:-claude-sonnet-4-6}"
GEMINI_MODEL="${GEMINI_MODEL:-gemini-3-flash-preview}"

MAX_TOKENS="${MAX_TOKENS:-512}"
DELAY="${DELAY:-0.5}"

OUT_DIR="./results/verify_sample"
mkdir -p "$OUT_DIR"

echo "============================================================"
echo " Task 2 API — Sample Verification"
echo " Problem index: ${SAMPLE_IDX}"
echo " Models: ${OPENAI_MODEL} / ${CLAUDE_MODEL} / ${GEMINI_MODEL}"
echo "============================================================"
echo ""

# Run one problem per API
for API in openai claude gemini; do
  case "$API" in
    openai) MODEL_ARG="--openai_model ${OPENAI_MODEL}" ;;
    claude) MODEL_ARG="--claude_model ${CLAUDE_MODEL}" ;;
    gemini) MODEL_ARG="--gemini_model ${GEMINI_MODEL}" ;;
  esac

  echo ">>> [${API}] Running problem ${SAMPLE_IDX} ..."
  python ./task2_eval/apis/evaluate_task2.py \
    --api "$API" \
    --problems "${PROBLEMS}" \
    --frames_dir "${FRAMES_DIR}" \
    --output "${OUT_DIR}/verify_${API}.json" \
    --start "${SAMPLE_IDX}" \
    --end "$(( SAMPLE_IDX + 1 ))" \
    --max_new_tokens "${MAX_TOKENS}" \
    --delay "${DELAY}" \
    ${MODEL_ARG}
  echo ""
done

# Pretty-print comparison
echo "============================================================"
echo " Results Comparison"
echo "============================================================"

python3 - "$OUT_DIR" <<'PYEOF'
import json, sys, os

out_dir = sys.argv[1]
apis = ["openai", "claude", "gemini"]
data = {}

for api in apis:
    path = os.path.join(out_dir, f"verify_{api}.json")
    if os.path.exists(path):
        data[api] = json.load(open(path))

if not data:
    print("No results found.")
    sys.exit(1)

# Get problem info from first available
first = next(iter(data.values()))
prob = first["results"][0]
print(f"\nProblem: {prob['problem_id']}  ({prob['num_turns']} turns)\n")

hdr = f"{'Turn':<6} {'Question':<12}"
for api in apis:
    if api in data:
        model = data[api]["meta"]["model"]
        label = f"{api} ({model})"
        hdr += f" {label:<35}"
print(hdr)
print("-" * len(hdr))

# Per turn, per question
q_keys = ["q1", "q2", "q3", "q4", "q4_2"]
q_labels = {"q1": "Q1 (block)", "q2": "Q2 (prev)", "q3": "Q3 (contact)",
            "q4": "Q4 (eval)", "q4_2": "Q4-2 (undo)"}

for ti in range(prob["num_turns"]):
    is_turn1 = (ti == 0)
    turn_num = ti + 1

    for qk in q_keys:
        if qk == "q2" and is_turn1:
            continue

        label = q_labels[qk]
        row = f"  {turn_num:<4} {label:<12}"

        for api in apis:
            if api not in data:
                row += f" {'N/A':<35}"
                continue

            turn_data = data[api]["results"][0]["turns"][ti]
            score = turn_data.get("score", {})
            parsed = turn_data.get("parsed", {})
            gt = turn_data.get("ground_truth", {})
            pred_data = parsed.get("data", {}) or {}

            score_key = f"{qk}_correct"

            if score_key not in score:
                row += f" {'—':<35}"
                continue

            correct = score[score_key]
            pred_val = pred_data.get(qk, "?")
            gt_val = gt.get(qk, "?")

            mark = "OK" if correct else "WRONG"
            cell = f"{mark}  pred={pred_val} gt={gt_val}"
            row += f" {cell:<35}"

        print(row)

    print()

# Summary per API
print("=" * 60)
print("Per-API accuracy:")
for api in apis:
    if api not in data:
        continue
    turns = data[api]["results"][0]["turns"]
    total = 0
    correct = 0
    for t in turns:
        s = t.get("score", {})
        if not s.get("scored"):
            continue
        total += s.get("n_scored", 0)
        correct += s.get("n_correct", 0)
    acc = (correct / total * 100) if total > 0 else 0
    model = data[api]["meta"]["model"]
    print(f"  {api:<8} ({model}): {correct}/{total} = {acc:.1f}%")

# Check JSON parse success
print("\nJSON parse success:")
for api in apis:
    if api not in data:
        continue
    turns = data[api]["results"][0]["turns"]
    ok = sum(1 for t in turns if t.get("parsed", {}).get("ok"))
    print(f"  {api:<8}: {ok}/{len(turns)} turns parsed")

# Show raw responses
print("\n" + "=" * 60)
print("Raw responses (for debugging):")
print("=" * 60)
for api in apis:
    if api not in data:
        continue
    print(f"\n--- {api} ({data[api]['meta']['model']}) ---")
    for t in data[api]["results"][0]["turns"]:
        fr = t.get("finish_reason")
        fr_str = f"  [finish_reason={fr}]" if fr else ""
        print(f"  Turn {t['turn']}:{fr_str}")
        raw = t.get("response_raw", "")
        # Truncate long responses
        if len(raw) > 300:
            raw = raw[:300] + "..."
        for line in raw.split("\n"):
            print(f"    {line}")
        print()

PYEOF

echo ""
echo "Full results saved in: ${OUT_DIR}/"
echo "Done."
