#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────────────────────────────────────
# Verify Task 1 local evaluation on a small sample (1 problem).
# Runs local HuggingFace models, prints per-turn scores side by side.
# Usage:
#   ./scripts/verify_task1_local_sample.sh            # default: problem idx 0
#   SAMPLE_IDX=1 ./scripts/verify_task1_local_sample.sh  # pick another problem
#   MODELS="qwen2p5_vl_7b" ./scripts/verify_task1_local_sample.sh  # single model
#   MODELS="all" ./scripts/verify_task1_local_sample.sh  # include llama4
# ──────────────────────────────────────────────────────────────────────────────

PROBLEMS="${PROBLEMS:-./problems/tangram.json}"
FRAMES_DIR="${FRAMES_DIR:-./frames}"
SAMPLE_IDX="${SAMPLE_IDX:-0}"

MAX_TOKENS="${MAX_TOKENS:-512}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
DTYPE="${DTYPE:-bf16}"

# Model selection: comma-separated tags, or "all"
# Default: 3 models (excluding llama4 which is very slow)
MODELS="${MODELS:-qwen2p5_vl_7b,internvl3_8b_hf,llava_onevision_qwen2_7b_ov_hf}"

IFS=',' read -ra MODEL_LIST <<< "$MODELS"

OUT_DIR="./results/verify_task1_local_sample"
mkdir -p "$OUT_DIR"

echo "============================================================"
echo " Task 1 Local — Sample Verification"
echo " Problem index: ${SAMPLE_IDX}"
echo " Models: ${MODELS}"
echo "============================================================"
echo ""

# Run one problem per model
for MODEL in "${MODEL_LIST[@]}"; do
  echo ">>> [${MODEL}] Running problem ${SAMPLE_IDX} ..."
  python ./task1_eval/local/evaluate_local_task1.py \
    --problems "${PROBLEMS}" \
    --frames_dir "${FRAMES_DIR}" \
    --output_dir "${OUT_DIR}" \
    --models "${MODEL}" \
    --start "${SAMPLE_IDX}" \
    --end "$(( SAMPLE_IDX + 1 ))" \
    --max_new_tokens "${MAX_TOKENS}" \
    --device_map "${DEVICE_MAP}" \
    --dtype "${DTYPE}"
  echo ""
done

# Pretty-print comparison
echo "============================================================"
echo " Results Comparison"
echo "============================================================"

python3 - "$OUT_DIR" "$MODELS" <<'PYEOF'
import json, sys, os

out_dir = sys.argv[1]
model_tags = sys.argv[2].split(",")
data = {}

for tag in model_tags:
    path = os.path.join(out_dir, f"task1__{tag}.json")
    if os.path.exists(path):
        data[tag] = json.load(open(path))

if not data:
    print("No results found.")
    sys.exit(1)

# Get problem info from first available
first = next(iter(data.values()))
prob = first["results"][0]
print(f"\nProblem: {prob['problem_id']}  ({prob['num_turns']} turns)\n")

col_width = 40
hdr = f"{'Turn':<6} {'Question':<12}"
for tag in model_tags:
    if tag in data:
        label = f"{tag}"
        hdr += f" {label:<{col_width}}"
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

        for tag in model_tags:
            if tag not in data:
                row += f" {'N/A':<{col_width}}"
                continue

            turn_data = data[tag]["results"][0]["turns"][ti]
            score = turn_data.get("score", {})
            parsed = turn_data.get("parsed", {})
            gt = turn_data.get("ground_truth", {})
            pred_data = parsed.get("data", {}) or {}

            score_key = f"{qk}_correct"

            if score_key not in score:
                row += f" {'—':<{col_width}}"
                continue

            correct = score[score_key]
            pred_val = pred_data.get(qk, "?")
            gt_val = gt.get(qk, "?")

            mark = "OK" if correct else "WRONG"
            cell = f"{mark}  pred={pred_val} gt={gt_val}"
            row += f" {cell:<{col_width}}"

        print(row)

    print()

# Summary per model
print("=" * 60)
print("Per-model accuracy:")
for tag in model_tags:
    if tag not in data:
        continue
    turns = data[tag]["results"][0]["turns"]
    total = 0
    correct = 0
    for t in turns:
        s = t.get("score", {})
        if not s.get("scored"):
            continue
        total += s.get("n_scored", 0)
        correct += s.get("n_correct", 0)
    acc = (correct / total * 100) if total > 0 else 0
    hf_id = data[tag]["meta"].get("hf_id", tag)
    print(f"  {tag:<30} ({hf_id}): {correct}/{total} = {acc:.1f}%")

# Check JSON parse success
print("\nJSON parse success:")
for tag in model_tags:
    if tag not in data:
        continue
    turns = data[tag]["results"][0]["turns"]
    ok = sum(1 for t in turns if t.get("parsed", {}).get("ok"))
    print(f"  {tag:<30}: {ok}/{len(turns)} turns parsed")

# Show raw responses
print("\n" + "=" * 60)
print("Raw responses (for debugging):")
print("=" * 60)
for tag in model_tags:
    if tag not in data:
        continue
    hf_id = data[tag]["meta"].get("hf_id", tag)
    print(f"\n--- {tag} ({hf_id}) ---")
    for t in data[tag]["results"][0]["turns"]:
        print(f"  Turn {t['turn']}:")
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
