#!/usr/bin/env bash
set -euo pipefail

mkdir -p ./results/local

PROBLEMS="${PROBLEMS:-./problems/wooden_puzzle.json}"
FRAMES_DIR="${FRAMES_DIR:-./frames}"

MAX_TOKENS="${MAX_TOKENS:-512}"

# Optional slicing
START="${START:-0}"
END="${END:-}"

# Device / precision
DEVICE_MAP="${DEVICE_MAP:-auto}"
DTYPE="${DTYPE:-bf16}"

# ---------- model selection ----------
# By default, run all models EXCEPT llama4 (it is very slow).
# Set MODELS="all" to include llama4, or run llama4 separately via:
#   ./scripts/run_task2_local.sh --llama4-only
# Individual models can be run with MODELS="qwen2p5_vl_7b" etc.
MODELS="${MODELS:-}"
LLAMA4_ONLY=false

for arg in "$@"; do
  case "$arg" in
    --llama4-only) LLAMA4_ONLY=true ;;
  esac
done

if [ "$LLAMA4_ONLY" = true ]; then
  MODEL_LIST=("llama4_scout_17b16e")
elif [ -n "$MODELS" ]; then
  IFS=',' read -ra MODEL_LIST <<< "$MODELS"
else
  # Default: all except llama4
  MODEL_LIST=(
    "qwen2p5_vl_7b"
    "internvl3_8b_hf"
    "llava_onevision_qwen2_7b_ov_hf"
  )
fi

# ---------- progress tracking ----------
TOTAL_TASKS=${#MODEL_LIST[@]}
CURRENT_TASK=0
RESUME_FROM="${RESUME_FROM:-1}"
_START=$(date +%s)

_fmt_sec() {
  local s=$1
  if   [ "$s" -ge 3600 ]; then printf "%dh %02dm %02ds" $((s/3600)) $(( (s%3600)/60 )) $((s%60))
  elif [ "$s" -ge 60 ];   then printf "%dm %02ds" $((s/60)) $((s%60))
  else printf "%ds" "$s"
  fi
}

run_task() {
  local label="$1"; shift
  CURRENT_TASK=$(( CURRENT_TASK + 1 ))

  local bar_width=30
  local filled=$(( bar_width * CURRENT_TASK / TOTAL_TASKS ))
  local empty=$(( bar_width - filled ))
  local bar
  bar="$(printf '%0.s#' $(seq 1 $filled))$(printf '%0.s-' $(seq 1 $empty))"
  local elapsed=$(( $(date +%s) - _START ))
  local eta="--"
  if [ "$CURRENT_TASK" -gt 1 ]; then
    local avg=$(( elapsed / (CURRENT_TASK - 1) ))
    local remaining=$(( avg * (TOTAL_TASKS - CURRENT_TASK + 1) ))
    eta="$(_fmt_sec $remaining)"
  fi

  if [ "$CURRENT_TASK" -lt "$RESUME_FROM" ]; then
    printf "\n[%s] %d/%d  [SKIPPED]\n>>> %s\n" "$bar" "$CURRENT_TASK" "$TOTAL_TASKS" "$label"
    return 0
  fi

  printf "\n[%s] %d/%d  elapsed: %s  ETA: %s\n" \
    "$bar" "$CURRENT_TASK" "$TOTAL_TASKS" "$(_fmt_sec $elapsed)" "$eta"
  printf ">>> Running: %s\n\n" "$label"
  "$@"
}

# helper: pass --end only if set
_end_arg() {
  local v="$1"
  if [ -n "$v" ]; then
    echo "--end" "$v"
  fi
}

# ---------- main loop ----------
for MODEL in "${MODEL_LIST[@]}"; do
  run_task "Task2 local  [${MODEL}]" \
    python ./task2_eval/local/evaluate_local_task2.py \
      --problems "${PROBLEMS}" \
      --frames_dir "${FRAMES_DIR}" \
      --output_dir "./results/local" \
      --models "${MODEL}" \
      --start "${START}" \
      $(_end_arg "${END}") \
      --max_new_tokens "${MAX_TOKENS}" \
      --device_map "${DEVICE_MAP}" \
      --dtype "${DTYPE}"
done

printf "\n[##############################] %d/%d  elapsed: %s  ETA: done\n" \
  "$TOTAL_TASKS" "$TOTAL_TASKS" "$(_fmt_sec $(( $(date +%s) - _START )))"

if [ "$LLAMA4_ONLY" = true ]; then
  printf "Llama4 Task2 completed.\n\n"
else
  printf "All Task2 local evaluations completed.\n\n"
fi
