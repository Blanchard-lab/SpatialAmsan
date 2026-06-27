#!/usr/bin/env bash
set -euo pipefail

. ./scripts/load_env.sh

mkdir -p ./results

PROBLEMS="${PROBLEMS:-./problems/tangram.json}"
FRAMES_DIR="${FRAMES_DIR:-./frames}"

OPENAI_MODEL="${OPENAI_MODEL:-gpt-5.2}"
CLAUDE_MODEL="${CLAUDE_MODEL:-claude-sonnet-4-6}"
GEMINI_MODEL="${GEMINI_MODEL:-gemini-3-flash-preview}"

DELAY="${DELAY:-0.5}"
MAX_TOKENS="${MAX_TOKENS:-2048}"

START="${START:-0}"
END="${END:-}"

# ---------- progress tracking ----------
TOTAL_TASKS=3
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

_end_arg() {
  local v="$1"
  if [ -n "$v" ]; then
    echo "--end" "$v"
  fi
}

# ---------- Task 1 API evaluation ----------

run_task "Task1 API  [openai / ${OPENAI_MODEL}]" \
  python ./task1_eval/apis/evaluate_task1.py \
    --api openai \
    --problems "${PROBLEMS}" \
    --frames_dir "${FRAMES_DIR}" \
    --output "./results/task1_openai.json" \
    --openai_model "${OPENAI_MODEL}" \
    --start "${START}" \
    $(_end_arg "${END}") \
    --max_new_tokens "${MAX_TOKENS}" \
    --delay "${DELAY}"

run_task "Task1 API  [claude / ${CLAUDE_MODEL}]" \
  python ./task1_eval/apis/evaluate_task1.py \
    --api claude \
    --problems "${PROBLEMS}" \
    --frames_dir "${FRAMES_DIR}" \
    --output "./results/task1_claude.json" \
    --claude_model "${CLAUDE_MODEL}" \
    --start "${START}" \
    $(_end_arg "${END}") \
    --max_new_tokens "${MAX_TOKENS}" \
    --delay "${DELAY}"

run_task "Task1 API  [gemini / ${GEMINI_MODEL}]" \
  python ./task1_eval/apis/evaluate_task1.py \
    --api gemini \
    --problems "${PROBLEMS}" \
    --frames_dir "${FRAMES_DIR}" \
    --output "./results/task1_gemini.json" \
    --gemini_model "${GEMINI_MODEL}" \
    --start "${START}" \
    $(_end_arg "${END}") \
    --max_new_tokens "${MAX_TOKENS}" \
    --delay "${DELAY}"

printf "\n[##############################] 3/3  elapsed: %s  ETA: done\n" \
  "$(_fmt_sec $(( $(date +%s) - _START )))"
printf "All Task1 API evaluations completed.\n\n"
