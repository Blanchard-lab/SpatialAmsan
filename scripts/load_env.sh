set -euo pipefail
if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

: "${OPENAI_API_KEY:?OPENAI_API_KEY is not set}"
: "${ANTHROPIC_API_KEY:?ANTHROPIC_API_KEY is not set}"
: "${GOOGLE_API_KEY:?GOOGLE_API_KEY is not set}"