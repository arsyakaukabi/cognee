#!/usr/bin/env bash
# Run retrieval evaluation after re-embedding.
# Edit INPUT_CSV (and optionally other paths) below, then run from repo root:
#   bash reembed_workflow/run_eval.sh

set -e

# --- Edit these to match your setup ---
INPUT_CSV="${INPUT_CSV:-/path/to/ground_truth.csv}"
OUTPUT_CSV="${OUTPUT_CSV:-reembed_workflow/results/eval_output.csv}"
LOG_FILE="${LOG_FILE:-reembed_workflow/results/eval.log}"
TRIPLETS_JSON="${TRIPLETS_JSON:-reembed_workflow/results/eval_triplets.json}"
TOP_K="${TOP_K:-5}"

# Repo root (script lives in reembed_workflow/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p "$(dirname "$OUTPUT_CSV")"

if [[ ! -f "$INPUT_CSV" ]]; then
  echo "Error: Input CSV not found: $INPUT_CSV"
  echo "Set INPUT_CSV or edit this script (run_eval.sh)."
  exit 1
fi

echo "Running retrieval evaluation..."
echo "  Input:    $INPUT_CSV"
echo "  Output:   $OUTPUT_CSV"
echo "  Log:      $LOG_FILE"
echo "  Top-K:    $TOP_K"

uv run python examples/python/retrieval_evaluation/run_eval_v2_compat.py \
  --input "$INPUT_CSV" \
  --output "$OUTPUT_CSV" \
  --log "$LOG_FILE" \
  --triplets-json "$TRIPLETS_JSON" \
  --top-k "$TOP_K"

echo "Done. Results: $OUTPUT_CSV"
