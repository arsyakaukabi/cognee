#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS_DIR="$ROOT_DIR/scripts"

if [ -f "$ROOT_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$ROOT_DIR/.env"
  set +a
fi

: "${MIGRATION_OUTPUT_DIR:=$ROOT_DIR/output}"

mkdir -p "$MIGRATION_OUTPUT_DIR"

if [[ "${MIGRATION_CLEAN_OUTPUT:-false}" == "true" || "${MIGRATION_CLEAN_OUTPUT:-false}" == "1" ]]; then
  rm -rf "$MIGRATION_OUTPUT_DIR/raw" "$MIGRATION_OUTPUT_DIR/neo4j_csv" "$MIGRATION_OUTPUT_DIR/reports"
fi

export PYTHONPATH="$SCRIPTS_DIR"

# Use a consistent Python interpreter to avoid pip/module mismatches.
PYTHON_BIN="$(command -v python3)"

"$PYTHON_BIN" "$SCRIPTS_DIR/export_kuzu.py"
"$PYTHON_BIN" "$SCRIPTS_DIR/transform_to_neo4j.py"
"$PYTHON_BIN" "$SCRIPTS_DIR/import_neo4j.py"
"$PYTHON_BIN" "$SCRIPTS_DIR/validate_neo4j.py"

echo "Migration pipeline completed. Reports in $MIGRATION_OUTPUT_DIR/reports"
