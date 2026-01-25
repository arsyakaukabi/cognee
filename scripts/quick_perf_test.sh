#!/bin/bash
# Quick Performance Test for Cognee Search Endpoint
# Shows full timing breakdown for each function in the call chain
#
# Usage: bash scripts/quick_perf_test.sh
#
# Make sure server is running:
#   uv run uvicorn cognee.api.client:app --host 0.0.0.0 --port 8000

set -e

QUERY="${1:-Apa syarat utk mendapatkan diskon d SUSHI OK Banjarmasin?}"
TOP_K="${2:-50}"

echo "======================================"
echo "COGNEE RETRIEVAL PERFORMANCE TEST"
echo "======================================"
echo "Query: $QUERY"
echo "Top K: $TOP_K"
echo ""

echo ">>> Sending request to /search/retrieval..."
START_TIME=$(date +%s%3N)

RESPONSE=$(curl -s -w "\n%{time_total}" \
  --location 'http://localhost:8000/api/v1/search/retrieval' \
  --header 'Content-Type: application/json' \
  --data "{
    \"query\": \"$QUERY\",
    \"top_k\": $TOP_K,
    \"search_type\": \"graph_completion_custom\"
  }")

CURL_TIME=$(echo "$RESPONSE" | tail -1)
RESULT_COUNT=$(echo "$RESPONSE" | head -1 | grep -o '"idKnowledge"' | wc -l)

echo ">>> Response received!"
echo ""

sleep 1

# Find the latest log file
LOG_FILE=$(ls -t logs/*.log 2>/dev/null | head -1)

if [ -z "$LOG_FILE" ]; then
    echo "ERROR: No log files found in logs/"
    exit 1
fi

echo "======================================"
echo "TIMING BREAKDOWN (from $LOG_FILE)"
echo "======================================"
echo ""

echo "--- BRUTE FORCE TRIPLET SEARCH ---"
grep "⏱️ \[STEP" "$LOG_FILE" | tail -10

echo ""
echo "--- RESULT MAPPING ---"
grep -E "⏱️ \[MAP_TRIPLETS" "$LOG_FILE" | tail -3

echo ""
echo "--- DB VECTOR SEARCH (per collection) ---"
grep "⚡ \[DB Vector Search\]" "$LOG_FILE" | tail -7 | awk -F'|' '{print $2}'

echo ""
echo "======================================"
echo "SUMMARY"
echo "======================================"
echo "  Total Curl Time:    ${CURL_TIME}s"
echo "  Results Returned:   $RESULT_COUNT items"
echo ""

# Calculate breakdown
EMBED=$(grep "⏱️ \[STEP 0\]" "$LOG_FILE" | tail -1 | grep -oP '\d+\.\d+ms' | head -1)
VECTOR=$(grep "⏱️ \[STEP 1\]" "$LOG_FILE" | tail -1 | grep -oP '\d+\.\d+ms' | head -1)
GRAPH=$(grep "⏱️ \[STEP 3\]" "$LOG_FILE" | tail -1 | grep -oP '\d+\.\d+ms' | head -1)
MAPPING=$(grep "⏱️ \[MAP_TRIPLETS\]" "$LOG_FILE" | tail -1 | grep -oP 'Total: \d+\.\d+ms' | grep -oP '\d+\.\d+')

echo "BREAKDOWN:"
echo "  [STEP 0] Embed Query:       $EMBED"
echo "  [STEP 1] Vector Search:     $VECTOR"
echo "  [STEP 3] Graph Projection:  $GRAPH"
echo "  [MAP]    Result Mapping:    ${MAPPING}ms"
echo ""
echo "======================================"
