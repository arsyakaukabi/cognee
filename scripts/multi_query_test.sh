#!/bin/bash
# Multi-Query Performance Test
# Tests different queries to analyze cold vs warm performance patterns
#
# Usage: bash scripts/multi_query_test.sh

set -e

echo "=============================================="
echo "MULTI-QUERY PERFORMANCE TEST"
echo "=============================================="
echo ""

# Define test queries
QUERIES=(
    "Apa syarat n ketentuan utk mendapatkan diskon di Gyukatsu Kyoto Katsugyu?"
    "Apa promo yang ditawarkan oleh Jakaichi Mart di Jakarta?"
    "Apa syrtnya n ketentuan untuk mendapatkan diskon 225rb di Traveloka?"
)

# Run each query twice (cold then warm)
for i in "${!QUERIES[@]}"; do
    QUERY="${QUERIES[$i]}"
    QUERY_NUM=$((i + 1))
    
    echo "=============================================="
    echo "QUERY $QUERY_NUM: ${QUERY:0:50}..."
    echo "=============================================="
    
    # First hit (cold)
    echo ""
    echo ">>> HIT 1 (Cold):"
    RESULT=$(curl -s -w "\n%{time_total}" \
      --location 'http://localhost:8000/api/v1/search/retrieval' \
      --header 'Content-Type: application/json' \
      --data "{
        \"query\": \"$QUERY\",
        \"top_k\": 50,
        \"search_type\": \"graph_completion_custom\"
      }")
    CURL_TIME_1=$(echo "$RESULT" | tail -1)
    echo "    Total Time: ${CURL_TIME_1}s"
    
    sleep 1
    
    # Second hit (warm)
    echo ">>> HIT 2 (Warm):"
    RESULT=$(curl -s -w "\n%{time_total}" \
      --location 'http://localhost:8000/api/v1/search/retrieval' \
      --header 'Content-Type: application/json' \
      --data "{
        \"query\": \"$QUERY\",
        \"top_k\": 50,
        \"search_type\": \"graph_completion_custom\"
      }")
    CURL_TIME_2=$(echo "$RESULT" | tail -1)
    echo "    Total Time: ${CURL_TIME_2}s"
    
    echo ""
done

# Show timing breakdown from logs
sleep 1
LOG_FILE=$(ls -t logs/*.log 2>/dev/null | head -1)

echo "=============================================="
echo "TIMING BREAKDOWN (last 15 entries from $LOG_FILE)"
echo "=============================================="
echo ""

echo "--- EMBED QUERY TIMES ---"
grep "⏱️ \[STEP 0\]" "$LOG_FILE" | tail -6

echo ""
echo "--- VECTOR SEARCH TIMES ---"
grep "⏱️ \[STEP 1\]" "$LOG_FILE" | tail -6

echo ""
echo "--- GRAPH PROJECTION TIMES ---"
grep "⏱️ \[STEP 3\]" "$LOG_FILE" | tail -6

echo ""
echo "--- RESULT MAPPING TIMES ---"
grep -E "⏱️ \[MAP_TRIPLETS" "$LOG_FILE" | tail -6

echo ""
echo "=============================================="
