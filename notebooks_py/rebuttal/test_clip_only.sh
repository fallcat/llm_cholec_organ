#!/usr/bin/env bash
set -euo pipefail

# Test CLIP Models Only - Quick validation with small sample

REPO_ROOT="/shared_data0/weiqiuy/llm_cholec_organ"
PY="${REPO_ROOT}/notebooks_py/eval_bbox_balanced_unbalanced.py"
RESULTS="${REPO_ROOT}/results"

# Test configuration - small sample for quick validation
SPLIT="unbalanced"
SEEDS="42"  # Just one seed for testing
FRAMES=10   # Just 10 frames for quick test
MODELS=("raso" "peskavlp")

echo "===== Testing CLIP Models Only ====="
echo "Models: ${MODELS[*]}"
echo "Split: ${SPLIT}"
echo "Seed: ${SEEDS}"
echo "Frames: ${FRAMES} (test mode)"
echo "====================================="

mkdir -p "${RESULTS}/logs"

# Track successes and failures
declare -a PASSED=()
declare -a FAILED=()

for MODEL in "${MODELS[@]}"; do
    LOG_FILE="${RESULTS}/logs/test_clip_${MODEL}_${SPLIT}.log"
    
    echo -n "Testing ${MODEL}... "
    
    # Run without hiding errors for testing
    if python "${PY}" \
        --model "${MODEL}" \
        --split "${SPLIT}" \
        --num-samples "${FRAMES}" \
        --seeds "${SEEDS}" \
        --output-dir "${RESULTS}"; then
        
        echo "✓ PASSED"
        PASSED+=("${MODEL}")
    else
        echo "✗ FAILED"
        FAILED+=("${MODEL}")
    fi
done

echo ""
echo "===== CLIP Models Test Summary ====="
echo "Passed: ${#PASSED[@]}/2"
if [ ${#PASSED[@]} -gt 0 ]; then
    echo "  ✓ ${PASSED[*]}"
fi

if [ ${#FAILED[@]} -gt 0 ]; then
    echo "Failed: ${#FAILED[@]}/2"
    echo "  ✗ ${FAILED[*]}"
fi
echo "====================================="

# Exit with error if any failed
if [ ${#FAILED[@]} -gt 0 ]; then
    exit 1
fi