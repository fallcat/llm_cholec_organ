#!/usr/bin/env bash
set -euo pipefail

# Test script for all models - small sample (10 frames, 1 seed)
# This tests that all models work before running the full evaluation

REPO_ROOT="/shared_data0/weiqiuy/llm_cholec_organ"
PY="${REPO_ROOT}/notebooks_py/eval_bbox_balanced_unbalanced.py"
RESULTS="${REPO_ROOT}/results"

# Test parameters
SPLITS=("balanced" "unbalanced")
SEEDS="42"  # Just one seed for testing
FRAMES=10   # Just 10 frames for quick test

# All models to test
API_MODELS=("gpt-4.1" "gemini-2.0-flash")
LOCAL_MODELS=("Qwen/Qwen2.5-VL-7B-Instruct" "mistralai/Pixtral-12B-2409")
CLIP_MODELS=("raso" "peskavlp")
SEG_MODELS=("cholenet" "gonogonet")

ALL_MODELS=("${API_MODELS[@]}" "${LOCAL_MODELS[@]}" "${CLIP_MODELS[@]}" "${SEG_MODELS[@]}")

echo "=========================================="
echo "Testing All Models - Quick Validation"
echo "=========================================="
echo "Models to test: ${#ALL_MODELS[@]}"
echo "Splits: ${SPLITS[*]}"
echo "Seed: ${SEEDS}"
echo "Frames: ${FRAMES}"
echo ""

# Create results directory
mkdir -p "${RESULTS}/logs"

# Track successes and failures
declare -a PASSED=()
declare -a FAILED=()

# Test each model
for MODEL in "${ALL_MODELS[@]}"; do
    echo "----------------------------------------"
    echo "Testing ${MODEL}..."
    
    SUCCESS=true
    
    for SPLIT in "${SPLITS[@]}"; do
        LOG_FILE="${RESULTS}/logs/test_${MODEL}_${SPLIT}.log"
        
        echo -n "  ${SPLIT} split... "
        
        # Run without catching errors - let them propagate
        python "${PY}" \
            --model "${MODEL}" \
            --split "${SPLIT}" \
            --num-samples "${FRAMES}" \
            --seeds "${SEEDS}" \
            --output-dir "${RESULTS}"
        
        # Check if results were actually generated
        RESULT_DIR="${RESULTS}/bbox_cholecseg8k_local_${SPLIT}${FRAMES}_seed42"
        SUMMARY_FILE="${RESULT_DIR}/summary_combined_zeroshot.json"
        
        if [ -f "${SUMMARY_FILE}" ]; then
            # Extract metrics
            PRESENCE_ACC=$(python3 -c "
import json
with open('${SUMMARY_FILE}', 'r') as f:
    data = json.load(f)
    acc = data.get('metrics', {}).get('presence_accuracy', 'N/A')
    if isinstance(acc, (int, float)):
                print(f'{acc:.1%}')
    else:
        print('N/A')
" 2>/dev/null || echo "N/A")
            
            echo "✓ (Presence: ${PRESENCE_ACC})"
        else
            echo "✗ (No results generated)"
            SUCCESS=false
        fi
    done
    
    if [ "${SUCCESS}" = true ]; then
        PASSED+=("${MODEL}")
    else
        FAILED+=("${MODEL}")
    fi
done

# Summary report
echo ""
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo "Passed: ${#PASSED[@]}/${#ALL_MODELS[@]}"
if [ ${#PASSED[@]} -gt 0 ]; then
    echo "  ✓ ${PASSED[*]}"
fi

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo "Failed: ${#FAILED[@]}/${#ALL_MODELS[@]}"
    echo "  ✗ ${FAILED[*]}"
    echo ""
    echo "Check logs for failed models:"
    for MODEL in "${FAILED[@]}"; do
        echo "  - ${RESULTS}/logs/test_${MODEL}_*.log"
    done
fi

# Final status
echo ""
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "✅ All models passed! Ready for full evaluation."
    exit 0
else
    echo "❌ Some models failed. Fix issues before running full evaluation."
    exit 1
fi