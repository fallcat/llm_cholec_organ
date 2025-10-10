#!/usr/bin/env bash
set -euo pipefail

# TEST SCRIPT - Small sample evaluation for quick testing
# This runs a tiny subset (10 samples, 2 seeds) for debugging

# Configuration variables
REPO_ROOT="/shared_data0/weiqiuy/llm_cholec_organ"
PY="${REPO_ROOT}/notebooks_py/eval_bbox_balanced_unbalanced.py"
AGG="${REPO_ROOT}/notebooks_py/aggregate_bbox_results.py"
RESULTS="${REPO_ROOT}/results"
BALANCED_DIR="${REPO_ROOT}/data_info/cholecseg8k_local_balanced_200"
UNBAL_DIR="${REPO_ROOT}/data_info/cholecseg8k_local_unbalanced_200"

# SMALL SAMPLE PARAMETERS FOR TESTING
SPLITS=("balanced" "unbalanced")
SEEDS="42,7"  # Only 2 seeds for testing
FRAMES=10     # Only 10 frames for quick test
SKIP_EXISTING="false"  # Don't skip to ensure fresh test

# Test with just one model
MODELS=("gpt-4.1")

echo "===== TEST RUN: Small Sample Evaluation ====="
echo "Models: ${MODELS[*]}"
echo "Splits: ${SPLITS[*]}"
echo "Seeds: ${SEEDS}"
echo "Frames per split: ${FRAMES} (TEST MODE)"
echo "Results directory: ${RESULTS}"
echo "============================================="

# Ensure necessary directories exist
mkdir -p "${RESULTS}/logs" "${UNBAL_DIR}"

# Run evaluations for each model and split
for MODEL in "${MODELS[@]}"; do
    for SPLIT in "${SPLITS[@]}"; do
        LOG_FILE="${RESULTS}/logs/test_small_${MODEL}_${SPLIT}_seeds.log"
        
        echo "Testing ${MODEL} on ${SPLIT} split with ${FRAMES} samples..."
        echo "Logging to: ${LOG_FILE}"
        
        python "${PY}" \
            --model "${MODEL}" \
            --split "${SPLIT}" \
            --num-samples "${FRAMES}" \
            --seeds "${SEEDS}" \
            --output-dir "${RESULTS}" \
            --skip-existing \
            > "${LOG_FILE}" 2>&1
        
        echo "Completed test for ${MODEL} on ${SPLIT} split"
    done
done

echo ""
echo "===== TEST RUN Complete ====="
echo "Results saved to: ${RESULTS}"
echo "Logs saved to: ${RESULTS}/logs/test_small_*"
echo ""
echo "Expected runtime: ~5-10 minutes (vs ~2 hours for full run)"
echo "=============================="