#!/usr/bin/env bash
set -euo pipefail

# API Models Only - Unbalanced 3 Seeds

REPO_ROOT="/shared_data0/weiqiuy/llm_cholec_organ"
PY="${REPO_ROOT}/notebooks_py/eval_bbox_balanced_unbalanced.py"
RESULTS="${REPO_ROOT}/results"

# Configuration
SPLIT="unbalanced"
SEEDS="42,7,2025"
FRAMES=200
MODELS=("gpt-4.1" "gemini-2.0-flash")

echo "===== Running API Models Only ====="
echo "Models: ${MODELS[*]}"
echo "Split: ${SPLIT}"
echo "Seeds: ${SEEDS}"
echo "Frames: ${FRAMES}"
echo "===================================="

mkdir -p "${RESULTS}/logs"

for MODEL in "${MODELS[@]}"; do
    LOG_FILE="${RESULTS}/logs/api_${MODEL}_${SPLIT}_3seeds.log"
    
    echo "Running ${MODEL}..."
    echo "Logging to: ${LOG_FILE}"
    
    python "${PY}" \
        --model "${MODEL}" \
        --split "${SPLIT}" \
        --num-samples "${FRAMES}" \
        --seeds "${SEEDS}" \
        --output-dir "${RESULTS}" \
        --skip-existing \
        > "${LOG_FILE}" 2>&1
    
    echo "✓ Completed ${MODEL}"
done

echo ""
echo "===== API Models Complete ====="
echo "Results: ${RESULTS}"
echo "Logs: ${RESULTS}/logs/api_*"
echo "==============================="