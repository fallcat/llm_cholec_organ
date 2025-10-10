#!/usr/bin/env bash
set -euo pipefail

# Local LVLMs Only - Unbalanced 3 Seeds

REPO_ROOT="/shared_data0/weiqiuy/llm_cholec_organ"
PY="${REPO_ROOT}/notebooks_py/eval_bbox_balanced_unbalanced.py"
RESULTS="${REPO_ROOT}/results"

# Configuration
SPLIT="unbalanced"
SEEDS="42,7,2025"
FRAMES=200
MODELS=("Qwen/Qwen2.5-VL-7B-Instruct" "mistralai/Pixtral-12B-2409")

echo "===== Running Local LVLMs Only ====="
echo "Models: ${MODELS[*]}"
echo "Split: ${SPLIT}"
echo "Seeds: ${SEEDS}"
echo "Frames: ${FRAMES}"
echo "====================================="

mkdir -p "${RESULTS}/logs"

for MODEL in "${MODELS[@]}"; do
    # Create safe log filename
    SAFE_MODEL=$(echo "${MODEL}" | sed 's/\//_/g')
    LOG_FILE="${RESULTS}/logs/local_${SAFE_MODEL}_${SPLIT}_3seeds.log"
    
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
echo "===== Local LVLMs Complete ====="
echo "Results: ${RESULTS}"
echo "Logs: ${RESULTS}/logs/local_*"
echo "================================"