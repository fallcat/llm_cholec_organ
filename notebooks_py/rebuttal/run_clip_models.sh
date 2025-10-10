#!/usr/bin/env bash
set -euo pipefail

# Configuration variables (edit these if needed)
REPO_ROOT="/shared_data0/weiqiuy/llm_cholec_organ"
PY="${REPO_ROOT}/notebooks_py/eval_bbox_balanced_unbalanced.py"
AGG="${REPO_ROOT}/notebooks_py/aggregate_bbox_results.py"
RESULTS="${REPO_ROOT}/results"
BALANCED_DIR="${REPO_ROOT}/data_info/cholecseg8k_local_balanced_200"
UNBAL_DIR="${REPO_ROOT}/data_info/cholecseg8k_local_unbalanced_200"

# Evaluation parameters
SPLITS=("unbalanced")
SEEDS="42,7,2025"
FRAMES=200
SKIP_EXISTING="true"

# CLIP encoder models
MODELS=("raso" "peskavlp")

echo "===== Running CLIP Models Evaluation ====="
echo "Models: ${MODELS[*]}"
echo "Splits: ${SPLITS[*]}"
echo "Seeds: ${SEEDS}"
echo "Frames per split: ${FRAMES}"
echo "Results directory: ${RESULTS}"
echo "=========================================="

# Ensure necessary directories exist
mkdir -p "${RESULTS}/logs" "${UNBAL_DIR}"

# Run evaluations for each model and split
for MODEL in "${MODELS[@]}"; do
    for SPLIT in "${SPLITS[@]}"; do
        LOG_FILE="${RESULTS}/logs/clip_models_${MODEL}_${SPLIT}_seeds.log"
        
        echo "Running ${MODEL} on ${SPLIT} split with seeds ${SEEDS}..."
        echo "Logging to: ${LOG_FILE}"
        
        python "${PY}" \
            --model "${MODEL}" \
            --split "${SPLIT}" \
            --num-samples "${FRAMES}" \
            --seeds "${SEEDS}" \
            --output-dir "${RESULTS}" \
            --skip-existing \
            > "${LOG_FILE}" 2>&1
        
        echo "Completed ${MODEL} on ${SPLIT} split"
    done
done

# Aggregate results
echo "Aggregating results..."
python "${AGG}" --results-root "${RESULTS}" --write-xlsx true

echo ""
echo "===== CLIP Models Evaluation Complete ====="
echo "Results saved to: ${RESULTS}"
echo "Logs saved to: ${RESULTS}/logs/clip_models_*"
echo "==========================================="