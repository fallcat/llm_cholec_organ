#!/bin/bash

# Full evaluation script for CholeNet and GoNoGoNet on all datasets
# This runs on ALL test examples (not just a subset)

echo "=========================================="
echo "CholeNet and GoNoGoNet Full Evaluation"
echo "=========================================="
echo "Starting at: $(date)"
echo ""

# Configuration for FULL evaluation
export EVAL_BATCH_MODE=true
export EVAL_BATCH_MODELS="cholenet,gonogonet"
export EVAL_BATCH_DATASETS="cholecseg8k,cholec_organs,cholec_gonogo"
export EVAL_NUM_SAMPLES=200  # Use all 200 test samples
export EVAL_USE_CACHE=true
export EVAL_PERSISTENT_DIR=true
export EVAL_DETECTION_MODE=combined
export EVAL_USE_FEWSHOT=false

echo "Configuration:"
echo "  Models: cholenet, gonogonet"
echo "  Datasets: cholecseg8k, cholec_organs, cholec_gonogo"
echo "  Samples: 200 (full test set)"
echo "  Detection mode: combined"
echo "  Cache: enabled"
echo "  Output: persistent directories"
echo ""

# Run the unified evaluation script in batch mode
python notebooks_py/eval_bbox_unified.py

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
echo "Finished at: $(date)"
echo ""
echo "Results saved in:"
echo "  - /shared_data0/weiqiuy/llm_cholec_organ/results/bbox_cholecseg8k_local_quick/"
echo "  - /shared_data0/weiqiuy/llm_cholec_organ/results/bbox_cholec_organs_quick/"
echo "  - /shared_data0/weiqiuy/llm_cholec_organ/results/bbox_cholec_gonogo_quick/"
echo ""
echo "Check the summary table above for performance metrics."
