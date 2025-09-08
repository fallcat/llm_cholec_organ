#!/bin/bash

# Run batch evaluation for CholeNet and GoNoGoNet on all datasets
# This uses the integrated batch mode in eval_bbox_unified.py

echo "=========================================="
echo "CholeNet and GoNoGoNet Batch Evaluation"
echo "=========================================="
echo ""

# Configuration
export EVAL_BATCH_MODE=true
export EVAL_NUM_SAMPLES=20  # Adjust for testing or full evaluation
export EVAL_USE_CACHE=true
export EVAL_PERSISTENT_DIR=true
export EVAL_DETECTION_MODE=combined
export EVAL_USE_FEWSHOT=false

# Run the unified evaluation script in batch mode
python notebooks_py/eval_bbox_unified.py

echo ""
echo "Evaluation complete!"
echo "Check the summary table above for results."