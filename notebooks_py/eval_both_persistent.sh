#!/bin/bash
# Script to evaluate both cell selection and pointing on the same samples
# Uses persistent directories to allow resuming and avoid re-evaluation

# Configuration
NUM_SAMPLES=20
MODELS='gpt-5-mini,claude-sonnet-4-20250514,gemini-2.5-flash'  # Specify models to evaluate
USE_CACHE=false  # Disable API cache to avoid collision issues
USE_PERSISTENT=true  # Use persistent directories

# Cell selection parameters
GRID_SIZE=3
TOP_K=1

echo "=========================================="
echo "Evaluation Configuration"
echo "=========================================="
echo "Number of samples: $NUM_SAMPLES"
echo "Models: $MODELS"
echo "Grid size: ${GRID_SIZE}x${GRID_SIZE}"
echo "Top-K: $TOP_K"
echo "Using persistent directories: $USE_PERSISTENT"
echo "API caching: $USE_CACHE"
echo ""

# Change to notebooks_py directory
cd /shared_data0/weiqiuy/llm_cholec_organ/notebooks_py

echo "=========================================="
echo "Step 1: Cell Selection Evaluation"
echo "=========================================="
echo "Output dir: ../results/cell_selection_G${GRID_SIZE}_K${TOP_K}/"
echo ""

# Run cell selection evaluation
EVAL_NUM_SAMPLES=$NUM_SAMPLES \
EVAL_MODELS=$MODELS \
EVAL_USE_CACHE=$USE_CACHE \
EVAL_PERSISTENT_DIR=$USE_PERSISTENT \
EVAL_GRID_SIZE=$GRID_SIZE \
EVAL_TOP_K=$TOP_K \
python3 eval_cell_selection_original_size.py

echo ""
echo "=========================================="
echo "Step 2: Pointing Evaluation"
echo "=========================================="
echo "Output dir: ../results/pointing_original/"
echo ""

# Run pointing evaluation on the same samples
EVAL_NUM_SAMPLES=$NUM_SAMPLES \
EVAL_MODELS=$MODELS \
EVAL_USE_CACHE=$USE_CACHE \
EVAL_PERSISTENT_DIR=$USE_PERSISTENT \
python3 eval_pointing_original_size.py

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
echo "Results saved to:"
echo "  - Cell selection: ../results/cell_selection_G${GRID_SIZE}_K${TOP_K}/"
echo "  - Pointing: ../results/pointing_original/"
echo ""
echo "To view results:"
echo "  - Check metrics files in each directory"
echo "  - Look for metrics_comparison.txt and metrics_comparison.json"