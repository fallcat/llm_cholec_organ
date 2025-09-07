#!/bin/bash

# Test script for naive baseline evaluation
# Always predicts the entire image as bounding box

echo "Running Naive Baseline Evaluation"
echo "================================="
echo

# Configuration
NUM_SAMPLES=${1:-200}  # Default to 200 samples
PRESENCE_MODE=${2:-"perfect"}  # Options: "perfect", "all", "random"
DATASET=${3:-"both"}  # Options: "cholecseg8k", "cholec_organs", "both"

echo "Configuration:"
echo "  Samples: $NUM_SAMPLES"
echo "  Presence mode: $PRESENCE_MODE"
echo "  Dataset: $DATASET"
echo

# Export environment variables
export EVAL_NUM_SAMPLES=$NUM_SAMPLES
export EVAL_PRESENCE_MODE=$PRESENCE_MODE
export EVAL_DATASET=$DATASET

# Run the naive baseline
python3 eval_bbox_naive_baseline.py

echo
echo "Naive baseline evaluation complete!"