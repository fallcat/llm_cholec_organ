#!/bin/bash

# Run separate scenarios (zero-shot and few-shot) for local models

echo "Running SEPARATE scenarios for local models..."
echo

echo "==================== Scenario 1: Zero-shot Separate ===================="
EVAL_DETECTION_MODE=separate EVAL_USE_FEWSHOT=false ./eval_bbox_local_single_scenario.sh

echo
echo "==================== Scenario 2: Few-shot Separate ===================="
EVAL_DETECTION_MODE=separate EVAL_USE_FEWSHOT=true ./eval_bbox_local_single_scenario.sh

echo
echo "Separate scenarios complete!"