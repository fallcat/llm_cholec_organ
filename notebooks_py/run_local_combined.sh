#!/bin/bash

# Run combined scenarios (zero-shot and few-shot) for local models

echo "Running COMBINED scenarios for local models..."
echo

echo "==================== Scenario 1: Zero-shot Combined ===================="
EVAL_DETECTION_MODE=combined EVAL_USE_FEWSHOT=false ./eval_bbox_local_single_scenario.sh

echo
echo "==================== Scenario 2: Few-shot Combined ===================="
EVAL_DETECTION_MODE=combined EVAL_USE_FEWSHOT=true ./eval_bbox_local_single_scenario.sh

echo
echo "Combined scenarios complete!"