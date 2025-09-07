#!/bin/bash

# Run all 4 scenarios for local models

echo "Running all 4 scenarios for local models..."
echo

echo "==================== Scenario 1: Zero-shot Combined ===================="
EVAL_DETECTION_MODE=combined EVAL_USE_FEWSHOT=false ./eval_bbox_local_single_scenario.sh

echo
echo "==================== Scenario 2: Few-shot Combined ===================="
EVAL_DETECTION_MODE=combined EVAL_USE_FEWSHOT=true ./eval_bbox_local_single_scenario.sh

echo
echo "==================== Scenario 3: Zero-shot Separate ===================="
EVAL_DETECTION_MODE=separate EVAL_USE_FEWSHOT=false ./eval_bbox_local_single_scenario.sh

echo
echo "==================== Scenario 4: Few-shot Separate ===================="
EVAL_DETECTION_MODE=separate EVAL_USE_FEWSHOT=true ./eval_bbox_local_single_scenario.sh

echo
echo "All 4 scenarios complete!"