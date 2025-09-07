#!/bin/bash

# Run all 4 scenarios for API models on cholec_gonogo dataset

echo "Running all 4 scenarios for API models on CHOLEC_GONOGO dataset..."
echo

# Set default samples if not provided
NUM_SAMPLES=${1:-200}
export EVAL_NUM_SAMPLES=$NUM_SAMPLES
export EVAL_USE_CACHE=true
export EVAL_PERSISTENT_DIR=true

echo "Using $NUM_SAMPLES samples per evaluation"
echo

# API Models to evaluate
API_MODELS=(
    "gpt-4.1"
    "claude-sonnet-4-20250514"
    "gemini-2.0-flash"
)

for model in "${API_MODELS[@]}"; do
    export EVAL_MODEL="$model"
    
    echo "==================== Model: $model ===================="
    
    echo "--- Scenario 1: Zero-shot Combined ---"
    EVAL_DETECTION_MODE=combined EVAL_USE_FEWSHOT=false python3 eval_bbox_cholec_gonogo.py
    
    echo
    echo "--- Scenario 2: Few-shot Combined ---"
    EVAL_DETECTION_MODE=combined EVAL_USE_FEWSHOT=true python3 eval_bbox_cholec_gonogo.py
    
    echo
    echo "--- Scenario 3: Zero-shot Separate ---"
    EVAL_DETECTION_MODE=separate EVAL_USE_FEWSHOT=false python3 eval_bbox_cholec_gonogo.py
    
    echo
    echo "--- Scenario 4: Few-shot Separate ---"
    EVAL_DETECTION_MODE=separate EVAL_USE_FEWSHOT=true python3 eval_bbox_cholec_gonogo.py
    
    echo
done

echo "All cholec_gonogo API evaluations complete!"
echo "Results saved in: results/bbox_cholec_gonogo_quick/"