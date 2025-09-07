#!/bin/bash

# Run all 4 scenarios for local models on cholec_organs dataset

echo "Running all 4 scenarios for local models on CHOLEC_ORGANS dataset..."
echo

# Set default samples if not provided
NUM_SAMPLES=${1:-200}
export EVAL_NUM_SAMPLES=$NUM_SAMPLES
export EVAL_USE_CACHE=true
export EVAL_PERSISTENT_DIR=true

echo "Using $NUM_SAMPLES samples per evaluation"
echo

# Local Models to evaluate
LOCAL_MODELS=(
    "llava-hf/llava-v1.6-mistral-7b-hf"
    "Qwen/Qwen2.5-VL-7B-Instruct"
    "mistralai/Pixtral-12B-2409"
)

for model in "${LOCAL_MODELS[@]}"; do
    export EVAL_MODEL="$model"
    
    echo "==================== Model: $model ===================="
    
    echo "--- Scenario 1: Zero-shot Combined ---"
    EVAL_DETECTION_MODE=combined EVAL_USE_FEWSHOT=false python3 eval_bbox_cholec_organs.py
    
    echo
    echo "--- Scenario 2: Few-shot Combined ---"
    EVAL_DETECTION_MODE=combined EVAL_USE_FEWSHOT=true python3 eval_bbox_cholec_organs.py
    
    echo
    echo "--- Scenario 3: Zero-shot Separate ---"
    EVAL_DETECTION_MODE=separate EVAL_USE_FEWSHOT=false python3 eval_bbox_cholec_organs.py
    
    echo
    echo "--- Scenario 4: Few-shot Separate ---"
    EVAL_DETECTION_MODE=separate EVAL_USE_FEWSHOT=true python3 eval_bbox_cholec_organs.py
    
    echo
    echo "Waiting 10 seconds to clear GPU memory..."
    sleep 10
done

echo "All cholec_organs local evaluations complete!"
echo "Results saved in: results/bbox_cholec_organs_quick/"