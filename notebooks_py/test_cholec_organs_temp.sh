#!/bin/bash

# Test script for cholec_organs with TEMPORARY output directory (not persistent)
# Tests ALL models with configurable scenarios

echo "Running test evaluation with TEMPORARY output folder..."
echo

# Set test configuration
NUM_SAMPLES=${1:-5}  # Default to 5 samples for quick testing
TEST_MODE=${2:-"zeroshot"}  # Options: "zeroshot", "fewshot", "all"
export EVAL_NUM_SAMPLES=$NUM_SAMPLES
export EVAL_USE_CACHE=true
export EVAL_PERSISTENT_DIR=false  # This creates timestamped folders instead

echo "Configuration:"
echo "  Samples: $NUM_SAMPLES"
echo "  Test mode: $TEST_MODE"
echo "  Output: Timestamped folders"
echo

# Define all models
API_MODELS=(
    "gpt-4.1"
    "claude-sonnet-4-20250514"
    "gemini-2.0-flash"
)

LOCAL_MODELS=(
    "llava-hf/llava-v1.6-mistral-7b-hf"
    "Qwen/Qwen2.5-VL-7B-Instruct"
    "mistralai/Pixtral-12B-2409"
)

# Combine all models
ALL_MODELS=("${API_MODELS[@]}" "${LOCAL_MODELS[@]}")

echo "Testing ${#ALL_MODELS[@]} models..."
echo

# Test each model
for model in "${ALL_MODELS[@]}"; do
    export EVAL_MODEL="$model"
    
    echo "==================== Model: $model ===================="
    
    if [ "$TEST_MODE" == "zeroshot" ] || [ "$TEST_MODE" == "all" ]; then
        echo "--- Zero-shot Combined ---"
        EVAL_DETECTION_MODE=combined EVAL_USE_FEWSHOT=false python3 eval_bbox_cholec_organs.py
        echo
    fi
    
    if [ "$TEST_MODE" == "fewshot" ] || [ "$TEST_MODE" == "all" ]; then
        echo "--- Few-shot Combined ---"
        EVAL_DETECTION_MODE=combined EVAL_USE_FEWSHOT=true python3 eval_bbox_cholec_organs.py
        echo
    fi
    
    if [ "$TEST_MODE" == "all" ]; then
        echo "--- Zero-shot Separate ---"
        EVAL_DETECTION_MODE=separate EVAL_USE_FEWSHOT=false python3 eval_bbox_cholec_organs.py
        echo
        
        echo "--- Few-shot Separate ---"
        EVAL_DETECTION_MODE=separate EVAL_USE_FEWSHOT=true python3 eval_bbox_cholec_organs.py
        echo
    fi
    
    # Add small delay for local models to clear GPU memory
    if [[ " ${LOCAL_MODELS[@]} " =~ " ${model} " ]]; then
        echo "Waiting 5 seconds to clear GPU memory..."
        sleep 5
    fi
    
    echo
done

echo "All tests complete! Results saved in timestamped folders under results/"