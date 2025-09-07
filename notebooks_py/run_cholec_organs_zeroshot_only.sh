#!/bin/bash

# Run ONLY zero-shot combined for CholecOrgans dataset
# This is faster than running all 4 scenarios

echo "Running Zero-shot Combined ONLY for CHOLEC_ORGANS dataset..."
echo

# Set default samples if not provided
NUM_SAMPLES=${1:-200}
MODEL_TYPE=${2:-"all"}  # Options: "api", "local", "all"

export EVAL_NUM_SAMPLES=$NUM_SAMPLES
export EVAL_USE_CACHE=true
export EVAL_PERSISTENT_DIR=true
export EVAL_DETECTION_MODE=combined
export EVAL_USE_FEWSHOT=false

echo "Configuration:"
echo "  Samples: $NUM_SAMPLES"
echo "  Model type: $MODEL_TYPE"
echo "  Scenario: Zero-shot Combined ONLY"
echo

# API Models
API_MODELS=(
    "gpt-4.1"
    "claude-sonnet-4-20250514"
    "gemini-2.0-flash"
)

# Local Models
LOCAL_MODELS=(
    "llava-hf/llava-v1.6-mistral-7b-hf"
    "Qwen/Qwen2.5-VL-7B-Instruct"
    "mistralai/Pixtral-12B-2409"
)

# Run API models if requested
if [ "$MODEL_TYPE" == "api" ] || [ "$MODEL_TYPE" == "all" ]; then
    echo "==================== API MODELS ===================="
    for model in "${API_MODELS[@]}"; do
        export EVAL_MODEL="$model"
        echo
        echo "--- Model: $model ---"
        echo "Running Zero-shot Combined..."
        python3 eval_bbox_cholec_organs.py
        echo "✓ Completed $model"
    done
fi

# Run local models if requested
if [ "$MODEL_TYPE" == "local" ] || [ "$MODEL_TYPE" == "all" ]; then
    echo
    echo "==================== LOCAL MODELS ===================="
    for model in "${LOCAL_MODELS[@]}"; do
        export EVAL_MODEL="$model"
        echo
        echo "--- Model: $model ---"
        echo "Running Zero-shot Combined..."
        python3 eval_bbox_cholec_organs.py
        echo "✓ Completed $model"
        
        echo "Waiting 10 seconds to clear GPU memory..."
        sleep 10
    done
fi

echo
echo "==================== COMPLETE ===================="
echo "Zero-shot Combined evaluation complete for CholecOrgans!"
echo "Results saved in: results/bbox_cholec_organs_quick/zeroshot_combined/"
echo
echo "To view results, check the individual model directories or run:"
echo "  ls -la results/bbox_cholec_organs_quick/zeroshot_combined/"