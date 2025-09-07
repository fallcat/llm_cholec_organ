#!/bin/bash

# Unified script to run zero-shot combined evaluation for any dataset
# Supports: cholecseg8k, cholec_organs, cholec_gonogo

echo "=========================================================================="
echo "UNIFIED ZERO-SHOT COMBINED EVALUATION"
echo "=========================================================================="
echo

# Configuration
DATASET=${1:-"cholecseg8k"}  # Default dataset
NUM_SAMPLES=${2:-200}        # Default samples
MODEL_TYPE=${3:-"all"}       # Options: "api", "local", "all"

# Validate dataset
if [[ ! "$DATASET" =~ ^(cholecseg8k|cholec_organs|cholec_gonogo)$ ]]; then
    echo "Error: Invalid dataset '$DATASET'"
    echo "Valid options: cholecseg8k, cholec_organs, cholec_gonogo"
    exit 1
fi

# Set environment variables
export EVAL_DATASET=$DATASET
export EVAL_NUM_SAMPLES=$NUM_SAMPLES
export EVAL_USE_CACHE=true
export EVAL_PERSISTENT_DIR=true
export EVAL_DETECTION_MODE=combined
export EVAL_USE_FEWSHOT=false

echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Samples: $NUM_SAMPLES"
echo "  Model type: $MODEL_TYPE"
echo "  Scenario: Zero-shot Combined ONLY"
echo

# Define models
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

# Function to run evaluation
run_model() {
    local model=$1
    export EVAL_MODEL="$model"
    
    echo "--- Model: $model ---"
    python3 eval_bbox_unified.py
    
    if [ $? -eq 0 ]; then
        echo "✓ Completed $model"
    else
        echo "✗ Failed $model"
    fi
}

# Run API models if requested
if [ "$MODEL_TYPE" == "api" ] || [ "$MODEL_TYPE" == "all" ]; then
    echo
    echo "==================== API MODELS ===================="
    for model in "${API_MODELS[@]}"; do
        echo
        run_model "$model"
    done
fi

# Run local models if requested
if [ "$MODEL_TYPE" == "local" ] || [ "$MODEL_TYPE" == "all" ]; then
    echo
    echo "==================== LOCAL MODELS ===================="
    for model in "${LOCAL_MODELS[@]}"; do
        echo
        run_model "$model"
        
        # Add delay for GPU memory clearing
        echo "Waiting 10 seconds to clear GPU memory..."
        sleep 10
    done
fi

# Results summary
echo
echo "==================== COMPLETE ===================="
echo "Zero-shot Combined evaluation complete for $DATASET!"
echo

# Determine results directory
case $DATASET in
    cholecseg8k)
        RESULTS_DIR="bbox_cholecseg8k_local_quick"
        ;;
    cholec_organs)
        RESULTS_DIR="bbox_cholec_organs_quick"
        ;;
    cholec_gonogo)
        RESULTS_DIR="bbox_cholec_gonogo_quick"
        ;;
esac

echo "Results saved in: results/${RESULTS_DIR}/zeroshot_combined/"
echo
echo "To view results:"
echo "  ls -la results/${RESULTS_DIR}/zeroshot_combined/"
echo
echo "To run different configurations:"
echo "  $0 <dataset> <samples> <model_type>"
echo "  Examples:"
echo "    $0 cholec_organs 200 api      # CholecOrgans, API models only"
echo "    $0 cholec_gonogo 200 local    # CholecGoNoGo, local models only"
echo "    $0 cholecseg8k 100 all        # CholecSeg8k, all models, 100 samples"
echo "=========================================================================="