#!/bin/bash

# Comprehensive unified evaluation script for all datasets and scenarios
# Supports: cholecseg8k, cholec_organs, cholec_gonogo
# Scenarios: zero-shot/few-shot × combined/separate

echo "=========================================================================="
echo "UNIFIED BOUNDING BOX EVALUATION"
echo "=========================================================================="
echo

# Parse arguments
DATASET=${1:-"cholecseg8k"}      # Dataset name
NUM_SAMPLES=${2:-200}             # Number of samples
SCENARIOS=${3:-"zeroshot_combined"}  # Scenarios to run
MODEL_TYPE=${4:-"all"}            # Model type: api, local, all

# Validate dataset
if [[ ! "$DATASET" =~ ^(cholecseg8k|cholec_organs|cholec_gonogo|all)$ ]]; then
    echo "Error: Invalid dataset '$DATASET'"
    echo "Valid options: cholecseg8k, cholec_organs, cholec_gonogo, all"
    exit 1
fi

# Parse scenarios
case $SCENARIOS in
    "zeroshot")
        RUN_SCENARIOS=("zeroshot_combined" "zeroshot_separate")
        ;;
    "fewshot")
        RUN_SCENARIOS=("fewshot_combined" "fewshot_separate")
        ;;
    "combined")
        RUN_SCENARIOS=("zeroshot_combined" "fewshot_combined")
        ;;
    "separate")
        RUN_SCENARIOS=("zeroshot_separate" "fewshot_separate")
        ;;
    "all")
        RUN_SCENARIOS=("zeroshot_combined" "zeroshot_separate" "fewshot_combined" "fewshot_separate")
        ;;
    *)
        # Single scenario specified
        RUN_SCENARIOS=("$SCENARIOS")
        ;;
esac

# Set base environment variables
export EVAL_NUM_SAMPLES=$NUM_SAMPLES
export EVAL_USE_CACHE=true
export EVAL_PERSISTENT_DIR=true

echo "Configuration:"
echo "  Dataset(s): $DATASET"
echo "  Samples: $NUM_SAMPLES"
echo "  Scenarios: ${RUN_SCENARIOS[@]}"
echo "  Model type: $MODEL_TYPE"
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

# Determine which datasets to run
if [ "$DATASET" == "all" ]; then
    DATASETS=("cholecseg8k" "cholec_organs" "cholec_gonogo")
else
    DATASETS=("$DATASET")
fi

# Function to run a single evaluation
run_evaluation() {
    local dataset=$1
    local model=$2
    local scenario=$3
    
    # Parse scenario into detection mode and fewshot flag
    if [[ "$scenario" == *"combined"* ]]; then
        export EVAL_DETECTION_MODE="combined"
    else
        export EVAL_DETECTION_MODE="separate"
    fi
    
    if [[ "$scenario" == "fewshot"* ]]; then
        export EVAL_USE_FEWSHOT="true"
    else
        export EVAL_USE_FEWSHOT="false"
    fi
    
    export EVAL_DATASET="$dataset"
    export EVAL_MODEL="$model"
    
    echo "  → Running: $scenario"
    python3 eval_bbox_unified.py
    
    if [ $? -eq 0 ]; then
        echo "    ✓ Success"
    else
        echo "    ✗ Failed"
        return 1
    fi
    
    return 0
}

# Track statistics
TOTAL_RUNS=0
SUCCESSFUL_RUNS=0
FAILED_RUNS=0

# Run evaluations for each dataset
for dataset in "${DATASETS[@]}"; do
    echo
    echo "=========================================================================="
    echo "DATASET: ${dataset^^}"
    echo "=========================================================================="
    
    # Run API models if requested
    if [ "$MODEL_TYPE" == "api" ] || [ "$MODEL_TYPE" == "all" ]; then
        echo
        echo "--- API MODELS ---"
        for model in "${API_MODELS[@]}"; do
            echo
            echo "Model: $model"
            for scenario in "${RUN_SCENARIOS[@]}"; do
                ((TOTAL_RUNS++))
                if run_evaluation "$dataset" "$model" "$scenario"; then
                    ((SUCCESSFUL_RUNS++))
                else
                    ((FAILED_RUNS++))
                fi
            done
        done
    fi
    
    # Run local models if requested
    if [ "$MODEL_TYPE" == "local" ] || [ "$MODEL_TYPE" == "all" ]; then
        echo
        echo "--- LOCAL MODELS ---"
        for model in "${LOCAL_MODELS[@]}"; do
            echo
            echo "Model: $model"
            for scenario in "${RUN_SCENARIOS[@]}"; do
                ((TOTAL_RUNS++))
                if run_evaluation "$dataset" "$model" "$scenario"; then
                    ((SUCCESSFUL_RUNS++))
                else
                    ((FAILED_RUNS++))
                fi
            done
            
            # Clear GPU memory for local models
            if [ ${#RUN_SCENARIOS[@]} -gt 0 ]; then
                echo "  Waiting 10 seconds to clear GPU memory..."
                sleep 10
            fi
        done
    fi
done

# Summary
echo
echo "=========================================================================="
echo "EVALUATION SUMMARY"
echo "=========================================================================="
echo "Total runs: $TOTAL_RUNS"
echo "Successful: $SUCCESSFUL_RUNS"
echo "Failed: $FAILED_RUNS"
echo

# Print results locations
echo "Results saved to:"
for dataset in "${DATASETS[@]}"; do
    case $dataset in
        cholecseg8k)
            echo "  - CholecSeg8k: results/bbox_cholecseg8k_local_quick/"
            ;;
        cholec_organs)
            echo "  - CholecOrgans: results/bbox_cholec_organs_quick/"
            ;;
        cholec_gonogo)
            echo "  - CholecGoNoGo: results/bbox_cholec_gonogo_quick/"
            ;;
    esac
done

echo
echo "Usage examples:"
echo "  $0 cholecseg8k 200 zeroshot_combined api    # Single scenario, API models"
echo "  $0 cholec_organs 200 all local              # All scenarios, local models"
echo "  $0 all 100 zeroshot all                     # All datasets, zero-shot only"
echo "  $0 cholec_gonogo 200 combined api           # Combined modes only"
echo
echo "=========================================================================="
echo "Completed at: $(date)"
echo "=========================================================================="