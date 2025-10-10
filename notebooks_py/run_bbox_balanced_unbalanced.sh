#!/bin/bash

# Batch script for balanced vs unbalanced bbox evaluation with multi-seed support
# This script runs evaluations for specified models across different configurations

echo "=========================================================================="
echo "BALANCED VS UNBALANCED BBOX EVALUATION"
echo "=========================================================================="
echo "Starting at: $(date)"
echo

# Configuration
NUM_SAMPLES=${1:-200}           # Default to 200 samples (full test set)
SPLIT_MODE=${2:-both}            # balanced, unbalanced, or both
SEEDS=${3:-"42,7,2025,518,1337"} # Default 5 seeds
SKIP_API=${4:-false}             # Set to "true" to skip API models
SKIP_LOCAL=${5:-false}           # Set to "true" to skip local models
SKIP_EXISTING=${6:-true}         # Skip already computed experiments

echo "Configuration:"
echo "  Samples: $NUM_SAMPLES"
echo "  Split mode: $SPLIT_MODE"
echo "  Seeds: $SEEDS"
echo "  Skip API models: $SKIP_API"
echo "  Skip local models: $SKIP_LOCAL"
echo "  Skip existing: $SKIP_EXISTING"
echo

# Create log directory
LOG_DIR="logs/bbox_balanced_unbalanced_$(date +%Y%m%d_%H%M%S)"
mkdir -p $LOG_DIR

# Function to run and log
run_with_log() {
    local cmd=$1
    local log_file=$2
    echo "Running: $cmd"
    echo "Logging to: $log_file"
    eval $cmd 2>&1 | tee $log_file
}

# Function to run evaluation for a specific configuration
run_evaluation() {
    local model=$1
    local split=$2
    local detection_mode=$3
    local use_fewshot=$4
    local description=$5
    
    echo
    echo "--- $description ---"
    echo "Model: $model"
    echo "Split: $split"
    echo "Detection: $detection_mode"
    echo "Few-shot: $use_fewshot"
    
    # Build command
    cmd="python3 eval_bbox_balanced_unbalanced.py"
    cmd="$cmd --model '$model'"
    cmd="$cmd --split $split"
    cmd="$cmd --seeds '$SEEDS'"
    cmd="$cmd --num-samples $NUM_SAMPLES"
    cmd="$cmd --detection-mode $detection_mode"
    
    if [ "$use_fewshot" == "true" ]; then
        cmd="$cmd --use-fewshot"
    fi
    
    if [ "$SKIP_EXISTING" == "true" ]; then
        cmd="$cmd --skip-existing"
    fi
    
    # Create log filename
    log_name="${model//\//_}_${split}_${detection_mode}_${use_fewshot}.log"
    
    # Run evaluation
    run_with_log "$cmd" "$LOG_DIR/$log_name"
}

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

# Determine which splits to run
if [ "$SPLIT_MODE" == "both" ]; then
    SPLITS=("balanced" "unbalanced")
elif [ "$SPLIT_MODE" == "balanced" ] || [ "$SPLIT_MODE" == "unbalanced" ]; then
    SPLITS=("$SPLIT_MODE")
else
    echo "Invalid split mode: $SPLIT_MODE"
    exit 1
fi

# ============================================================================
# PART 1: API MODELS
# ============================================================================
if [ "$SKIP_API" != "true" ]; then
    echo
    echo "=========================================================================="
    echo "PART 1: API MODELS"
    echo "=========================================================================="
    
    for model in "${API_MODELS[@]}"; do
        for split in "${SPLITS[@]}"; do
            # All 4 scenarios
            run_evaluation "$model" "$split" "combined" "false" \
                "$model - $split - Zero-shot Combined"
            
            run_evaluation "$model" "$split" "combined" "true" \
                "$model - $split - Few-shot Combined"
            
            run_evaluation "$model" "$split" "separate" "false" \
                "$model - $split - Zero-shot Separate"
            
            run_evaluation "$model" "$split" "separate" "true" \
                "$model - $split - Few-shot Separate"
        done
    done
else
    echo "Skipping API models (SKIP_API=true)"
fi

# ============================================================================
# PART 2: LOCAL MODELS
# ============================================================================
if [ "$SKIP_LOCAL" != "true" ]; then
    echo
    echo "=========================================================================="
    echo "PART 2: LOCAL MODELS"
    echo "=========================================================================="
    
    for model in "${LOCAL_MODELS[@]}"; do
        for split in "${SPLITS[@]}"; do
            # All 4 scenarios
            run_evaluation "$model" "$split" "combined" "false" \
                "$model - $split - Zero-shot Combined"
            
            run_evaluation "$model" "$split" "combined" "true" \
                "$model - $split - Few-shot Combined"
            
            run_evaluation "$model" "$split" "separate" "false" \
                "$model - $split - Zero-shot Separate"
            
            run_evaluation "$model" "$split" "separate" "true" \
                "$model - $split - Few-shot Separate"
            
            # Wait to clear GPU memory for local models
            if [ "$split" == "${SPLITS[-1]}" ]; then
                echo "Waiting 10 seconds to clear GPU memory..."
                sleep 10
            fi
        done
    done
else
    echo "Skipping local models (SKIP_LOCAL=true)"
fi

# ============================================================================
# PART 3: AGGREGATION
# ============================================================================
echo
echo "=========================================================================="
echo "PART 3: AGGREGATING RESULTS"
echo "=========================================================================="

# Run aggregation script if multiple seeds were used
if [[ "$SEEDS" == *","* ]]; then
    echo "Running multi-seed aggregation..."
    python3 aggregate_bbox_results.py \
        --results-dir "/shared_data0/weiqiuy/llm_cholec_organ/results" \
        --split-modes "$SPLIT_MODE" \
        --seeds "$SEEDS" \
        --num-samples $NUM_SAMPLES
else
    echo "Single seed - skipping aggregation"
fi

# ============================================================================
# COMPLETION
# ============================================================================
echo
echo "=========================================================================="
echo "EVALUATION COMPLETE"
echo "=========================================================================="
echo "Completed at: $(date)"
echo
echo "Results saved to:"
echo "  - Individual: results/bbox_cholecseg8k_local_{split}{samples}_seed{seed}/"
echo "  - Logs: $LOG_DIR/"
echo
echo "Experiments run:"

# Count experiments
n_models=0
if [ "$SKIP_API" != "true" ]; then
    n_models=$((n_models + ${#API_MODELS[@]}))
fi
if [ "$SKIP_LOCAL" != "true" ]; then
    n_models=$((n_models + ${#LOCAL_MODELS[@]}))
fi

n_splits=${#SPLITS[@]}
n_scenarios=4  # (zero/few-shot) x (combined/separate)
n_seeds=$(echo "$SEEDS" | tr ',' '\n' | wc -l)
total_experiments=$((n_models * n_splits * n_scenarios * n_seeds))

echo "  - Models: $n_models"
echo "  - Splits: $n_splits"
echo "  - Scenarios: $n_scenarios"
echo "  - Seeds: $n_seeds"
echo "  - TOTAL: $total_experiments experiments"
echo "=========================================================================="