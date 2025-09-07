#!/bin/bash

# Bounding Box Evaluation - API Models - Single Scenario
# This script runs bbox evaluation for all API models with a single configuration
# The configuration is controlled via environment variables

echo "=========================================="
echo "BOUNDING BOX EVALUATION - API MODELS - SINGLE SCENARIO"
echo "=========================================="
echo

# Configuration from environment or defaults
NUM_SAMPLES=${EVAL_NUM_SAMPLES:-200}  # Default to 200 samples
USE_CACHE=${EVAL_USE_CACHE:-true}     # Default to using cache
DETECTION_MODE=${EVAL_DETECTION_MODE:-combined}  # combined or separate
USE_FEWSHOT=${EVAL_USE_FEWSHOT:-false}  # true or false
USE_PERSISTENT_DIR=${EVAL_PERSISTENT_DIR:-true}  # Use persistent output dir

# Determine scenario name
if [ "$USE_FEWSHOT" = "true" ]; then
    SHOT_TYPE="Few-shot"
else
    SHOT_TYPE="Zero-shot"
fi
SCENARIO_NAME="$SHOT_TYPE ${DETECTION_MODE^}"

echo "Configuration:"
echo "  Scenario: $SCENARIO_NAME"
echo "  Samples: $NUM_SAMPLES"
echo "  Cache: $USE_CACHE"
echo "  Detection mode: $DETECTION_MODE"
echo "  Few-shot: $USE_FEWSHOT"
echo "  Persistent directory: $USE_PERSISTENT_DIR"
echo

# API Models to evaluate
API_MODELS=(
    "gpt-4.1"
    "claude-sonnet-4-20250514"
    "gemini-2.0-flash"
)

# Display names for models
declare -A MODEL_NAMES
MODEL_NAMES["gpt-4.1"]="GPT-4.1"
MODEL_NAMES["claude-sonnet-4-20250514"]="Claude Sonnet 4"
MODEL_NAMES["gemini-2.0-flash"]="Gemini 2.0 Flash"

echo "Models to evaluate:"
for model in "${API_MODELS[@]}"; do
    echo "  - ${MODEL_NAMES[$model]}"
done
echo

# Track timing
START_TIME=$(date +%s)

# Results tracking
SUCCESS_COUNT=0
FAIL_COUNT=0
FAILED_MODELS=()

# Run evaluation for each API model
for model in "${API_MODELS[@]}"; do
    display_name="${MODEL_NAMES[$model]}"
    
    echo "=========================================="
    echo "Evaluating: $display_name"
    echo "=========================================="
    
    # Export all necessary environment variables
    export EVAL_MODEL="$model"
    export EVAL_NUM_SAMPLES="$NUM_SAMPLES"
    export EVAL_USE_CACHE="$USE_CACHE"
    export EVAL_DETECTION_MODE="$DETECTION_MODE"
    export EVAL_USE_FEWSHOT="$USE_FEWSHOT"
    export EVAL_PERSISTENT_DIR="$USE_PERSISTENT_DIR"
    
    # Run the evaluation
    if python3 eval_bbox_quick_test.py; then
        echo "✅ Successfully evaluated $display_name"
        ((SUCCESS_COUNT++))
    else
        echo "❌ Failed to evaluate $display_name"
        ((FAIL_COUNT++))
        FAILED_MODELS+=("$display_name")
    fi
    
    echo
done

# Calculate duration
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

# Summary
echo "=========================================="
echo "EVALUATION SUMMARY"
echo "=========================================="
echo "Scenario: $SCENARIO_NAME"
echo "Total models: ${#API_MODELS[@]}"
echo "Successful: $SUCCESS_COUNT"
echo "Failed: $FAIL_COUNT"
if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
    echo "Failed models: ${FAILED_MODELS[*]}"
fi
echo "Total time: ${MINUTES}m ${SECONDS}s"
echo

# Results location
if [ "$USE_PERSISTENT_DIR" = "true" ]; then
    echo "Results saved in: results/bbox_cholecseg8k_local_quick/"
else
    echo "Results saved in: results/bbox_cholecseg8k_local_*/"
fi

echo "Summary files:"
echo "  - summary_${DETECTION_MODE}_$([ "$USE_FEWSHOT" = "true" ] && echo "fewshot" || echo "zeroshot").json"
echo

if [ $FAIL_COUNT -eq 0 ]; then
    echo "✨ All API models evaluated successfully!"
    exit 0
else
    echo "⚠️  Some models failed evaluation."
    exit 1
fi