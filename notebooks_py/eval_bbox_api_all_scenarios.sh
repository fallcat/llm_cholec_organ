#!/bin/bash

# Bounding Box Evaluation - API Models - All 4 Scenarios
# This script runs all 4 scenarios for API models sequentially:
# 1. Zero-shot Combined
# 2. Few-shot Combined
# 3. Zero-shot Separate
# 4. Few-shot Separate

set -e  # Exit on error

echo "=========================================="
echo "BOUNDING BOX EVALUATION - API MODELS - ALL SCENARIOS"
echo "=========================================="
echo

# Configuration
NUM_SAMPLES=${1:-200}  # Default to 200 samples, can override with first argument
USE_CACHE=${2:-true}   # Default to using cache
USE_PERSISTENT_DIR=${3:-true}  # Default to persistent directory

echo "Configuration:"
echo "  Samples per model: $NUM_SAMPLES"
echo "  Cache: $USE_CACHE"
echo "  Output: $([ "$USE_PERSISTENT_DIR" = "true" ] && echo "persistent" || echo "timestamped")"
echo "  Scenarios: 4 (Zero/Few-shot × Combined/Separate)"
echo

# Define the 4 scenarios
declare -a SCENARIOS=(
    "combined:false:Zero-shot Combined"
    "combined:true:Few-shot Combined"
    "separate:false:Zero-shot Separate"
    "separate:true:Few-shot Separate"
)

echo "Scenarios to run:"
for scenario in "${SCENARIOS[@]}"; do
    IFS=':' read -r mode fewshot name <<< "$scenario"
    echo "  - $name"
done
echo

# Track overall timing
OVERALL_START=$(date +%s)

# Track results
SCENARIO_COUNT=0
FAILED_SCENARIOS=()

# Run each scenario
for scenario in "${SCENARIOS[@]}"; do
    IFS=':' read -r mode fewshot name <<< "$scenario"
    
    ((SCENARIO_COUNT++))
    
    echo
    echo "=========================================="
    echo "SCENARIO $SCENARIO_COUNT/4: $name"
    echo "=========================================="
    echo "Detection mode: $mode"
    echo "Few-shot: $fewshot"
    echo
    
    # Export configuration for this scenario
    export EVAL_NUM_SAMPLES="$NUM_SAMPLES"
    export EVAL_USE_CACHE="$USE_CACHE"
    export EVAL_DETECTION_MODE="$mode"
    export EVAL_USE_FEWSHOT="$fewshot"
    export EVAL_PERSISTENT_DIR="$USE_PERSISTENT_DIR"
    
    # Start timer for this scenario
    SCENARIO_START=$(date +%s)
    
    # Run the single scenario script
    if ./eval_bbox_api_single_scenario.sh; then
        SCENARIO_END=$(date +%s)
        SCENARIO_DURATION=$((SCENARIO_END - SCENARIO_START))
        SCENARIO_MINUTES=$((SCENARIO_DURATION / 60))
        SCENARIO_SECONDS=$((SCENARIO_DURATION % 60))
        
        echo "✅ Completed scenario: $name"
        echo "   Time: ${SCENARIO_MINUTES}m ${SCENARIO_SECONDS}s"
    else
        echo "❌ Failed scenario: $name"
        FAILED_SCENARIOS+=("$name")
        
        # Continue with next scenario even if one fails
        continue
    fi
    
    echo
    echo "Waiting 5 seconds before next scenario..."
    sleep 5
done

# Calculate total duration
OVERALL_END=$(date +%s)
OVERALL_DURATION=$((OVERALL_END - OVERALL_START))
OVERALL_MINUTES=$((OVERALL_DURATION / 60))
OVERALL_SECONDS=$((OVERALL_DURATION % 60))

# Final summary
echo
echo "=========================================="
echo "ALL SCENARIOS COMPLETE"
echo "=========================================="
echo "Total scenarios: 4"
echo "Successful: $((4 - ${#FAILED_SCENARIOS[@]}))"
echo "Failed: ${#FAILED_SCENARIOS[@]}"

if [ ${#FAILED_SCENARIOS[@]} -gt 0 ]; then
    echo "Failed scenarios:"
    for failed in "${FAILED_SCENARIOS[@]}"; do
        echo "  - $failed"
    done
fi

echo "Total time: ${OVERALL_MINUTES}m ${OVERALL_SECONDS}s"
echo

# Results location
if [ "$USE_PERSISTENT_DIR" = "true" ]; then
    RESULTS_DIR="results/bbox_cholecseg8k_local_quick"
else
    RESULTS_DIR="results/bbox_cholecseg8k_local_*"
fi

echo "Results saved in: $RESULTS_DIR"
echo "Summary files created:"
echo "  - summary_combined_zeroshot.json"
echo "  - summary_combined_fewshot.json"
echo "  - summary_separate_zeroshot.json"
echo "  - summary_separate_fewshot.json"
echo

echo "To aggregate results across all scenarios:"
echo "  python3 aggregate_bbox_results.py"
echo

if [ ${#FAILED_SCENARIOS[@]} -eq 0 ]; then
    echo "✨ All API model scenarios completed successfully!"
    exit 0
else
    echo "⚠️  Some scenarios failed. Check the logs above for details."
    exit 1
fi