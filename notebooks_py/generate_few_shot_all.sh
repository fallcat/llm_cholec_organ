#!/bin/bash
# Generate few-shot examples for all datasets (both separate and combined modes)

set -e  # Exit on error

echo "=========================================="
echo "Generating Few-Shot Examples for All Datasets"
echo "=========================================="
echo ""

# Change to script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to run generation for a dataset
generate_for_dataset() {
    local dataset=$1
    local mode=${2:-both}
    local force=${3:-false}
    local min_combined=${4:-3}
    
    echo -e "${BLUE}=========================================="
    echo -e "Processing: $dataset"
    echo -e "Mode: $mode"
    echo -e "Force regenerate: $force"
    echo -e "Min combined examples: $min_combined"
    echo -e "==========================================${NC}"
    
    if [ "$force" = "true" ]; then
        FORCE_FLAG="--force"
    else
        FORCE_FLAG=""
    fi
    
    # Run the Python script with min-combined parameter
    if python generate_few_shot_sep_comb.py --dataset "$dataset" --mode "$mode" --min-combined "$min_combined" $FORCE_FLAG; then
        echo -e "${GREEN}✓ Successfully generated for $dataset${NC}"
    else
        echo -e "${RED}✗ Failed to generate for $dataset${NC}"
        return 1
    fi
    
    echo ""
}

# Parse command line arguments
MODE="both"
FORCE="false"
DATASETS=""
MIN_COMBINED=3

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --force)
            FORCE="true"
            shift
            ;;
        --datasets)
            DATASETS="$2"
            shift 2
            ;;
        --min-combined)
            MIN_COMBINED="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --mode MODE        Generation mode: separate, combined, or both (default: both)"
            echo "  --force            Force regeneration of cached files"
            echo "  --datasets LIST    Comma-separated list of datasets (default: all)"
            echo "                     Options: cholecseg8k_local,cholec_organs,cholec_gonogo"
            echo "  --min-combined N   Minimum number of combined examples (default: 3)"
            echo "  --help             Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Generate both modes for all datasets (min 3 combined)"
            echo "  $0 --mode separate                    # Only separate mode for all"
            echo "  $0 --datasets cholecseg8k_local       # Only cholecseg8k (both modes)"
            echo "  $0 --force --mode combined            # Force regenerate combined only"
            echo "  $0 --min-combined 5                   # Use minimum 5 combined examples"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Default to all datasets if not specified
if [ -z "$DATASETS" ]; then
    DATASETS="cholecseg8k_local,cholec_organs,cholec_gonogo"
fi

# Convert comma-separated list to array
IFS=',' read -ra DATASET_ARRAY <<< "$DATASETS"

# Track results
SUCCESS_COUNT=0
FAIL_COUNT=0
FAILED_DATASETS=()

# Start timer
START_TIME=$(date +%s)

echo -e "${BLUE}Configuration:${NC}"
echo "  Datasets: ${DATASET_ARRAY[@]}"
echo "  Mode: $MODE"
echo "  Force regenerate: $FORCE"
echo "  Min combined examples: $MIN_COMBINED"
echo ""

# Process each dataset
for dataset in "${DATASET_ARRAY[@]}"; do
    if generate_for_dataset "$dataset" "$MODE" "$FORCE" "$MIN_COMBINED"; then
        ((SUCCESS_COUNT++))
    else
        ((FAIL_COUNT++))
        FAILED_DATASETS+=("$dataset")
    fi
done

# End timer
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Print summary
echo -e "${BLUE}=========================================="
echo -e "Summary"
echo -e "==========================================${NC}"
echo -e "Time taken: ${DURATION} seconds"
echo -e "Successful: ${GREEN}$SUCCESS_COUNT${NC}"
echo -e "Failed: ${RED}$FAIL_COUNT${NC}"

if [ ${#FAILED_DATASETS[@]} -gt 0 ]; then
    echo -e "${RED}Failed datasets: ${FAILED_DATASETS[@]}${NC}"
fi

echo ""

# Print file locations
echo -e "${BLUE}Output locations:${NC}"
for dataset in "${DATASET_ARRAY[@]}"; do
    OUTPUT_DIR="/shared_data0/weiqiuy/llm_cholec_organ/data_info/${dataset}_balanced_200"
    if [ -d "$OUTPUT_DIR" ]; then
        echo "  $dataset:"
        echo "    Directory: $OUTPUT_DIR"
        
        # List key files if they exist
        if [ -f "$OUTPUT_DIR/fewshot_plan_bbox_200.json" ]; then
            echo "    ✓ Separate bbox plan: fewshot_plan_bbox_200.json"
        fi
        if [ -f "$OUTPUT_DIR/fewshot_plan_pointing_200.json" ]; then
            echo "    ✓ Separate pointing plan: fewshot_plan_pointing_200.json"
        fi
        if [ -f "$OUTPUT_DIR/fewshot_plan_bbox_combined_greedy.json" ]; then
            echo "    ✓ Combined bbox plan: fewshot_plan_bbox_combined_greedy.json"
        fi
    fi
done

echo ""

if [ $FAIL_COUNT -eq 0 ]; then
    echo -e "${GREEN}✨ All datasets processed successfully!${NC}"
    exit 0
else
    echo -e "${RED}⚠️  Some datasets failed. Check the logs above.${NC}"
    exit 1
fi