#!/bin/bash
# Generate combined few-shot examples for all datasets with minimum 3 examples

set -e  # Exit on error

echo "=========================================="
echo "Generating Combined Few-Shot Examples for All Datasets"
echo "=========================================="
echo ""

# Change to script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
MIN_EXAMPLES=3
MAX_EXAMPLES=""  # No limit by default
FORCE_REGENERATE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --min)
            MIN_EXAMPLES="$2"
            shift 2
            ;;
        --max)
            MAX_EXAMPLES="$2"
            shift 2
            ;;
        --force)
            FORCE_REGENERATE=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --min N      Minimum number of examples (default: 3)"
            echo "  --max N      Maximum number of examples (default: no limit)"
            echo "  --force      Force regeneration of cached files"
            echo "  --help       Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                    # Use defaults (min=3, no max)"
            echo "  $0 --min 5 --max 10   # Between 5 and 10 examples"
            echo "  $0 --force            # Force regenerate all"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build command
CMD="python generate_few_shot_sep_comb.py --mode combined --min-combined $MIN_EXAMPLES"

if [ -n "$MAX_EXAMPLES" ]; then
    CMD="$CMD --max-combined $MAX_EXAMPLES"
fi

if [ "$FORCE_REGENERATE" = "true" ]; then
    CMD="$CMD --force"
fi

# List of datasets
DATASETS=("cholecseg8k_local" "cholec_organs" "cholec_gonogo")

# Summary
echo -e "${BLUE}Configuration:${NC}"
echo "  Minimum examples: $MIN_EXAMPLES"
if [ -n "$MAX_EXAMPLES" ]; then
    echo "  Maximum examples: $MAX_EXAMPLES"
else
    echo "  Maximum examples: No limit"
fi
echo "  Force regenerate: $FORCE_REGENERATE"
echo "  Datasets: ${DATASETS[@]}"
echo ""

# Track results
SUCCESS_COUNT=0
FAIL_COUNT=0
FAILED_DATASETS=()

# Start timer
START_TIME=$(date +%s)

# Process each dataset
for dataset in "${DATASETS[@]}"; do
    echo -e "${BLUE}=========================================="
    echo -e "Processing: $dataset"
    echo -e "==========================================${NC}"
    
    if $CMD --dataset "$dataset"; then
        echo -e "${GREEN}✓ Successfully generated for $dataset${NC}"
        ((SUCCESS_COUNT++))
    else
        echo -e "${RED}✗ Failed to generate for $dataset${NC}"
        ((FAIL_COUNT++))
        FAILED_DATASETS+=("$dataset")
    fi
    
    echo ""
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

# Show generated files
echo -e "${BLUE}Generated Combined Plans:${NC}"
for dataset in "${DATASETS[@]}"; do
    OUTPUT_DIR="/shared_data0/weiqiuy/llm_cholec_organ/data_info/${dataset}_balanced_200"
    
    # Check for combined plan files
    BBOX_FILE="$OUTPUT_DIR/fewshot_plan_bbox_combined_greedy.json"
    POINTING_FILE="$OUTPUT_DIR/fewshot_plan_pointing_combined_greedy.json"
    
    echo "  $dataset:"
    if [ -f "$BBOX_FILE" ]; then
        SIZE=$(du -h "$BBOX_FILE" | cut -f1)
        # Get number of examples from the file
        N_EXAMPLES=$(python -c "import json; print(json.load(open('$BBOX_FILE'))['metadata']['num_examples'])" 2>/dev/null || echo "?")
        echo -e "    ${GREEN}✓${NC} Bounding box plan: $N_EXAMPLES examples ($SIZE)"
    elif [ -f "$POINTING_FILE" ]; then
        SIZE=$(du -h "$POINTING_FILE" | cut -f1)
        N_EXAMPLES=$(python -c "import json; print(json.load(open('$POINTING_FILE'))['metadata']['num_examples'])" 2>/dev/null || echo "?")
        echo -e "    ${YELLOW}✓${NC} Pointing plan: $N_EXAMPLES examples ($SIZE)"
    else
        echo -e "    ${RED}✗${NC} No combined plan found"
    fi
done

echo ""

# Show example usage
if [ $SUCCESS_COUNT -gt 0 ]; then
    echo -e "${GREEN}✨ Success! Combined few-shot plans generated.${NC}"
    echo ""
    echo "Example usage in evaluation:"
    echo "  # Use combined few-shot examples for evaluation"
    echo "  python eval_combined.py --dataset cholecseg8k_local \\"
    echo "    --fewshot-plan data_info/cholecseg8k_local_balanced_200/fewshot_plan_bbox_combined_greedy.json"
else
    echo -e "${RED}⚠️ No datasets were successfully processed.${NC}"
fi

exit $FAIL_COUNT