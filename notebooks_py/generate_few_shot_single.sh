#!/bin/bash
# Generate few-shot examples for a single dataset using environment variable

set -e  # Exit on error

# Check if DATASET_NAME is set
if [ -z "$DATASET_NAME" ]; then
    echo "Error: DATASET_NAME environment variable not set"
    echo ""
    echo "Usage:"
    echo "  DATASET_NAME=cholecseg8k_local ./generate_few_shot_single.sh"
    echo "  DATASET_NAME=cholec_organs ./generate_few_shot_single.sh"
    echo "  DATASET_NAME=cholec_gonogo ./generate_few_shot_single.sh"
    echo ""
    echo "Optional environment variables:"
    echo "  FORCE_REGENERATE=true    # Force regeneration of cached files"
    echo "  MODE=separate            # Mode: separate, combined, or both (default: both)"
    exit 1
fi

# Get optional environment variables
MODE=${MODE:-both}
FORCE_REGENERATE=${FORCE_REGENERATE:-false}

echo "=========================================="
echo "Generating Few-Shot Examples"
echo "=========================================="
echo "Dataset: $DATASET_NAME"
echo "Mode: $MODE"
echo "Force regenerate: $FORCE_REGENERATE"
echo ""

# Change to script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Build command
CMD="python generate_few_shot_sep_comb.py --dataset $DATASET_NAME --mode $MODE"

if [ "$FORCE_REGENERATE" = "true" ]; then
    CMD="$CMD --force"
fi

# Run the command
echo "Running: $CMD"
echo ""

if $CMD; then
    echo ""
    echo "✅ Successfully generated few-shot examples for $DATASET_NAME"
    
    # Show output location
    OUTPUT_DIR="/shared_data0/weiqiuy/llm_cholec_organ/data_info/${DATASET_NAME}_balanced_200"
    echo ""
    echo "📁 Output location: $OUTPUT_DIR"
    
    # List generated files
    if [ -d "$OUTPUT_DIR" ]; then
        echo ""
        echo "📄 Generated files:"
        ls -lh "$OUTPUT_DIR"/*.json 2>/dev/null | awk '{print "   " $9 " (" $5 ")"}'
    fi
else
    echo ""
    echo "❌ Failed to generate few-shot examples for $DATASET_NAME"
    exit 1
fi