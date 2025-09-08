#!/bin/bash

# Script to run RASO evaluation on all three datasets
# Usage: EVAL_NUM_SAMPLES=50 ./run_raso_all.sh
# Or just: ./run_raso_all.sh (defaults to 10 samples)

# Set default number of samples if not provided
EVAL_NUM_SAMPLES=${EVAL_NUM_SAMPLES:-200}

echo "=========================================="
echo "RASO Evaluation on All Datasets"
echo "Samples per dataset: $EVAL_NUM_SAMPLES"
echo "=========================================="

# Array of datasets to test
datasets=("cholec_organs" "cholec_gonogo" "cholecseg8k")

# Track start time
start_time=$(date +%s)

# Run evaluation for each dataset
for dataset in "${datasets[@]}"; do
    echo ""
    echo "=========================================="
    echo "Dataset: $dataset"
    echo "=========================================="
    
    # Clean previous results for this dataset/model
    if [ "$dataset" = "cholecseg8k" ]; then
        rm -rf "/shared_data0/weiqiuy/llm_cholec_organ/results/bbox_${dataset}_local_quick/zeroshot_combined/raso" 2>/dev/null
        rm -rf "/shared_data0/weiqiuy/llm_cholec_organ/results/bbox_${dataset}_quick/zeroshot_combined/raso" 2>/dev/null
    else
        rm -rf "/shared_data0/weiqiuy/llm_cholec_organ/results/bbox_${dataset}_quick/zeroshot_combined/raso" 2>/dev/null
    fi
    
    # Run evaluation
    EVAL_DATASET=$dataset \
    EVAL_MODEL=raso \
    EVAL_NUM_SAMPLES=$EVAL_NUM_SAMPLES \
    EVAL_USE_CACHE=true \
    python3 notebooks_py/eval_bbox_unified.py
    
    # Check if it succeeded
    if [ $? -eq 0 ]; then
        echo "✓ $dataset completed successfully"
    else
        echo "✗ $dataset failed"
    fi
done

# Calculate total time
end_time=$(date +%s)
duration=$((end_time - start_time))

echo ""
echo "=========================================="
echo "All evaluations completed!"
echo "Total time: ${duration} seconds"
echo "=========================================="

# Display summary of results
echo ""
echo "Results Summary:"
echo "----------------"
for dataset in "${datasets[@]}"; do
    # Try both possible result directory names
    if [ "$dataset" = "cholecseg8k" ]; then
        # Check both local and non-local paths for cholecseg8k
        metrics_file="/shared_data0/weiqiuy/llm_cholec_organ/results/bbox_${dataset}_local_quick/zeroshot_combined/raso/metrics.json"
        if [ ! -f "$metrics_file" ]; then
            metrics_file="/shared_data0/weiqiuy/llm_cholec_organ/results/bbox_${dataset}_quick/zeroshot_combined/raso/metrics.json"
        fi
    else
        metrics_file="/shared_data0/weiqiuy/llm_cholec_organ/results/bbox_${dataset}_quick/zeroshot_combined/raso/metrics.json"
    fi
    
    if [ -f "$metrics_file" ]; then
        accuracy=$(python3 -c "import json; print(json.load(open('$metrics_file'))['presence_accuracy']*100)")
        printf "%-20s: %5.1f%% presence accuracy\n" "$dataset" "$accuracy"
    else
        printf "%-20s: No results found\n" "$dataset"
    fi
done
