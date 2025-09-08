#!/bin/bash
# Script to run PeskaVLP evaluation on all three datasets

echo "================================================================"
echo "Running PeskaVLP on all datasets"
echo "================================================================"
echo ""

# Configuration
MODEL="peskavlp"
NUM_SAMPLES=200  # Adjust this as needed
USE_CACHE="true"
PERSISTENT_DIR="true"
DETECTION_MODE="combined"
USE_FEWSHOT="false"

# Base directory
BASE_DIR="/shared_data0/weiqiuy/llm_cholec_organ/notebooks_py"

# Function to run evaluation
run_evaluation() {
    local dataset=$1
    local display_name=$2
    
    echo "================================================================"
    echo "Dataset: $display_name"
    echo "================================================================"
    echo "Starting evaluation at: $(date)"
    echo ""
    
    cd $BASE_DIR
    
    EVAL_DATASET=$dataset \
    EVAL_MODEL=$MODEL \
    EVAL_NUM_SAMPLES=$NUM_SAMPLES \
    EVAL_USE_CACHE=$USE_CACHE \
    EVAL_PERSISTENT_DIR=$PERSISTENT_DIR \
    EVAL_DETECTION_MODE=$DETECTION_MODE \
    EVAL_USE_FEWSHOT=$USE_FEWSHOT \
    python eval_bbox_unified.py
    
    if [ $? -eq 0 ]; then
        echo "✓ $display_name evaluation completed successfully"
    else
        echo "✗ $display_name evaluation failed"
    fi
    
    echo "Completed at: $(date)"
    echo ""
    echo ""
}

# Run on all three datasets
echo "Starting PeskaVLP evaluation pipeline"
echo "Model: $MODEL"
echo "Samples per dataset: $NUM_SAMPLES"
echo "Detection mode: $DETECTION_MODE"
echo "Cache: $USE_CACHE"
echo ""

# Dataset 1: CholecSeg8k
run_evaluation "cholecseg8k" "CHOLECSEG8K"

# Dataset 2: Cholec Organs
run_evaluation "cholec_organs" "CHOLEC ORGANS"

# Dataset 3: Cholec GoNoGo
run_evaluation "cholec_gonogo" "CHOLEC GONOGO"

# Summary
echo "================================================================"
echo "EVALUATION SUMMARY"
echo "================================================================"
echo ""

# Function to extract metrics from summary file
print_metrics() {
    local dataset=$1
    local display_name=$2
    local results_dir=$3
    
    local summary_file="/shared_data0/weiqiuy/llm_cholec_organ/results/$results_dir/summary_combined_zeroshot.json"
    
    if [ -f "$summary_file" ]; then
        echo "$display_name Results:"
        echo -n "  Presence Accuracy: "
        python -c "import json; data=json.load(open('$summary_file')); print(f\"{data['peskavlp']['metrics']['presence_accuracy']*100:.1f}%\")" 2>/dev/null || echo "N/A"
        echo -n "  Mean IoU (Bbox-to-Bbox): "
        python -c "import json; data=json.load(open('$summary_file')); print(f\"{data['peskavlp']['metrics']['mean_iou_bbox_to_bbox']:.3f}\")" 2>/dev/null || echo "N/A"
        echo -n "  Mean IoU (Bbox-to-Mask): "
        python -c "import json; data=json.load(open('$summary_file')); print(f\"{data['peskavlp']['metrics']['mean_iou_bbox_to_mask']:.3f}\")" 2>/dev/null || echo "N/A"
        echo ""
    else
        echo "$display_name: No results found"
    fi
}

# Print metrics for each dataset
print_metrics "cholecseg8k" "CholecSeg8k" "bbox_cholecseg8k_local_quick"
print_metrics "cholec_organs" "Cholec Organs" "bbox_cholec_organs_quick"
print_metrics "cholec_gonogo" "Cholec GoNoGo" "bbox_cholec_gonogo_quick"

echo ""
echo "================================================================"
echo "All evaluations completed at: $(date)"
echo "================================================================"