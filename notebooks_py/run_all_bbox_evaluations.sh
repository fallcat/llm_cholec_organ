#!/bin/bash

# Comprehensive evaluation script for all bounding box experiments
# Runs all models, all scenarios, and naive baselines on both datasets

echo "=========================================================================="
echo "COMPREHENSIVE BOUNDING BOX EVALUATION"
echo "=========================================================================="
echo "Starting at: $(date)"
echo

# Configuration
NUM_SAMPLES=${1:-200}  # Default to 200 samples (full test set)
SKIP_API=${2:-false}    # Set to "true" to skip API models (saves money)
SKIP_LOCAL=${3:-false}  # Set to "true" to skip local models (saves GPU time)
SKIP_BASELINE=${4:-false}  # Set to "true" to skip naive baseline

export EVAL_NUM_SAMPLES=$NUM_SAMPLES
export EVAL_USE_CACHE=true
export EVAL_PERSISTENT_DIR=true  # Use persistent folders for full runs

echo "Configuration:"
echo "  Samples: $NUM_SAMPLES"
echo "  Skip API models: $SKIP_API"
echo "  Skip local models: $SKIP_LOCAL"
echo "  Skip baseline: $SKIP_BASELINE"
echo "  Output: Persistent directories"
echo

# Create log directory
LOG_DIR="logs/bbox_eval_$(date +%Y%m%d_%H%M%S)"
mkdir -p $LOG_DIR

# Function to run and log
run_with_log() {
    local cmd=$1
    local log_file=$2
    echo "Running: $cmd"
    echo "Logging to: $log_file"
    eval $cmd 2>&1 | tee $log_file
}

# ============================================================================
# PART 1: NAIVE BASELINE
# ============================================================================
if [ "$SKIP_BASELINE" != "true" ]; then
    echo
    echo "=========================================================================="
    echo "PART 1: NAIVE BASELINE"
    echo "=========================================================================="
    echo
    
    # Run naive baseline with different presence modes
    for presence_mode in perfect all random; do
        echo "--- Naive Baseline (presence=$presence_mode) ---"
        run_with_log \
            "EVAL_PRESENCE_MODE=$presence_mode EVAL_DATASET=both python3 eval_bbox_naive_baseline.py" \
            "$LOG_DIR/naive_baseline_${presence_mode}.log"
        echo
    done
else
    echo "Skipping naive baseline (SKIP_BASELINE=true)"
fi

# ============================================================================
# PART 2: CHOLECSEG8K DATASET
# ============================================================================
echo
echo "=========================================================================="
echo "PART 2: CHOLECSEG8K DATASET EVALUATION"
echo "=========================================================================="
echo

# API Models for CholecSeg8k
if [ "$SKIP_API" != "true" ]; then
    echo "--- CholecSeg8k: API Models ---"
    
    API_MODELS=(
        "gpt-4.1"
        "claude-sonnet-4-20250514"
        "gemini-2.0-flash"
    )
    
    for model in "${API_MODELS[@]}"; do
        export EVAL_MODEL="$model"
        echo
        echo "Model: $model"
        
        # All 4 scenarios
        echo "  1/4: Zero-shot Combined"
        run_with_log \
            "EVAL_DETECTION_MODE=combined EVAL_USE_FEWSHOT=false python3 eval_bbox_quick_test.py" \
            "$LOG_DIR/cholecseg8k_${model}_zeroshot_combined.log"
        
        echo "  2/4: Few-shot Combined"
        run_with_log \
            "EVAL_DETECTION_MODE=combined EVAL_USE_FEWSHOT=true python3 eval_bbox_quick_test.py" \
            "$LOG_DIR/cholecseg8k_${model}_fewshot_combined.log"
        
        echo "  3/4: Zero-shot Separate"
        run_with_log \
            "EVAL_DETECTION_MODE=separate EVAL_USE_FEWSHOT=false python3 eval_bbox_quick_test.py" \
            "$LOG_DIR/cholecseg8k_${model}_zeroshot_separate.log"
        
        echo "  4/4: Few-shot Separate"
        run_with_log \
            "EVAL_DETECTION_MODE=separate EVAL_USE_FEWSHOT=true python3 eval_bbox_quick_test.py" \
            "$LOG_DIR/cholecseg8k_${model}_fewshot_separate.log"
    done
else
    echo "Skipping API models (SKIP_API=true)"
fi

# Local Models for CholecSeg8k
if [ "$SKIP_LOCAL" != "true" ]; then
    echo
    echo "--- CholecSeg8k: Local Models ---"
    
    LOCAL_MODELS=(
        "llava-hf/llava-v1.6-mistral-7b-hf"
        "Qwen/Qwen2.5-VL-7B-Instruct"
        "mistralai/Pixtral-12B-2409"
    )
    
    for model in "${LOCAL_MODELS[@]}"; do
        export EVAL_MODEL="$model"
        echo
        echo "Model: $model"
        
        # All 4 scenarios
        echo "  1/4: Zero-shot Combined"
        run_with_log \
            "EVAL_DETECTION_MODE=combined EVAL_USE_FEWSHOT=false python3 eval_bbox_quick_test.py" \
            "$LOG_DIR/cholecseg8k_${model//\//_}_zeroshot_combined.log"
        
        echo "  2/4: Few-shot Combined"
        run_with_log \
            "EVAL_DETECTION_MODE=combined EVAL_USE_FEWSHOT=true python3 eval_bbox_quick_test.py" \
            "$LOG_DIR/cholecseg8k_${model//\//_}_fewshot_combined.log"
        
        echo "  3/4: Zero-shot Separate"
        run_with_log \
            "EVAL_DETECTION_MODE=separate EVAL_USE_FEWSHOT=false python3 eval_bbox_quick_test.py" \
            "$LOG_DIR/cholecseg8k_${model//\//_}_zeroshot_separate.log"
        
        echo "  4/4: Few-shot Separate"
        run_with_log \
            "EVAL_DETECTION_MODE=separate EVAL_USE_FEWSHOT=true python3 eval_bbox_quick_test.py" \
            "$LOG_DIR/cholecseg8k_${model//\//_}_fewshot_separate.log"
        
        echo "  Waiting 10 seconds to clear GPU memory..."
        sleep 10
    done
else
    echo "Skipping local models (SKIP_LOCAL=true)"
fi

# ============================================================================
# PART 3: CHOLEC_ORGANS DATASET
# ============================================================================
echo
echo "=========================================================================="
echo "PART 3: CHOLEC_ORGANS DATASET EVALUATION"
echo "=========================================================================="
echo

# API Models for CholecOrgans
if [ "$SKIP_API" != "true" ]; then
    echo "--- CholecOrgans: API Models ---"
    
    for model in "${API_MODELS[@]}"; do
        export EVAL_MODEL="$model"
        echo
        echo "Model: $model"
        
        # All 4 scenarios
        echo "  1/4: Zero-shot Combined"
        run_with_log \
            "EVAL_DETECTION_MODE=combined EVAL_USE_FEWSHOT=false python3 eval_bbox_cholec_organs.py" \
            "$LOG_DIR/cholec_organs_${model}_zeroshot_combined.log"
        
        echo "  2/4: Few-shot Combined"
        run_with_log \
            "EVAL_DETECTION_MODE=combined EVAL_USE_FEWSHOT=true python3 eval_bbox_cholec_organs.py" \
            "$LOG_DIR/cholec_organs_${model}_fewshot_combined.log"
        
        echo "  3/4: Zero-shot Separate"
        run_with_log \
            "EVAL_DETECTION_MODE=separate EVAL_USE_FEWSHOT=false python3 eval_bbox_cholec_organs.py" \
            "$LOG_DIR/cholec_organs_${model}_zeroshot_separate.log"
        
        echo "  4/4: Few-shot Separate"
        run_with_log \
            "EVAL_DETECTION_MODE=separate EVAL_USE_FEWSHOT=true python3 eval_bbox_cholec_organs.py" \
            "$LOG_DIR/cholec_organs_${model}_fewshot_separate.log"
    done
else
    echo "Skipping API models (SKIP_API=true)"
fi

# Local Models for CholecOrgans
if [ "$SKIP_LOCAL" != "true" ]; then
    echo
    echo "--- CholecOrgans: Local Models ---"
    
    for model in "${LOCAL_MODELS[@]}"; do
        export EVAL_MODEL="$model"
        echo
        echo "Model: $model"
        
        # All 4 scenarios
        echo "  1/4: Zero-shot Combined"
        run_with_log \
            "EVAL_DETECTION_MODE=combined EVAL_USE_FEWSHOT=false python3 eval_bbox_cholec_organs.py" \
            "$LOG_DIR/cholec_organs_${model//\//_}_zeroshot_combined.log"
        
        echo "  2/4: Few-shot Combined"
        run_with_log \
            "EVAL_DETECTION_MODE=combined EVAL_USE_FEWSHOT=true python3 eval_bbox_cholec_organs.py" \
            "$LOG_DIR/cholec_organs_${model//\//_}_fewshot_combined.log"
        
        echo "  3/4: Zero-shot Separate"
        run_with_log \
            "EVAL_DETECTION_MODE=separate EVAL_USE_FEWSHOT=false python3 eval_bbox_cholec_organs.py" \
            "$LOG_DIR/cholec_organs_${model//\//_}_zeroshot_separate.log"
        
        echo "  4/4: Few-shot Separate"
        run_with_log \
            "EVAL_DETECTION_MODE=separate EVAL_USE_FEWSHOT=true python3 eval_bbox_cholec_organs.py" \
            "$LOG_DIR/cholec_organs_${model//\//_}_fewshot_separate.log"
        
        echo "  Waiting 10 seconds to clear GPU memory..."
        sleep 10
    done
else
    echo "Skipping local models (SKIP_LOCAL=true)"
fi

# ============================================================================
# PART 4: SUMMARY GENERATION
# ============================================================================
echo
echo "=========================================================================="
echo "PART 4: GENERATING SUMMARY REPORTS"
echo "=========================================================================="
echo

# Create summary script
cat > $LOG_DIR/generate_summary.py << 'EOF'
#!/usr/bin/env python3
import json
from pathlib import Path
import pandas as pd

# Collect all results
results_dirs = [
    Path("/shared_data0/weiqiuy/llm_cholec_organ/results/bbox_cholecseg8k_local_quick"),
    Path("/shared_data0/weiqiuy/llm_cholec_organ/results/bbox_cholec_organs_quick"),
]

all_results = []

for results_dir in results_dirs:
    if not results_dir.exists():
        continue
    
    dataset = "cholecseg8k" if "cholecseg8k" in results_dir.name else "cholec_organs"
    
    for mode_dir in results_dir.glob("*"):
        if not mode_dir.is_dir():
            continue
        
        mode = mode_dir.name
        
        for model_dir in mode_dir.glob("*"):
            if not model_dir.is_dir():
                continue
            
            summary_files = list(model_dir.glob("summary*.json"))
            if summary_files:
                with open(summary_files[0], 'r') as f:
                    data = json.load(f)
                    metrics = data.get('metrics', {})
                    all_results.append({
                        'dataset': dataset,
                        'mode': mode,
                        'model': model_dir.name,
                        'presence_acc': metrics.get('presence_accuracy', 0),
                        'iou_bbox': metrics.get('mean_iou_bbox_to_bbox', 0),
                        'iou_mask': metrics.get('mean_iou_bbox_to_mask', 0),
                    })

# Create summary DataFrame
if all_results:
    df = pd.DataFrame(all_results)
    df = df.sort_values(['dataset', 'mode', 'model'])
    
    print("\nSUMMARY TABLE:")
    print("="*100)
    print(df.to_string(index=False))
    
    # Save to CSV
    df.to_csv('evaluation_summary.csv', index=False)
    print(f"\nSummary saved to: evaluation_summary.csv")
else:
    print("No results found")
EOF

python3 $LOG_DIR/generate_summary.py

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
echo "  - CholecSeg8k: results/bbox_cholecseg8k_local_quick/"
echo "  - CholecOrgans: results/bbox_cholec_organs_quick/"
echo "  - Naive baseline: results/naive_baseline_*/"
echo "  - Logs: $LOG_DIR/"
echo
echo "To view results:"
echo "  - Individual JSONs: Check the results directories"
echo "  - Summary notebook: Run notebooks/main_table_result.ipynb"
echo "  - Logs: Check $LOG_DIR/"
echo
echo "Total experiments run:"
echo "  - Naive baseline: 3 modes × 2 datasets = 6"
echo "  - CholecSeg8k: 6 models × 4 scenarios = 24"
echo "  - CholecOrgans: 6 models × 4 scenarios = 24"
echo "  - TOTAL: 54 experiments"
echo "=========================================================================="