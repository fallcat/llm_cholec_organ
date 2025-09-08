#!/bin/bash

# Script to run CholeNet and GoNoGoNet evaluation on all datasets
# This will test both models on their native datasets and cross-dataset evaluation

echo "=========================================="
echo "CholeNet and GoNoGoNet Full Evaluation"
echo "=========================================="
echo ""

# Configuration
NUM_SAMPLES=20  # Adjust this for quick testing or full evaluation
USE_CACHE=true
PERSISTENT_DIR=true
DETECTION_MODE=combined

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to run evaluation and capture results
run_eval() {
    local dataset=$1
    local model=$2
    local samples=${3:-$NUM_SAMPLES}
    
    echo -e "${BLUE}Running $model on $dataset with $samples samples...${NC}"
    
    EVAL_DATASET=$dataset \
    EVAL_MODEL=$model \
    EVAL_NUM_SAMPLES=$samples \
    EVAL_USE_CACHE=$USE_CACHE \
    EVAL_PERSISTENT_DIR=$PERSISTENT_DIR \
    EVAL_DETECTION_MODE=$DETECTION_MODE \
    python notebooks_py/eval_bbox_unified.py 2>&1 | tail -20
    
    echo ""
}

# ============================================
# CholeNet Evaluation
# ============================================
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}CholeNet Evaluation${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""

echo "1. CholeNet on CholecSeg8k (can detect 3/13 organs)"
echo "   Expected: Liver, Gallbladder, Hepatocystic Triangle"
echo "   -------------------------------------------------"
run_eval cholecseg8k cholenet $NUM_SAMPLES

echo "2. CholeNet on CholecOrgans (native dataset - all 3 organs)"
echo "   Expected: Perfect detection for 3 organs"
echo "   -------------------------------------------------"
run_eval cholec_organs cholenet $NUM_SAMPLES

echo "3. CholeNet on CholecGoNoGo (cross-dataset)"
echo "   Expected: Hepatocystic Triangle → Go Zone mapping"
echo "   -------------------------------------------------"
run_eval cholec_gonogo cholenet $NUM_SAMPLES

# ============================================
# GoNoGoNet Evaluation
# ============================================
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}GoNoGoNet Evaluation${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""

echo "1. GoNoGoNet on CholecSeg8k (cannot detect organs)"
echo "   Expected: All organs marked as absent"
echo "   -------------------------------------------------"
run_eval cholecseg8k gonogonet $NUM_SAMPLES

echo "2. GoNoGoNet on CholecOrgans (cross-dataset)"
echo "   Expected: Go Zone → Hepatocystic Triangle mapping"
echo "   -------------------------------------------------"
run_eval cholec_organs gonogonet $NUM_SAMPLES

echo "3. GoNoGoNet on CholecGoNoGo (native dataset)"
echo "   Expected: Perfect Go/NoGo zone detection"
echo "   -------------------------------------------------"
run_eval cholec_gonogo gonogonet $NUM_SAMPLES

# ============================================
# Summary
# ============================================
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}Evaluation Complete!${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""
echo "Results saved in:"
echo "  - /shared_data0/weiqiuy/llm_cholec_organ/results/bbox_cholecseg8k_local_quick/"
echo "  - /shared_data0/weiqiuy/llm_cholec_organ/results/bbox_cholec_organs_quick/"
echo "  - /shared_data0/weiqiuy/llm_cholec_organ/results/bbox_cholec_gonogo_quick/"
echo ""
echo "Model mappings:"
echo "  CholeNet → GoNoGo: Hepatocystic Triangle → Go Zone"
echo "  GoNoGoNet → Organs: Go Zone → Hepatocystic Triangle, NoGo → Background"