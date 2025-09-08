#!/bin/bash
# Script to test GoNoGoNet and CholeNet models with existing evaluation pipeline

echo "============================================"
echo "Testing Local Segmentation Models"
echo "============================================"

# Test GoNoGoNet on CholecGoNoGo dataset
echo ""
echo "1. Testing GoNoGoNet on CholecGoNoGo dataset..."
echo "---------------------------------------------"
cd /shared_data0/weiqiuy/llm_cholec_organ/notebooks_py
EVAL_MODEL=gonogo EVAL_NUM_SAMPLES=2 EVAL_USE_CACHE=false EVAL_DETECTION_MODE=combined \
    python eval_bbox_cholec_gonogo.py

# Test CholeNet on CholecSeg8k dataset  
echo ""
echo "2. Testing CholeNet on CholecSeg8k dataset..."
echo "---------------------------------------------"
EVAL_MODEL=cholenet EVAL_NUM_SAMPLES=2 EVAL_USE_CACHE=false EVAL_DETECTION_MODE=combined \
    python eval_bbox_cholecseg8k.py

echo ""
echo "============================================"
echo "Testing Complete!"
echo "============================================"