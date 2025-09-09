#!/bin/bash
# Test script to run bbox evaluation with CholeNet and GoNoGoNet

echo "=================================================="
echo "Testing CholeNet on CholecSeg8k dataset"
echo "=================================================="

# Test CholeNet on CholecSeg8k
EVAL_DATASET=cholecseg8k \
EVAL_MODEL=cholenet \
EVAL_NUM_SAMPLES=2 \
EVAL_USE_CACHE=true \
EVAL_PERSISTENT_DIR=true \
EVAL_DETECTION_MODE=combined \
python notebooks_py/eval_bbox_unified.py

echo ""
echo "=================================================="
echo "Testing GoNoGoNet on CholecSeg8k dataset"
echo "=================================================="

# Test GoNoGoNet on CholecSeg8k
EVAL_DATASET=cholecseg8k \
EVAL_MODEL=gonogo \
EVAL_NUM_SAMPLES=2 \
EVAL_USE_CACHE=true \
EVAL_PERSISTENT_DIR=true \
EVAL_DETECTION_MODE=combined \
python notebooks_py/eval_bbox_unified.py

echo ""
echo "=================================================="
echo "Testing CholeNet on Cholec Organs dataset"
echo "=================================================="

# Test CholeNet on Cholec Organs (its native dataset)
EVAL_DATASET=cholec_organs \
EVAL_MODEL=cholenet \
EVAL_NUM_SAMPLES=2 \
EVAL_USE_CACHE=true \
EVAL_PERSISTENT_DIR=true \
EVAL_DETECTION_MODE=combined \
python notebooks_py/eval_bbox_unified.py

echo ""
echo "=================================================="
echo "Testing GoNoGoNet on Cholec GoNoGo dataset"
echo "=================================================="

# Test GoNoGoNet on Cholec GoNoGo (its native dataset)
EVAL_DATASET=cholec_gonogo \
EVAL_MODEL=gonogo \
EVAL_NUM_SAMPLES=2 \
EVAL_USE_CACHE=true \
EVAL_PERSISTENT_DIR=true \
EVAL_DETECTION_MODE=combined \
python notebooks_py/eval_bbox_unified.py

echo ""
echo "=================================================="
echo "All tests completed!"
echo "=================================================="