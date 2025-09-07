#!/bin/bash

# Bounding Box Evaluation - Single Model, All Configurations
# This script runs all 4 bbox configurations for a single specified model

# Check if model name is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <model_name> [num_samples] [use_cache]"
    echo ""
    echo "Examples:"
    echo "  $0 gpt-4.1                                    # Run with defaults (2 samples, cache enabled)"
    echo "  $0 'claude-sonnet-4-20250514' 5               # Run with 5 samples"
    echo "  $0 'llava-hf/llava-v1.6-mistral-7b-hf' 10 false  # 10 samples, no cache"
    echo ""
    echo "Available models:"
    echo "  API Models:"
    echo "    - gpt-4.1"
    echo "    - claude-sonnet-4-20250514"
    echo "    - gemini-2.0-flash"
    echo "  Local Models (vLLM):"
    echo "    - llava-hf/llava-v1.6-mistral-7b-hf"
    echo "    - Qwen/Qwen2.5-VL-7B-Instruct"
    echo "    - mistralai/Pixtral-12B-2409"
    echo "    - deepseek-ai/deepseek-vl2"
    echo "    - nvidia/Llama-3.2-11B-Vision-Surgical-CholecT50"
    exit 1
fi

# Parse arguments
MODEL=$1
NUM_SAMPLES=${2:-2}  # Default to 2 samples
USE_CACHE=${3:-true}  # Default to using cache

# Create a display name for the model
case "$MODEL" in
    "gpt-4.1")
        MODEL_DISPLAY="GPT-4.1"
        ;;
    "claude-sonnet-4-20250514")
        MODEL_DISPLAY="Claude Sonnet 4"
        ;;
    "gemini-2.0-flash")
        MODEL_DISPLAY="Gemini 2.0 Flash"
        ;;
    "llava-hf/llava-v1.6-mistral-7b-hf")
        MODEL_DISPLAY="LLaVA v1.6 Mistral 7B"
        ;;
    "Qwen/Qwen2.5-VL-7B-Instruct")
        MODEL_DISPLAY="Qwen2.5-VL 7B"
        ;;
    "mistralai/Pixtral-12B-2409")
        MODEL_DISPLAY="Pixtral 12B"
        ;;
    "deepseek-ai/deepseek-vl2")
        MODEL_DISPLAY="DeepSeek VL2"
        ;;
    "nvidia/Llama-3.2-11B-Vision-Surgical-CholecT50")
        MODEL_DISPLAY="Llama 3.2 11B Vision Surgical"
        ;;
    *)
        MODEL_DISPLAY="$MODEL"
        ;;
esac

echo "=========================================="
echo "BOUNDING BOX EVALUATION - SINGLE MODEL"
echo "=========================================="
echo ""
echo "Model: $MODEL_DISPLAY"
echo "Model ID: $MODEL"
echo "Samples: $NUM_SAMPLES"
echo "Cache: $USE_CACHE"
echo "Configurations: All 4 (Zero/Few-shot × Combined/Separate)"
echo ""

# Track start time
START_TIME=$(date +%s)

echo "=========================================="
echo "Starting Evaluations"
echo "=========================================="

# Run all 4 configurations
echo ""
echo "[1/4] Zero-shot Combined..."
echo "----------------------------------------"
EVAL_MODEL="$MODEL" \
EVAL_NUM_SAMPLES=$NUM_SAMPLES \
EVAL_USE_CACHE=$USE_CACHE \
EVAL_DETECTION_MODE=combined \
EVAL_USE_FEWSHOT=false \
python3 eval_bbox_quick_test.py

if [ $? -ne 0 ]; then
    echo "❌ Zero-shot Combined failed"
else
    echo "✅ Zero-shot Combined completed"
fi

echo ""
echo "[2/4] Few-shot Combined..."
echo "----------------------------------------"
EVAL_MODEL="$MODEL" \
EVAL_NUM_SAMPLES=$NUM_SAMPLES \
EVAL_USE_CACHE=$USE_CACHE \
EVAL_DETECTION_MODE=combined \
EVAL_USE_FEWSHOT=true \
python3 eval_bbox_quick_test.py

if [ $? -ne 0 ]; then
    echo "❌ Few-shot Combined failed"
else
    echo "✅ Few-shot Combined completed"
fi

echo ""
echo "[3/4] Zero-shot Separate..."
echo "----------------------------------------"
EVAL_MODEL="$MODEL" \
EVAL_NUM_SAMPLES=$NUM_SAMPLES \
EVAL_USE_CACHE=$USE_CACHE \
EVAL_DETECTION_MODE=separate \
EVAL_USE_FEWSHOT=false \
python3 eval_bbox_quick_test.py

if [ $? -ne 0 ]; then
    echo "❌ Zero-shot Separate failed"
else
    echo "✅ Zero-shot Separate completed"
fi

echo ""
echo "[4/4] Few-shot Separate..."
echo "----------------------------------------"
EVAL_MODEL="$MODEL" \
EVAL_NUM_SAMPLES=$NUM_SAMPLES \
EVAL_USE_CACHE=$USE_CACHE \
EVAL_DETECTION_MODE=separate \
EVAL_USE_FEWSHOT=true \
python3 eval_bbox_quick_test.py

if [ $? -ne 0 ]; then
    echo "❌ Few-shot Separate failed"
else
    echo "✅ Few-shot Separate completed"
fi

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED / 60))
ELAPSED_SEC=$((ELAPSED % 60))

echo ""
echo "=========================================="
echo "EVALUATION COMPLETE"
echo "=========================================="
echo ""
echo "Model: $MODEL_DISPLAY"
echo "Total time: ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
echo ""
echo "Results saved in: results/bbox_cholecseg8k_local_*/"
echo "Summary files:"
echo "  - summary_combined_zeroshot.json"
echo "  - summary_combined_fewshot.json"
echo "  - summary_separate_zeroshot.json"
echo "  - summary_separate_fewshot.json"
echo ""
echo "To run another model:"
echo "  $0 <another_model_name> $NUM_SAMPLES $USE_CACHE"
echo ""
echo "To aggregate results:"
echo "  python3 aggregate_bbox_results.py"