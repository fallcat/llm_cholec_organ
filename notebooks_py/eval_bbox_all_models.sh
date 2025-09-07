#!/bin/bash

# Bounding Box Quick Test - All Models, All Configurations
# This script runs bbox evaluation for all models and all 4 configurations
# Each model+config is loaded separately for vLLM compatibility

echo "=========================================="
echo "BOUNDING BOX EVALUATION - ALL MODELS"
echo "=========================================="
echo

# Configuration
NUM_SAMPLES=${1:-2}  # Default to 2 samples, can override with first argument
USE_CACHE=${2:-true}  # Default to using cache

echo "Configuration:"
echo "  Samples per model: $NUM_SAMPLES"
echo "  Cache: $USE_CACHE"
echo "  Configurations: Zero-shot Combined, Few-shot Combined, Zero-shot Separate, Few-shot Separate"
echo

# Function to run all 4 configurations for a model
run_model_configs() {
    local model=$1
    local model_display=$2
    
    echo
    echo "=========================================="
    echo "Evaluating: $model_display"
    echo "=========================================="
    
    echo
    echo "  [1/4] Zero-shot Combined..."
    EVAL_MODEL="$model" \
    EVAL_NUM_SAMPLES=$NUM_SAMPLES \
    EVAL_USE_CACHE=$USE_CACHE \
    EVAL_DETECTION_MODE=combined \
    EVAL_USE_FEWSHOT=false \
    python3 eval_bbox_quick_test.py
    
    echo
    echo "  [2/4] Few-shot Combined..."
    EVAL_MODEL="$model" \
    EVAL_NUM_SAMPLES=$NUM_SAMPLES \
    EVAL_USE_CACHE=$USE_CACHE \
    EVAL_DETECTION_MODE=combined \
    EVAL_USE_FEWSHOT=true \
    python3 eval_bbox_quick_test.py
    
    echo
    echo "  [3/4] Zero-shot Separate..."
    EVAL_MODEL="$model" \
    EVAL_NUM_SAMPLES=$NUM_SAMPLES \
    EVAL_USE_CACHE=$USE_CACHE \
    EVAL_DETECTION_MODE=separate \
    EVAL_USE_FEWSHOT=false \
    python3 eval_bbox_quick_test.py
    
    echo
    echo "  [4/4] Few-shot Separate..."
    EVAL_MODEL="$model" \
    EVAL_NUM_SAMPLES=$NUM_SAMPLES \
    EVAL_USE_CACHE=$USE_CACHE \
    EVAL_DETECTION_MODE=separate \
    EVAL_USE_FEWSHOT=true \
    python3 eval_bbox_quick_test.py
    
    echo
    echo "✅ Completed all 4 configurations for $model_display"
}

# API Models
echo "=========================================="
echo "API MODELS"
echo "=========================================="

run_model_configs "gpt-4.1" "GPT-4.1"
run_model_configs "claude-sonnet-4-20250514" "Claude Sonnet 4"
run_model_configs "gemini-2.0-flash" "Gemini 2.0 Flash"

# Local Models (vLLM)
echo
echo "=========================================="
echo "LOCAL MODELS (vLLM)"
echo "=========================================="

run_model_configs "llava-hf/llava-v1.6-mistral-7b-hf" "LLaVA v1.6 Mistral 7B"
run_model_configs "Qwen/Qwen2.5-VL-7B-Instruct" "Qwen2.5-VL 7B"
run_model_configs "mistralai/Pixtral-12B-2409" "Pixtral 12B"
run_model_configs "deepseek-ai/deepseek-vl2" "DeepSeek VL2"
run_model_configs "nvidia/Llama-3.2-11B-Vision-Surgical-CholecT50" "Llama 3.2 11B Vision Surgical"

echo
echo "=========================================="
echo "ALL EVALUATIONS COMPLETE"
echo "=========================================="
echo
echo "Results saved in: results/bbox_cholecseg8k_local_*/"
echo "Each model has 4 summary files:"
echo "  - summary_combined_zeroshot.json"
echo "  - summary_combined_fewshot.json"
echo "  - summary_separate_zeroshot.json"
echo "  - summary_separate_fewshot.json"
echo
echo "To aggregate results across all models and configs:"
echo "  python3 aggregate_bbox_results.py"