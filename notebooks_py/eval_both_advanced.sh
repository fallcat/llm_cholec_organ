#!/bin/bash
# Advanced script for evaluating both cell selection and pointing
# Supports command-line arguments and resume functionality

# Default configuration
NUM_SAMPLES=20
MODELS='gpt-5-mini'  # Default to single fast model
USE_CACHE=false
USE_PERSISTENT=true
GRID_SIZE=3
TOP_K=1
SKIP_CELL=false
SKIP_POINTING=false
SKIP_ZERO_SHOT=false
SKIP_FEW_SHOT=false

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --models)
            MODELS="$2"
            shift 2
            ;;
        --all-models)
            MODELS='gpt-5-mini,claude-sonnet-4-20250514,gemini-2.5-flash,llava-hf/llava-v1.6-mistral-7b-hf,Qwen/Qwen2.5-VL-7B-Instruct,mistralai/Pixtral-12B-2409,deepseek-ai/deepseek-vl2'
            shift
            ;;
        --grid)
            GRID_SIZE="$2"
            shift 2
            ;;
        --topk)
            TOP_K="$2"
            shift 2
            ;;
        --use-cache)
            USE_CACHE=true
            shift
            ;;
        --no-persistent)
            USE_PERSISTENT=false
            shift
            ;;
        --skip-cell)
            SKIP_CELL=true
            shift
            ;;
        --skip-pointing)
            SKIP_POINTING=true
            shift
            ;;
        --skip-zero-shot)
            SKIP_ZERO_SHOT=true
            shift
            ;;
        --skip-few-shot)
            SKIP_FEW_SHOT=true
            shift
            ;;
        --quick)
            NUM_SAMPLES=5
            MODELS='gpt-5-mini'
            echo "Quick mode: 5 samples with gpt-5-mini"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --samples N          Number of samples to evaluate (default: 20)"
            echo "  --models MODELS      Comma-separated list of models (default: gpt-5-mini)"
            echo "  --all-models         Use all 7 available models"
            echo "  --grid SIZE          Grid size for cell selection: 3 or 4 (default: 3)"
            echo "  --topk K             Top-K for cell selection: 1 or 3 (default: 1)"
            echo "  --use-cache          Enable API response caching"
            echo "  --no-persistent      Don't use persistent directories"
            echo "  --skip-cell          Skip cell selection evaluation"
            echo "  --skip-pointing      Skip pointing evaluation"
            echo "  --skip-zero-shot     Skip zero-shot evaluation"
            echo "  --skip-few-shot      Skip few-shot evaluation"
            echo "  --quick              Quick test with 5 samples and gpt-5-mini"
            echo "  --help               Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Quick test"
            echo "  $0 --quick"
            echo ""
            echo "  # Full evaluation with all models"
            echo "  $0 --samples 100 --all-models"
            echo ""
            echo "  # Test 4x4 grid with top-3 selection"
            echo "  $0 --grid 4 --topk 3"
            echo ""
            echo "  # Only pointing evaluation"
            echo "  $0 --skip-cell --samples 50"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Display configuration
echo "=========================================="
echo "Evaluation Configuration"
echo "=========================================="
echo "Number of samples: $NUM_SAMPLES"
echo "Models: $MODELS"
if [ "$SKIP_CELL" = false ]; then
    echo "Cell selection: Grid ${GRID_SIZE}x${GRID_SIZE}, Top-K: $TOP_K"
else
    echo "Cell selection: SKIPPED"
fi
if [ "$SKIP_POINTING" = false ]; then
    echo "Pointing: ENABLED"
else
    echo "Pointing: SKIPPED"
fi
echo "Persistent directories: $USE_PERSISTENT"
echo "API caching: $USE_CACHE"
if [ "$SKIP_ZERO_SHOT" = true ]; then
    echo "Zero-shot: SKIPPED"
fi
if [ "$SKIP_FEW_SHOT" = true ]; then
    echo "Few-shot: SKIPPED"
fi
echo ""

# Change to notebooks_py directory
cd /shared_data0/weiqiuy/llm_cholec_organ/notebooks_py || exit 1

# Cell Selection Evaluation
if [ "$SKIP_CELL" = false ]; then
    echo "=========================================="
    echo "Cell Selection Evaluation"
    echo "=========================================="
    if [ "$USE_PERSISTENT" = true ]; then
        echo "Output: ../results/cell_selection_G${GRID_SIZE}_K${TOP_K}/"
    else
        echo "Output: ../results/cell_selection_G${GRID_SIZE}_K${TOP_K}_<timestamp>/"
    fi
    echo ""
    
    EVAL_NUM_SAMPLES=$NUM_SAMPLES \
    EVAL_MODELS=$MODELS \
    EVAL_USE_CACHE=$USE_CACHE \
    EVAL_PERSISTENT_DIR=$USE_PERSISTENT \
    EVAL_GRID_SIZE=$GRID_SIZE \
    EVAL_TOP_K=$TOP_K \
    EVAL_SKIP_ZERO_SHOT=$SKIP_ZERO_SHOT \
    EVAL_SKIP_FEW_SHOT=$SKIP_FEW_SHOT \
    python3 eval_cell_selection_original_size.py
    
    if [ $? -ne 0 ]; then
        echo "Error: Cell selection evaluation failed"
        exit 1
    fi
    echo ""
fi

# Pointing Evaluation
if [ "$SKIP_POINTING" = false ]; then
    echo "=========================================="
    echo "Pointing Evaluation"
    echo "=========================================="
    if [ "$USE_PERSISTENT" = true ]; then
        echo "Output: ../results/pointing_original/"
    else
        echo "Output: ../results/pointing_original_<timestamp>/"
    fi
    echo ""
    
    EVAL_NUM_SAMPLES=$NUM_SAMPLES \
    EVAL_MODELS=$MODELS \
    EVAL_USE_CACHE=$USE_CACHE \
    EVAL_PERSISTENT_DIR=$USE_PERSISTENT \
    EVAL_SKIP_ZERO_SHOT=$SKIP_ZERO_SHOT \
    EVAL_SKIP_FEW_SHOT=$SKIP_FEW_SHOT \
    python3 eval_pointing_original_size.py
    
    if [ $? -ne 0 ]; then
        echo "Error: Pointing evaluation failed"
        exit 1
    fi
    echo ""
fi

echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="

# Show results locations
if [ "$USE_PERSISTENT" = true ]; then
    if [ "$SKIP_CELL" = false ]; then
        echo "Cell selection results: ../results/cell_selection_G${GRID_SIZE}_K${TOP_K}/"
        if [ -f "../results/cell_selection_G${GRID_SIZE}_K${TOP_K}/metrics_comparison.txt" ]; then
            echo "  ✓ Metrics summary available"
        fi
    fi
    if [ "$SKIP_POINTING" = false ]; then
        echo "Pointing results: ../results/pointing_original/"
        if [ -f "../results/pointing_original/metrics_comparison.txt" ]; then
            echo "  ✓ Metrics summary available"
        fi
    fi
else
    echo "Check timestamped directories in ../results/"
fi