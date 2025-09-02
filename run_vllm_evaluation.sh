#!/bin/bash
# Script to run vLLM evaluation with proper CUDA settings

echo "Setting up environment for vLLM..."

# Set Python multiprocessing to use spawn method
export PYTHONPATH="/shared_data0/weiqiuy/llm_cholec_organ/src:$PYTHONPATH"

# Disable fork safety check
export CUDA_LAUNCH_BLOCKING=1
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"

# Run with spawn method enforced
echo "Running evaluation with vLLM models..."
python3 -c "import multiprocessing; multiprocessing.set_start_method('spawn', force=True); exec(open('notebooks/evaluate_endopoint_models.py').read())" --samples 3 --vlm-only "$@"