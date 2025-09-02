# Tests Directory

This directory contains test scripts for the llm_cholec_organ project.

## Test Files

### Model Testing
- `test_vlm_models.py` - Test Vision-Language Models (LLaVA, Qwen, Pixtral, DeepSeek)
- `test_llava_next.py` - Test LLaVA-NeXT model specifically
- `test_llava_simple.py` - Simple LLaVA test without vLLM
- `test_llava_debug.py` - Debug script for LLaVA issues
- `test_vllm_fixed.py` - Test vLLM with fixes
- `test_vllm_fix.py` - vLLM fix validation
- `test_vllm_with_spawn.py` - Test vLLM with multiprocessing spawn

### Integration Testing
- `test_final_integration.py` - Full integration test
- `test_endopoint_loading.py` - Test endopoint package loading
- `test_dataset_split.py` - Test dataset splitting functionality

### Metrics & Analysis
- `test_json_metrics.py` - Test JSON metrics output functionality
- `analyze_metrics_json.py` - Analyze metrics from JSON output files

## Running Tests

From the project root directory:

```bash
# Run a specific test
python tests/test_vlm_models.py

# Test JSON metrics functionality
python tests/test_json_metrics.py

# Analyze metrics from evaluation results
python tests/analyze_metrics_json.py results/pointing_original_*/metrics_comparison.json
```

## Quick Model Tests

Test individual VLM models:

```bash
# LLaVA
python tests/test_llava_next.py

# All VLM models
python tests/test_vlm_models.py
```