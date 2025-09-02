# JSON Metrics Output and VLM Fixes

Date: 2025-01-09

## Summary
Added JSON output for metrics comparison, fixed DeepSeek-VL2 dtype issues, and organized test files into a dedicated folder.

## Changes Made

### 1. JSON Metrics Output

#### Modified Files
- `src/endopoint/eval/enhanced_evaluator.py`

#### Changes
Added JSON output alongside text output in `save_comparison_table()` method:

```python
# Save both text and JSON versions
comparison_file_txt = self.output_dir / "metrics_comparison.txt"
comparison_file_json = self.output_dir / "metrics_comparison.json"
```

#### JSON Structure
```json
{
  "metadata": {
    "timestamp": "...",
    "canvas_width": 224,
    "canvas_height": 224,
    "organ_names": [...],
    "models": [...]
  },
  "results": {
    "model_name": {
      "evaluation_type": {
        "n_examples": N,
        "per_organ": [
          {
            "id": 1,
            "label": "Liver",
            "TP": 2,
            "FN": 1,
            "TN": 0,
            "FP": 0,
            "PresenceAcc": 0.667,
            "Hit@Point|Present": 0.5,
            "GatedAcc": 0.333,
            "Precision": 1.0,
            "Recall": 0.667,
            "F1": 0.8
          }
        ],
        "aggregate": {
          "total_TP": 5,
          "total_FN": 1,
          "total_TN": 0,
          "total_FP": 0,
          "macro_presence_acc": 0.833,
          "macro_hit_rate": 0.583,
          "macro_gated_acc": 0.5,
          "macro_precision": 1.0,
          "macro_recall": 0.833,
          "macro_f1": 0.9,
          "micro_precision": 1.0,
          "micro_recall": 0.833,
          "micro_f1": 0.909
        }
      }
    }
  }
}
```

### 2. DeepSeek-VL2 Dtype Fixes

#### Problem
DeepSeek-VL2 was failing with dtype mismatches:
- "Input type (c10::BFloat16) and bias type (c10::Half) should be the same"
- "BatchCollateOutput.to() missing 1 required positional argument: 'device'"

#### Solution (in `src/endopoint/models/vllm.py`)
1. Load model without specifying dtype initially
2. Convert entire model to float16 with `.half()`
3. Fixed BatchCollateOutput device handling
4. Use autocast context for prepare_inputs_embeds
5. Force inputs_embeds to float16

```python
# Load model without specifying dtype, then convert
self.model = AutoModelForCausalLM.from_pretrained(
    self.model_name,
    trust_remote_code=True,
)

# Convert entire model to float16 to avoid dtype mismatches
self.model = self.model.half()

if torch.cuda.is_available():
    self.model = self.model.cuda()
```

### 3. Test Organization

#### Created `tests/` Directory
Moved all test files from root to `tests/`:
- `test_dataset_split.py`
- `test_endopoint_loading.py`
- `test_final_integration.py`
- `test_json_metrics.py`
- `test_llava_debug.py`
- `test_llava_next.py`
- `test_llava_simple.py`
- `test_vllm_fixed.py`
- `test_vllm_fix.py`
- `test_vllm_with_spawn.py`
- `test_vlm_models.py`
- `analyze_metrics_json.py`

#### Created Test Utilities
- `tests/test_json_metrics.py` - Validates JSON output functionality
- `tests/analyze_metrics_json.py` - Analyzes and visualizes metrics from JSON

### 4. Test Commands for VLM Models

#### LLaVA (vLLM backend)
```bash
EVAL_QUICK_TEST=true EVAL_NUM_SAMPLES=3 EVAL_MODELS='llava-hf/llava-v1.6-mistral-7b-hf' python3 notebooks_py/eval_pointing_original_size.py
```

#### Qwen2.5-VL (transformers backend)
```bash
EVAL_QUICK_TEST=true EVAL_NUM_SAMPLES=3 EVAL_MODELS='Qwen/Qwen2.5-VL-7B-Instruct' python3 notebooks_py/eval_pointing_original_size.py
```

#### Pixtral (vLLM backend)
```bash
EVAL_QUICK_TEST=true EVAL_NUM_SAMPLES=3 EVAL_MODELS='mistralai/Pixtral-12B-2409' python3 notebooks_py/eval_pointing_original_size.py
```

#### DeepSeek-VL2 (transformers backend)
```bash
EVAL_QUICK_TEST=true EVAL_NUM_SAMPLES=3 EVAL_MODELS='deepseek-ai/deepseek-vl2-tiny' python3 notebooks_py/eval_pointing_original_size.py
```

## Benefits

1. **JSON Metrics**: Enables programmatic analysis, plotting, and comparison of results
2. **DeepSeek Fixes**: Resolves dtype inconsistencies that prevented model from running
3. **Test Organization**: Cleaner project structure with all tests in dedicated folder
4. **Analysis Tools**: Easy-to-use scripts for analyzing evaluation results

## Usage Examples

### Analyze Metrics
```python
import json

# Load metrics
with open('results/pointing_original_20250109/metrics_comparison.json') as f:
    data = json.load(f)

# Access model performance
for model in data['results']:
    metrics = data['results'][model]['zero_shot']['aggregate']
    print(f"{model}: F1={metrics['macro_f1']:.3f}")
```

### Command Line Analysis
```bash
# Analyze most recent results
python tests/analyze_metrics_json.py

# Analyze specific results
python tests/analyze_metrics_json.py results/pointing_original_20250109/metrics_comparison.json
```

## Outstanding Issues

1. DeepSeek-VL2 may still have occasional dtype issues depending on PyTorch/CUDA versions
2. Some VLM models require significant VRAM (12B+ parameters)
3. vLLM backend requires specific CUDA configurations

## Next Steps

1. Add plotting functionality using matplotlib/seaborn
2. Create comparison charts across models
3. Add statistical significance testing
4. Implement caching for VLM model outputs