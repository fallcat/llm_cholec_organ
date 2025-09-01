# Fix Pointing Evaluation Pipeline

## Problem Statement
The pointing evaluation pipeline had several critical issues:
1. Few-shot and zero-shot evaluations were returning identical results
2. Few-shot execution was suspiciously fast (1.35s vs 25.20s for zero-shot)
3. Few-shot examples were not actually being used by the models
4. Cache keys were identical for zero-shot and few-shot, causing incorrect cache hits

## Root Cause Analysis
The investigation revealed that in `src/endopoint/eval/pointing.py`:
- Few-shot examples were being prepared but never passed to the model
- Both zero-shot and few-shot were calling: `model([(user_prompt, img_pil)], system_prompt=system_prompt)[0]`
- This resulted in identical cache keys and API calls for both modes

## Changes Made

### 1. Fixed Few-Shot Implementation
**File: `src/endopoint/eval/pointing.py`**

Changed the `run_pointing_on_canvas` function to properly pass few-shot examples:

```python
# Before (incorrect):
if few_shot_examples:
    # Examples were prepared but ignored
    response = model([(user_prompt, img_pil)], system_prompt=system_prompt)[0]

# After (correct):
if few_shot_examples:
    # Build a single tuple with all few-shot examples and the current query
    prompt_parts = []
    prompt_parts.append("Here are some examples:\n")
    
    for i, (ex_img, ex_response) in enumerate(few_shot_examples, 1):
        ex_pil = tensor_to_pil(ex_img)
        ex_prompt = f"\nExample {i}: {user_prompt_builder(ex_response['name'])}"
        prompt_parts.append(ex_prompt)
        prompt_parts.append(ex_pil)
        # Add expected response...
        
    prompt_parts.append(f"\nNow for the actual query: {user_prompt}")
    prompt_parts.append(img_pil)
    
    response = model([tuple(prompt_parts)], system_prompt=system_prompt)[0]
```

This ensures:
- All few-shot example images and texts are included
- Cache key is different (includes all example data)
- Model actually sees and learns from the examples

### 2. Added Cache Control
**Files Modified:**
- `src/endopoint/eval/evaluator.py`
- `notebooks_py/eval_pointing.py`

Added `use_cache` parameter throughout the pipeline:

```python
# In PointingEvaluator.__init__
def __init__(self, ..., use_cache: bool = True):
    self.use_cache = use_cache

# In load_model calls
model = self.load_model(model_name, use_cache=self.use_cache)

# In main function
def main(num_samples=None, models=None, use_cache=True):
    ...
    evaluator = PointingEvaluator(..., use_cache=use_cache)
```

### 3. Created Cache Management Utilities
**File: `notebooks_py/clear_cache.py`**

Created a utility to clear the model cache when needed:
```python
def clear_cache():
    cache_dir = Path.home() / ".cache" / "endopoint" / "models"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
```

### 4. Enhanced Evaluation Script
**File: `notebooks_py/eval_pointing.py`**

Added features for better control and testing:
- Linspace-based sample selection for even distribution
- Environment variable support for configuration
- Cache control via `EVAL_USE_CACHE` environment variable
- Notebook-friendly design (no argparse issues)

## Usage Examples

### Clear stale cache (one-time):
```bash
python3 clear_cache.py
```

### Run without cache:
```python
# In notebook
main(num_samples=5, models=['gpt-4o-mini'], use_cache=False)

# From command line
EVAL_USE_CACHE=false python3 eval_pointing.py
```

### Quick test with subset:
```python
# 5 samples with linspace selection
main(num_samples=5, models=['gpt-4o-mini'])
```

## Key Improvements
1. **Correct Few-Shot Learning**: Models now receive and use few-shot examples with images
2. **Unique Cache Keys**: Zero-shot and few-shot have different cache keys
3. **Cache Control**: Can disable cache for testing or clear stale entries
4. **Flexible Evaluation**: Support for sample subsets, specific models, and various configurations
5. **Notebook Compatibility**: Works seamlessly in both notebooks and command line

## Verification
After the fix:
- Zero-shot and few-shot show different results
- Execution times are similar for both modes (no suspicious 1.35s cache hits)
- Few-shot examples with images are properly utilized
- Cache can be controlled or bypassed as needed

## Related Files Changed
- `src/endopoint/eval/pointing.py` - Core fix for few-shot implementation
- `src/endopoint/eval/evaluator.py` - Added cache control
- `src/endopoint/datasets/cholecseg8k.py` - Added label2id property
- `src/endopoint/models/base.py` - Fixed Python 3.6 compatibility (Protocol → ABC)
- `src/endopoint/datasets/base.py` - Fixed Python 3.6 compatibility
- `notebooks_py/eval_pointing.py` - Enhanced with cache control and linspace selection
- `notebooks_py/clear_cache.py` - Cache management utility