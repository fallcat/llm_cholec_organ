# Cell Selection Implementation

## Date: 2025-09-02

## Overview
Implemented cell selection as an alternative to exact (x,y) pointing for organ localization in laparoscopic surgery images. Cell selection divides images into G×G grids and asks models to select cells containing organs.

## Key Changes

### 1. Core Implementation Files

#### Created New Files:
- `src/endopoint/eval/cell_selection.py` - Core cell selection computation
  - `compute_cell_ground_truth()` - Computes ground truth cells from organ masks
  - `compute_cell_metrics()` - Calculates Cell@K, precision, recall, F1
  - `get_cell_labels()` - Generates grid labels (A1-C3 for 3×3, A1-D4 for 4×4)
  - `point_to_cell()` - Converts (x,y) coordinates to cell labels

- `notebooks_py/eval_cell_selection_original_size.py` - Main evaluation script
  - Supports grid sizes G=3 and G=4
  - Supports top-K selection (K=1 or K=3)
  - Evaluates zero-shot and few-shot configurations
  - Computes comprehensive metrics

- `notebooks_py/eval_both_persistent.sh` - Simple evaluation script for both tasks
- `notebooks_py/eval_both_advanced.sh` - Advanced script with command-line options

#### Modified Files:
- `src/endopoint/prompts/builders.py`
  - Added `build_cell_selection_system_prompt()`
  - Added `build_cell_selection_system_prompt_strict()`
  - Added `build_cell_selection_user_prompt()`

- `src/endopoint/eval/parser.py`
  - Added `parse_cell_selection_json()`
  - Added `validate_cell_selection_response()`

- `src/endopoint/eval/pointing.py`
  - Added `run_cell_selection_on_canvas()` function
  - Fixed batch API format for proper cache key generation

- `src/endopoint/models/__init__.py`
  - Added `create_model()` function for model instantiation

- `src/endopoint/utils/io.py`
  - Added `ensure_dir()` function

### 2. Cache Key Collision Fix

#### Problem:
Zero-shot and few-shot configurations were returning identical results due to cache key collisions.

#### Root Cause:
The few-shot examples were using placeholder values (hardcoded cells and presence) instead of computing actual ground truth, making all configurations effectively identical.

#### Solution:
1. Fixed few-shot example generation to compute actual ground truth cells for each organ
2. Ensured few-shot examples are included in the prompt tuple (not system prompt) so they affect the cache key
3. The cache key is generated from the entire prompt content including images

### 3. Persistent Directory Feature

#### Added Environment Variable:
- `EVAL_PERSISTENT_DIR=true` - Uses fixed directory names without timestamps

#### Benefits:
- Skip already evaluated samples when resuming
- Accumulate results across multiple runs
- Add new models without re-evaluating existing ones
- Provides evaluation-level caching independent of API-level caching

#### Directory Structure:
```
results/
├── cell_selection_G3_K1/          # Persistent (no timestamp)
│   ├── zero_shot/
│   ├── fewshot_standard/
│   └── fewshot_hard_negatives/
└── pointing_original/              # Persistent (no timestamp)
```

### 4. Evaluation Scripts

#### Basic Usage:
```bash
# Evaluate with persistent directories (recommended)
EVAL_PERSISTENT_DIR=true EVAL_USE_CACHE=false python3 eval_cell_selection_original_size.py

# Quick test
./eval_both_advanced.sh --quick

# Full evaluation
./eval_both_advanced.sh --samples 20 --all-models
```

#### Environment Variables:
- `EVAL_NUM_SAMPLES` - Number of samples to evaluate
- `EVAL_MODELS` - Comma-separated list of models
- `EVAL_USE_CACHE` - Use API response caching (default: true)
- `EVAL_PERSISTENT_DIR` - Use persistent directories (default: false)
- `EVAL_GRID_SIZE` - Grid size (3 or 4, default: 3)
- `EVAL_TOP_K` - Max cells to predict (1 or 3, default: 1)
- `EVAL_SKIP_ZERO_SHOT` - Skip zero-shot evaluation
- `EVAL_SKIP_FEW_SHOT` - Skip few-shot evaluation

## Metrics Implemented

### Cell Selection Metrics:
- **Presence Accuracy**: Binary classification of organ presence
- **Cell@K**: Whether any predicted cell contains the organ
- **Cell Precision**: Fraction of predicted cells that are correct
- **Cell Recall**: Fraction of ground truth cells that are predicted
- **Cell F1**: Harmonic mean of precision and recall
- **Gated Accuracy**: Combined presence + localization accuracy

### Ground Truth Computation:
- Divides image into G×G grid
- Counts organ pixels in each cell
- Cell is positive if pixels > min_pixels threshold (default: 50)
- Returns set of cells containing the organ

## Testing Commands

```bash
# Test cell selection with cache disabled
EVAL_MODELS='gpt-5-mini' EVAL_NUM_SAMPLES=2 EVAL_USE_CACHE=false python3 eval_cell_selection_original_size.py

# Test with persistent directory
EVAL_PERSISTENT_DIR=true EVAL_MODELS='gpt-5-mini' EVAL_NUM_SAMPLES=5 python3 eval_cell_selection_original_size.py

# Compare both tasks
./eval_both_persistent.sh
```

## Known Issues Resolved

1. ✅ Import errors for missing modules
2. ✅ Dataset adapter API compatibility
3. ✅ Model adapter batch format
4. ✅ Cache key collision between configurations
5. ✅ Few-shot examples using placeholder values

## Future Improvements

1. Add visualization of predicted vs ground truth cells
2. Support for multi-organ cell selection
3. Add confidence scores for cell predictions
4. Implement adaptive grid sizing based on organ size
5. Add cell-level IoU metrics