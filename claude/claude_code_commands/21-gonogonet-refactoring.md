# GoNoGoNet Model Refactoring

## Date: 2025-09-08

## Overview
Refactored all GoNoGo model references to use the consistent naming convention "GoNoGoNet" throughout the codebase. This ensures clarity that we're referring to the GoNoGoNet neural network model, not just a generic "gonogo" concept.

## Changes Made

### 1. File Renaming
- `src/endopoint/models/gonogo.py` → `src/endopoint/models/gonogonet.py`
- `src/endopoint/models/gonogo_adapter.py` → `src/endopoint/models/gonogonet_adapter.py`

### 2. Class Renaming
- `GoNoGoAdapter` → `GoNoGoNetAdapter`

### 3. Model Name Updates
- Default model name changed from `"gonogo"` to `"gonogonet"`
- Cache directory changed from `/cache/gonogo` to `/cache/gonogonet`
- Results now save to `gonogonet/` directories instead of `gonogo/`

### 4. Files Modified

#### Python Files
- `src/endopoint/models/__init__.py`
  - Updated import: `from .gonogonet_adapter import GoNoGoNetAdapter`
  - Updated factory: Creates `GoNoGoNetAdapter` for "gonogonet" model ID
  - Updated exports in `__all__`

- `src/endopoint/models/gonogonet_adapter.py`
  - Import: `from .gonogonet import GoNoGoNet, load_gonogo_model`
  - Class name: `GoNoGoNetAdapter`
  - Default model_name: `"gonogonet"`
  - Default cache_dir: `/cache/gonogonet`

- `debug_models.py`
  - Import: `from endopoint.models.gonogonet import GoNoGoNet, load_gonogo_model`

- `tests/test_gonogo_model.py`
  - Imports updated to use `gonogonet` and `GoNoGoNetAdapter`

- `tests/test_gonogo_masks.py`
  - Imports updated to use `gonogonet` and `GoNoGoNetAdapter`

#### Shell Scripts
- `run_cholenet_gonogo_full.sh`
  - `EVAL_BATCH_MODELS="cholenet,gonogonet"`

- `run_all_cholenet_gonogo_eval.sh`
  - All `run_eval` calls use `gonogonet` instead of `gonogo`

- `notebooks_py/eval_bbox_unified.py`
  - Default models: `'cholenet,gonogonet'`
  - Model comparison: `elif model == "gonogonet":`

#### Configuration Files
- `run_cholenet_gonogo_with_summary.py`
  - `MODELS = ["cholenet", "gonogonet"]`
  - Capabilities dictionary uses `"gonogonet"` key

- `test_model_discovery.py`
  - MODEL_NAME_MAPPING includes `"gonogonet": "GoNoGoNet"`

## Migration Guide

### For Existing Results
If you have existing results in `gonogo/` directories, rename them:
```bash
# Rename all gonogo directories to gonogonet
mv results/bbox_cholecseg8k_local_quick/zeroshot_combined/gonogo \
   results/bbox_cholecseg8k_local_quick/zeroshot_combined/gonogonet

mv results/bbox_cholec_organs_quick/zeroshot_combined/gonogo \
   results/bbox_cholec_organs_quick/zeroshot_combined/gonogonet

mv results/bbox_cholec_gonogo_quick/zeroshot_combined/gonogo \
   results/bbox_cholec_gonogo_quick/zeroshot_combined/gonogonet
```

### For Running Evaluations
The evaluation scripts now use `gonogonet`:
```bash
# Full evaluation
./run_cholenet_gonogo_full.sh

# Single evaluation
EVAL_MODEL=gonogonet EVAL_DATASET=cholec_gonogo EVAL_NUM_SAMPLES=200 \
  python notebooks_py/eval_bbox_unified.py
```

## Testing
Two test scripts verify the refactoring:

1. `test_gonogonet_refactor.py` - Tests the refactoring is complete
2. `test_gonogonet_changes.py` - Verifies all references are updated

Run them with:
```bash
python3 test_gonogonet_refactor.py
python3 test_gonogonet_changes.py
```

## Rationale
The refactoring improves code clarity by:
1. Using consistent "GoNoGoNet" naming that clearly indicates this is a neural network model
2. Avoiding confusion between the model and the dataset concepts
3. Aligning with naming conventions for other models (CholeNet, not "chole")
4. Making the codebase more maintainable and understandable

## Impact
- All new evaluations will save results to `gonogonet/` directories
- The model is now consistently referred to as GoNoGoNet throughout documentation and code
- No functional changes - only naming consistency improvements

## Related Files
- Main notebook: `notebooks/main_table_results_new.ipynb` updated to discover models dynamically
- Dataset adapter: `CholecGoNoGoAdapter` remains unchanged (it's for the dataset, not the model)