# RASO Model Integration

## Date: 2025-01-08

## Summary
Integrated the RASO (Recognize Anything in Surgery) model into the endopoint evaluation framework for organ presence detection across three datasets: CholecSeg8k, CholecOrgans, and CholecGoNoGo.

## Problem Solved
- RASO model was not integrated with the unified bbox evaluation pipeline
- Initial integration showed 0% accuracy due to case sensitivity mismatch between RASO's lowercase output and the evaluation framework's Title Case expectations
- RASO only provides organ presence detection (no bounding boxes) which needed special handling

## Changes Made

### 1. Created RASO Model Wrapper (`src/endopoint/models/raso.py`)
- Added dataset-specific model loaders:
  - `load_raso_cholecseg8k()` - loads model with 13 organ classes
  - `load_raso_cholec_organs()` - loads model with 4 organ classes  
  - `load_raso_cholec_gonogo()` - loads model with 3 zone classes
- Each loader uses the appropriate pretrained weights and tag list file
- Supports custom tag lists via `tag_list` parameter

### 2. Created RASO Adapter (`src/endopoint/models/raso_adapter.py`)
- Implements `ModelAdapter` interface for integration with evaluation framework
- Key features:
  - Extracts organ names from prompts (preserving original capitalization)
  - Maps RASO's lowercase outputs back to original Title Case names
  - Returns `null` for bounding boxes (presence-only detection)
  - Includes caching with image hashing
  - Verbose mode for debugging

### 3. Updated Model Factory (`src/endopoint/models/__init__.py`)
- Added `RASOAdapter` import
- Modified `create_model()` to handle 'raso' model type
- Added `dataset` parameter to support dataset-specific RASO models

### 4. Updated Bbox Evaluator (`src/endopoint/eval/bbox_evaluator.py`)
- Added `dataset_name` parameter to `BoundingBoxEvaluator.__init__()`
- Passes dataset name when creating models to enable proper RASO model selection

### 5. Updated Evaluation Script (`notebooks_py/eval_bbox_unified.py`)
- Passes `dataset_name` to `BoundingBoxEvaluator` for RASO support

### 6. Created Test Scripts
- `debug_raso_eval.py` - Debug script for testing BBoxPrediction parsing
- `debug_raso_combined.py` - Debug script for testing full RASO pipeline
- `test_raso_simple.py` - Simple test script for all three datasets
- `run_raso_all.sh` - Bash script to run RASO on all datasets with configurable samples

## Key Technical Details

### Case Sensitivity Fix
The main issue was that RASO outputs lowercase organ names ("liver", "gallbladder") but the evaluation expects Title Case ("Liver", "Gallbladder"). The solution:

1. Extract both original and lowercase organ names from the prompt
2. Use lowercase names for RASO detection
3. Map results back to original capitalization for the response

```python
def _extract_organs_from_prompt(self, prompt: str) -> Tuple[List[str], List[str]]:
    # Returns (original_names, lowercase_names)
    # e.g., (["Liver", "Gallbladder"], ["liver", "gallbladder"])

def _format_detection_response(self, detected_organs: List[str], 
                              requested_organs: List[str],
                              original_organ_names: List[str]) -> str:
    # Maps RASO's lowercase results back to original names
```

### Dataset-Specific Models
Each dataset uses different RASO weights and label files:
- CholecSeg8k: 13 classes, swin_l architecture
- CholecOrgans: 4 classes, specialized for key organs
- CholecGoNoGo: 3 classes, for safe/unsafe zones

## Results
After integration, RASO achieves the following presence accuracy on 2-sample tests:
- CholecSeg8k: 75.0%
- CholecOrgans: 50.0%  
- CholecGoNoGo: 25.0%

Bounding box IoU metrics are 0.0 as expected since RASO only performs presence detection.

## Usage
```bash
# Run RASO on all datasets (default 10 samples)
./run_raso_all.sh

# Run with custom sample count
EVAL_NUM_SAMPLES=50 ./run_raso_all.sh

# Run on individual dataset
EVAL_DATASET=cholec_organs EVAL_MODEL=raso EVAL_NUM_SAMPLES=50 python3 notebooks_py/eval_bbox_unified.py
```

## Files Modified
- `src/endopoint/models/raso.py` (updated with dataset-specific loaders)
- `src/endopoint/models/raso_adapter.py` (created)
- `src/endopoint/models/__init__.py` (added RASO support)
- `src/endopoint/eval/bbox_evaluator.py` (added dataset_name parameter)
- `notebooks_py/eval_bbox_unified.py` (passes dataset_name)

## Files Created
- `debug_raso_eval.py`
- `debug_raso_combined.py`
- `test_raso_simple.py`
- `run_raso_all.sh`

## Notes
- RASO requires specific pretrained weights from `/shared_data0/weiqiuy/github/hf_repos/raso/`
- Label files are located at `/shared_data0/weiqiuy/github/raso/raso/`
- Results are saved to `bbox_{dataset}_quick/` or `bbox_{dataset}_local_quick/` directories
- The integration preserves exact organ names from label files (no underscore conversion)