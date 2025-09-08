# Changes Since Last Commit

## Last Commit
- **Commit Hash**: c3f0b71
- **Message**: feat: Implement unified bbox evaluation system with naive baseline and bootstrap CI
- **Date**: Latest on main branch

## Overview of Changes
This document tracks all changes made since the last commit, primarily focused on extending naive baseline support and fixing dataset configurations.

## Modified Files

### 1. `notebooks/main_table_result.ipynb`
**Purpose**: Extended results visualization to support all three datasets and multiple naive baselines

**Key Changes**:
- Added support for two naive baseline strategies:
  - `Naive (Full Box)`: Always predicts entire image as bounding box
  - `Naive (Random Box)`: Generates random bounding boxes with constraints
- Extended to load results from all three datasets:
  - CholecSeg8k (200 samples)
  - CholecOrgans (140 samples) 
  - CholecGoNoGo (151 samples)
- Updated `MODEL_NAME_MAPPING` to include naive baseline variants
- Modified loading logic to handle multiple dataset directories
- Added dataset-specific result aggregation

### 2. `notebooks_py/eval_bbox_naive_baseline.py`
**Purpose**: Fixed naive baseline implementation to remove "cheating" and support all datasets

**Key Changes**:
- Fixed presence prediction to not use ground truth (was "cheating")
  - Added `presence_mode="all"` to always predict all organs present
  - Removed default "perfect" mode that used oracle presence
- Added `box_mode` parameter with two strategies:
  - `"full"`: Predict entire image as bbox
  - `"random"`: Generate random valid bboxes (x2>x1, y2>y1)
- Added support for CholecGoNoGo dataset
- **CRITICAL FIX**: Corrected CholecGoNoGo image dimensions:
  - Was: 854x480 (incorrect)
  - Now: 640x384 (correct, verified from source)
- Added deterministic random seed for reproducible random boxes

### 3. `notebooks/test_models/raso.ipynb`
**Purpose**: Testing notebook for RASO model integration

**Key Changes**:
- Added new experimental cells for testing RASO model
- Expanded evaluation pipeline for additional model testing
- Added visualization and debugging utilities

### 4. Output Files Modified
- `notebooks_py/cholec_gonogo_api.out`: Updated with new run outputs
- `notebooks_py/run_cholec_organs_zeroshot_only_api.out`: Added new evaluation results

## New Files Created

### 1. `notebooks/collect_all_labels.ipynb`
**Purpose**: Collect all ground truth labels from three datasets

**Status**: Created but needs update to collect ID2LABEL text labels instead of just lowercase organ names

**Current Implementation**:
- Loads all three dataset adapters
- Extracts organ presence and bounding boxes
- Saves to JSON/pickle formats
- Currently saves lowercase organ names

**TODO**: Update to extract actual ID2LABEL mappings like:
- "Go (Safe to Incise)" instead of "go zone"
- "NoGo (Unsafe to Incise)" instead of "no-go zone"

### 2. `check_dataset_sizes.py`
**Purpose**: Utility script to verify actual image dimensions and dataset properties

**Key Findings**:
- Confirmed CholecSeg8k: 854x480, 12 organs
- Confirmed CholecOrgans: 640x384, 3 organs  
- **Discovered CholecGoNoGo: 640x384, 2 organs** (not 854x480 as incorrectly documented)

### 3. Data and Result Directories
- `data_info/`: Contains balanced indices and label collections
- `results/`: Evaluation results for all models and datasets
- `vis/`: Visualizations generated during evaluation

## Critical Issues Fixed

### 1. Naive Baseline "Cheating"
**Problem**: Original implementation used ground truth presence (oracle)
**Solution**: Implemented `presence_mode="all"` to always predict all organs present

### 2. CholecGoNoGo Wrong Dimensions
**Problem**: Documentation stated 854x480 but actual images are 640x384
**Solution**: Updated all references to use correct 640x384 dimensions
**Impact**: All CholecGoNoGo experiments may need to be rerun with correct dimensions

### 3. Random Box Generation
**Problem**: No baseline for random box predictions
**Solution**: Added deterministic random box generation with seed-based reproducibility

## Pending Tasks

1. **Update `collect_all_labels.ipynb`**: 
   - Extract ID2LABEL text labels instead of lowercase names
   - Save full descriptive labels from dataset adapters

2. **Rerun CholecGoNoGo Experiments**:
   - All prior experiments used wrong image dimensions
   - Need to rerun with correct 640x384 size

3. **Complete Naive Baseline Evaluation**:
   - Run `naive_baseline_all_random` for all datasets
   - Ensure results are saved in correct directories

## Impact Assessment

### High Impact Changes
- CholecGoNoGo dimension fix affects all prior experiments
- Naive baseline fix changes baseline performance metrics

### Medium Impact Changes  
- Multi-dataset support in main results notebook
- New baseline strategies provide better comparisons

### Low Impact Changes
- Test notebook additions
- Output file updates

## Next Steps

1. Complete the label collection notebook update for ID2LABEL text labels
2. Rerun all CholecGoNoGo experiments with correct dimensions
3. Generate complete results table with all baselines and datasets
4. Document final performance metrics

## Commands to Reproduce Changes

```bash
# Check current status
git status
git diff --stat

# Run naive baseline for all datasets
cd notebooks_py
python eval_bbox_naive_baseline.py --dataset cholecseg8k --mode all --box full
python eval_bbox_naive_baseline.py --dataset cholec_organs --mode all --box full  
python eval_bbox_naive_baseline.py --dataset cholec_gonogo --mode all --box full

# Generate results table
cd ../notebooks
jupyter nbconvert --to notebook --execute main_table_result.ipynb
```

## Notes
- All changes maintain backward compatibility with existing code
- Bootstrap confidence intervals remain at 1000 samples
- Model name mappings preserve legacy "Native Baseline" for compatibility