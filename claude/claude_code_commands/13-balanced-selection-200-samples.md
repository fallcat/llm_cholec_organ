# 13. Balanced Selection with 200 Samples and 30% Minimum Quota

## Date: 2025-01-06

## Overview
Updated the balanced selection implementation to select 200 test samples with a 30% minimum quota for rare classes, and documented the methodology in the paper.

## Key Changes

### 1. Fixed Minimum Quota Implementation
**Problem**: The balanced selection was only giving rare classes 20 samples instead of a percentage-based quota.

**Solution**: Updated `auto_configure_selection_params` to use percentage-based quotas:
- Started with 20% (40 samples out of 200)
- Tested 30% (60 samples)
- Tested 40% (80 samples) 
- **Settled on 30% (60 samples)** as the optimal balance

**Files Modified**:
- `src/endopoint/fewshot/analysis.py`:
  ```python
  'min_quota_rare': int(0.30 * n_test_samples),  # 30% of test samples
  ```

### 2. Created Refactored Analysis Module
**Purpose**: Move repetitive balance analysis code from notebooks to reusable modules.

**New Components**:
- `src/endopoint/fewshot/analysis.py`:
  - `DatasetBalanceAnalyzer` class for dataset balance analysis
  - `auto_configure_selection_params()` for automatic parameter configuration
  - `summarize_fewshot_plan()` for plan statistics

**Integration**:
- `src/endopoint/fewshot/unified.py`:
  - Added `run_balanced_selection_pipeline()` method for complete workflow
  - Added `get_analyzer()` method to access analyzer
  - Enhanced logging to show configuration parameters

### 3. Updated Notebooks

#### `notebooks/test_unified_fewshot_simplified.ipynb`
- Added `force_regenerate` parameter to clear old cache
- Updated to show 30% minimum quota (60 samples)
- Added verification to check if 200 samples were actually selected
- Enhanced configuration display

#### `notebooks/test_unified_fewshot.ipynb`
- Fixed duplicate configuration cells
- Added proper test execution cells for each dataset
- Updated test function with correct quota calculation

### 4. Updated Python Script
**File**: `notebooks_py/prepare_fewshot_examples.py`

Completely rewrote to use refactored modules:
- Uses `UnifiedFewShotSelector` with pipeline method
- Selects 200 balanced test samples with 30% minimum quota
- Builds few-shot plans for both pointing and bbox tasks
- Creates comprehensive visualizations
- Generates summary statistics

**Key Configuration**:
```python
CONFIG = {
    "n_test_samples": 200,  # Updated from 100
    "min_quota_rare": 60,   # 30% of 200
    ...
}
```

### 5. Paper Documentation

#### Method Section (`sections/03-method.tex`)
Updated "Data Curation: Target-Aware Balanced Subset" with:
- 200-frame subset specification
- Video-level splits to avoid leakage
- Target prevalence formula:
  ```latex
  t_k = max(p_k, 0.30) for rare classes
        min(p_k, 0.90) for ubiquitous non-guaranteed classes
        p_k otherwise
  ```
- Specific rare classes: Blood, Cystic Duct, Hepatic Vein, Liver Ligament
- Co-occurrence note explaining inevitable increase in always-present tissues

#### Appendix (`sections/appendix/algorithm_balanced_selection.tex`)
Added comprehensive documentation:
- **Selection Objective and Rationale**: Mathematical formulation
- **Weighted deviation minimization**: With rare class weighting
- **Implementation details**: 30% quota = 60 samples for N=200
- **Actual results**: Showing achieved distributions
  - Rare classes: 12% → 30%, 4.1% → 30%, etc.
  - Balance improvement: StdDev from 37.2% to 31.8%

#### Table Caption (`figures/tab_main_table.tex`)
Updated caption to include subset policy:
```latex
"200-frame balanced subset where rare classes are boosted to ≥30% 
prevalence and ubiquitous classes are capped (e.g., Abdominal Wall ≤90%)"
```

## Results Achieved

### Distribution Improvements (200 samples)
| Class | Original | Selected | Change |
|-------|----------|----------|--------|
| Blood | 12.0% | 30.0% | ⬆️ +18% |
| Cystic Duct | 4.1% | 30.0% | ⬆️ +25.9% |
| Hepatic Vein | 5.5% | 30.0% | ⬆️ +24.5% |
| Liver Ligament | 4.2% | 30.0% | ⬆️ +25.8% |
| Abdominal Wall | 87.1% | 90.0% | ➡️ +2.9% |
| Liver | 100% | 100% | ➡️ 0% |
| Fat | 94.4% | 99.5% | ➡️ +5.1% |

### Balance Metrics
- **Original StdDev**: 37.2%
- **Selected StdDev**: 31.8%
- **Balance Improvement**: 14.5%

## Technical Details

### Cache Management
- Cache files store 200 indices with proper metadata
- `force_regenerate=True` clears old cache when parameters change
- File naming: `balanced_test_indices_advanced_200.json`

### Video-Level Constraints
- Maximum 2 frames per video
- 13 training videos → up to 26 frames per video theoretically
- Prevents temporal correlation in test set

### Presence Definition
- `min_pixels = 50` threshold
- Filters out annotation noise and tiny slivers
- Ensures meaningful organ presence

## Lessons Learned

1. **Percentage vs Fixed Quotas**: Always use percentage-based quotas for scalability
2. **Cache Invalidation**: Important to clear cache when algorithm parameters change
3. **Co-occurrence Effects**: Boosting rare classes inevitably increases common anatomy
4. **Balance Trade-offs**: 30% minimum provides good balance without over-constraining

## Next Steps

- Run full evaluation with the 200-sample balanced test set
- Compare model performance on rare vs common classes
- Analyze which models benefit most from balanced selection
- Consider extending to other surgical datasets

## Files Changed Summary

```
Modified:
- src/endopoint/fewshot/analysis.py (new)
- src/endopoint/fewshot/unified.py
- src/endopoint/fewshot/__init__.py
- notebooks/test_unified_fewshot_simplified.ipynb
- notebooks/test_unified_fewshot.ipynb
- notebooks_py/prepare_fewshot_examples.py
- sections/03-method.tex
- sections/appendix/algorithm_balanced_selection.tex
- figures/tab_main_table.tex
```

## Command to Reproduce

```python
# In notebook or script
from endopoint.datasets import build_dataset
from endopoint.fewshot import UnifiedFewShotSelector

dataset = build_dataset("cholecseg8k_local", 
                        data_dir="/shared_data0/weiqiuy/datasets/cholecseg8k")

selector = UnifiedFewShotSelector(
    dataset=dataset,
    output_dir="./data_info/cholecseg8k_balanced_200",
    n_test_samples=200,
    seed=42,
    cache_enabled=True
)

results = selector.run_balanced_selection_pipeline(
    split="train",
    visualize=True,
    save_summary=True
)
```

This will generate 200 balanced test samples with 30% minimum quota for rare classes.