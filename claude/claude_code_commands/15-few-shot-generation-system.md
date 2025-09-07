# Few-Shot Generation System

## Overview

A comprehensive system for generating few-shot examples for organ detection tasks, supporting both **separate** (per-organ) and **combined** (multi-organ) evaluation modes.

## Key Innovation: Combined Mode with Greedy Set Cover

The combined mode uses a greedy set cover algorithm to select the minimum number of training images that collectively cover all organ classes. This dramatically reduces the number of few-shot examples needed:

- **CholecSeg8k**: 3 images cover all 12 organs (75% reduction)
- **CholecOrgans**: 3 images cover all 3 organs  
- **CholecGoNoGo**: 3 images cover both classes

## Files Created

### Core Scripts

1. **`notebooks_py/generate_few_shot_sep_comb.py`**
   - Main Python script implementing the generation logic
   - Supports both separate and combined modes
   - Uses greedy set cover algorithm for efficient selection
   - Ensures minimum 3 examples for diversity (configurable)

2. **`notebooks_py/generate_few_shot_all.sh`**
   - Main batch script to generate for all datasets
   - Default: generates BOTH separate and combined modes
   - Processes all three datasets: cholecseg8k_local, cholec_organs, cholec_gonogo
   - Colorized output with progress tracking

3. **`notebooks_py/generate_few_shot_single.sh`**
   - Single dataset script using environment variables
   - Good for CI/CD integration

4. **`notebooks_py/generate_combined_all.sh`**
   - Specialized script for combined mode only
   - Quick regeneration of combined plans

## Algorithm: Greedy Set Cover

The combined mode uses a greedy set cover algorithm to minimize the number of few-shot examples:

```python
def select_examples_from_presence_matrix(presence_matrix, max_examples=None):
    1. Start with empty selection
    2. While organs remain uncovered:
       - Find image covering most uncovered organs
       - Add to selection
       - Update covered organs
    3. If selection < 3 images:
       - Add random examples for diversity
    4. Return selected indices
```

## Generated Files

For each dataset, the system generates:

### Separate Mode Files
- `fewshot_plan_bbox_[N].json` - Bounding box examples per organ
- `fewshot_plan_pointing_[N].json` - Pointing examples per organ
- `presence_matrix_train.npy` - Binary matrix of organ presence

### Combined Mode Files  
- `fewshot_plan_bbox_combined_greedy.json` - Combined bounding box examples
- `fewshot_plan_pointing_combined_greedy.json` - Combined pointing examples (fallback)

## Usage

### Quick Start (Recommended)

Generate both separate and combined examples for all datasets:

```bash
cd notebooks_py
./generate_few_shot_all.sh
```

### Advanced Options

```bash
# Generate with 5 minimum combined examples
./generate_few_shot_all.sh --min-combined 5

# Only combined mode
./generate_few_shot_all.sh --mode combined

# Only specific dataset
./generate_few_shot_all.sh --datasets cholecseg8k_local

# Force regenerate (clear cache)
./generate_few_shot_all.sh --force

# All options
./generate_few_shot_all.sh --mode both --min-combined 3 --force
```

### Environment Variables

For single dataset generation:

```bash
DATASET_NAME=cholec_organs MODE=combined ./generate_few_shot_single.sh
```

## Output Structure

Each dataset's output is saved to:
```
data_info/{dataset_name}_balanced_200/
├── fewshot_plan_bbox_combined_greedy.json    # Combined bbox plan
├── fewshot_plan_bbox_200.json                # Separate bbox plan
├── fewshot_plan_pointing_200.json            # Separate pointing plan
├── presence_matrix_train.npy                 # Organ presence matrix
└── pipeline_summary.json                     # Generation summary
```

## Combined Plan Structure

The combined plan JSON contains:

```json
{
  "metadata": {
    "creation_method": "greedy_set_cover",
    "task_type": "bounding_box",
    "num_examples": 3,
    "organs_covered": 12,
    "min_examples": 3,
    "selected_indices": [860, 4000, 1280]
  },
  "examples": [
    {
      "idx": 860,
      "organs_present": ["Liver", "Gallbladder", ...],
      "bboxes": {
        "Liver": {
          "bboxes": [[x1, y1, x2, y2], ...],
          "num_regions": 2
        },
        ...
      }
    }
  ]
}
```

## Benefits of Combined Mode

1. **Efficiency**: 75% fewer API calls for CholecSeg8k (3 vs 12 examples)
2. **Context**: Models see organs in natural context with other organs
3. **Realism**: Examples show typical multi-organ scenarios
4. **Coverage**: Greedy algorithm ensures all organs are represented
5. **Diversity**: Minimum 3 examples provides varied contexts

## Implementation Details

### Greedy Set Cover Algorithm
- Iteratively selects images that cover the most uncovered organs
- Ensures complete coverage of all organ classes
- Adds random examples if needed to reach minimum threshold

### Presence Matrix
- Binary matrix (N_examples × N_organs)
- Entry [i,j] = 1 if organ j is present in example i
- Used for efficient coverage computation

### Minimum Examples
- Default: 3 examples minimum
- Ensures diversity even when fewer would suffice
- Configurable via `--min-combined` parameter

## Troubleshooting

### Regenerate from Scratch
```bash
./generate_few_shot_all.sh --force
```

### Check Generated Files
```bash
ls -la data_info/*/fewshot_plan_*combined*.json
```

### Verify Coverage
```python
import json
plan = json.load(open('data_info/cholecseg8k_local_balanced_200/fewshot_plan_bbox_combined_greedy.json'))
print(f"Examples: {plan['metadata']['num_examples']}")
print(f"Organs covered: {plan['metadata']['organs_covered']}")
```

## Integration with Evaluation

Use the generated combined plans in evaluation:

```python
# Load combined plan
plan_path = "data_info/cholecseg8k_local_balanced_200/fewshot_plan_bbox_combined_greedy.json"

# Use in evaluation script
python eval_bbox.py \
    --dataset cholecseg8k_local \
    --fewshot-plan $plan_path \
    --mode combined
```

## Performance Impact

Using combined mode with greedy selection:
- **API Cost Reduction**: ~75% for CholecSeg8k
- **Evaluation Speed**: 4x faster (3 calls vs 12)
- **Memory Efficiency**: Fewer examples to store/process
- **Accuracy**: Comparable or better due to contextual learning