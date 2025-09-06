# Storage Structure for Bounding Box Evaluation

## Overview
This document explains how evaluation results and caches are organized to prevent data overwrites when running different evaluation configurations.

## Results Directory Structure

```
results/
└── bbox_{dataset}_{timestamp}/
    ├── zeroshot_separate/           # Zero-shot with separate queries
    │   ├── gpt-4.1/
    │   │   ├── predictions.json
    │   │   └── metrics.json
    │   ├── claude-sonnet-4/
    │   └── gemini-2.0-flash/
    │
    ├── zeroshot_combined/           # Zero-shot with combined query
    │   └── {model_name}/
    │
    ├── fewshot_separate/            # Few-shot with separate queries
    │   └── {model_name}/
    │
    ├── fewshot_combined/            # Few-shot with combined query
    │   └── {model_name}/
    │
    ├── summary_metrics.csv         # Overall comparison table
    └── ablation_results.json        # Ablation study results
```

## Cache Structure

The cache system uses MD5 hashes to create unique keys for each configuration:

```
cache/
└── bbox_eval/
    └── {hash}.json
```

### Cache Key Components
The cache key includes:
1. **Model name** (e.g., "gpt-4.1")
2. **Image index** (e.g., "860")
3. **Detection mode** ("separate" or "combined")
4. **Few-shot status** ("zeroshot" or "fewshot")
5. **Prompt prefix** (first 500 chars)

Example: `gpt-4.1:860:separate:zeroshot:{prompt_text}`

## Key Features

### 1. Non-Overlapping Results
- Each combination of (zero/few-shot) × (separate/combined) has its own directory
- Results from one configuration never overwrite another
- Allows partial runs (e.g., only zero-shot for model A, both for model B)

### 2. Cache Isolation
- Cache keys include all configuration parameters
- Prevents cache hits across different evaluation modes
- Ensures reproducibility when re-running specific configurations

### 3. Timestamp-Based Runs
- Each evaluation run gets a unique timestamp directory
- Preserves historical results
- Easy comparison between different runs

## Usage Examples

### Running Specific Configurations
```python
# Run only zero-shot combined for quick testing
evaluator.evaluate_model(
    model_name="gpt-4.1",
    detection_mode="combined",
    use_fewshot=False
)

# Run full ablation (all 4 combinations)
for mode in ["separate", "combined"]:
    for use_fs in [False, True]:
        evaluator.evaluate_model(
            model_name="gpt-4.1",
            detection_mode=mode,
            use_fewshot=use_fs
        )
```

### Accessing Results
```python
# Load specific configuration results
import json
from pathlib import Path

results_dir = Path("results/bbox_cholecseg8k_20250906_212507")
zeroshot_sep = results_dir / "zeroshot_separate/gpt-4.1/metrics.json"

with open(zeroshot_sep) as f:
    metrics = json.load(f)
```

## Benefits

1. **Incremental Evaluation**: Run different configurations at different times without data loss
2. **Parallel Processing**: Multiple configurations can run simultaneously without conflicts
3. **Easy Comparison**: Clear directory structure makes it easy to compare approaches
4. **Cache Efficiency**: Shared cache across runs, but isolated by configuration
5. **Debugging**: Can re-run specific configurations without affecting others

## Best Practices

1. **Always specify detection_mode**: Be explicit about "separate" vs "combined"
2. **Use consistent timestamps**: All configurations in one experiment should share a timestamp
3. **Document configuration**: Save experiment configuration in the results directory
4. **Clean old caches periodically**: Cache directory can grow large over time