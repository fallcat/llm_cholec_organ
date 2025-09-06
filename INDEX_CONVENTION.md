# Index Convention: Global Indices

## Overview
This codebase uses **global indices** throughout for consistency and reproducibility. A global index refers to the position of a sample in the complete dataset (0 to N-1 where N is the total number of samples).

## Why Global Indices?

1. **Reproducibility**: Papers can reference exact sample numbers (e.g., "evaluated on samples 860, 900, 940...")
2. **Consistency**: Same index always refers to the same sample, regardless of split configuration
3. **Debugging**: Easy to track specific problematic samples across experiments
4. **Cross-split tracking**: Can follow samples if they move between splits in different configurations

## Implementation

### Dataset Adapters

All dataset adapters provide two methods:

```python
# Get by split-relative index (legacy, avoid using)
example = adapter.get_example(split='test', index=5)  # 5th sample in test split

# Get by global index (preferred)
example = adapter.get_example_by_global_index(860)  # Sample #860 in full dataset
```

### Test Indices

The balanced test indices stored in JSON files are **global indices**:

```json
{
  "indices": [860, 900, 940, 980, ...]  // Global indices
}
```

### Evaluation Pipeline

The evaluation pipeline always uses global indices:

```python
# Load test indices (these are global)
with open('balanced_test_indices.json') as f:
    test_indices = json.load(f)['indices']

# Evaluate using global indices
for test_idx in test_indices:
    example = adapter.get_example_by_global_index(test_idx)
    # Process example...
```

### Split Information

To get all global indices for a specific split:

```python
# Get all global indices in test split
test_global_indices = adapter.get_test_indices()

# Or for any split
train_global_indices = adapter._splits['train']  # Internal structure
```

## Converting Between Index Types

If you need to convert:

```python
# Global to split-relative
def global_to_split_relative(global_idx, split, adapter):
    split_indices = adapter._splits[split]
    if global_idx in split_indices:
        return split_indices.index(global_idx)
    else:
        raise ValueError(f"Global index {global_idx} not in {split}")

# Split-relative to global  
def split_relative_to_global(split_idx, split, adapter):
    return adapter._splits[split][split_idx]
```

## Best Practices

1. **Always use global indices** in:
   - Test/validation sets
   - Few-shot example selection
   - Error analysis and debugging
   - Results reporting

2. **Document indices** in papers and reports:
   - "We evaluate on 200 samples with global indices: [list]"
   - Makes results fully reproducible

3. **Cache keys** should include global indices:
   - Ensures cache hits across different runs
   - Example: `model_name:global_idx:prompt_hash`

## Migration Guide

If you have old code using split-relative indices:

```python
# Old (split-relative)
for i in range(len(test_split)):
    example = dataset['test'][i]

# New (global)
test_indices = adapter.get_test_indices()  # or load from JSON
for global_idx in test_indices:
    example = adapter.get_example_by_global_index(global_idx)
```

## Debugging

To debug index issues:

```python
# Check if index is global or split-relative
print(f"Total dataset size: {adapter.total()}")
print(f"Test split size: {adapter.total('test')}")
print(f"Test global indices: {adapter.get_test_indices()[:5]}...")

# If index > split size, it's likely global
if index >= adapter.total('test'):
    print(f"Index {index} is global (test has {adapter.total('test')} samples)")
```