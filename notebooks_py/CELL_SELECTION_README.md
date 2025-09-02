# Cell Selection Implementation for Organ Localization

## Overview

Cell selection is an alternative to exact (x,y) pointing for organ localization. Instead of asking models for precise pixel coordinates, the image is divided into a G×G grid and the model selects which cell(s) contain the organ. This approach is often more robust for Vision-Language Models (VLMs).

## Implementation Status

✅ **Completed Components:**

1. **Core Module** (`src/endopoint/eval/cell_selection.py`)
   - Ground truth computation from organ masks
   - Cell metrics (Cell@K, precision, recall, F1)
   - Coordinate conversion utilities
   - Visualization helpers

2. **Prompt Builders** (`src/endopoint/prompts/builders.py`)
   - `build_cell_selection_system_prompt()` - Standard prompt
   - `build_cell_selection_system_prompt_strict()` - Strict version
   - `build_cell_selection_user_prompt()` - User prompt

3. **Parser** (`src/endopoint/eval/parser.py`)
   - `parse_cell_selection_json()` - Parse model responses
   - Validation and consistency enforcement
   - Fallback regex extraction for non-JSON responses

4. **Evaluation Script** (`notebooks_py/eval_cell_selection_original_size.py`)
   - Full evaluation pipeline
   - Support for different grid sizes (G=3, G=4)
   - Support for different top-K values (K=1, K=3)
   - Zero-shot and few-shot evaluation modes

5. **Integration** (`src/endopoint/eval/pointing.py`)
   - `run_cell_selection_on_canvas()` - Main evaluation function
   - Integration with existing model adapters

## Usage

### Prerequisites

Ensure you have the following Python packages installed:
```bash
pip install numpy torch pandas tqdm datasets pillow
```

### Running Evaluations

```bash
cd notebooks_py/

# Quick test with 5 samples (3x3 grid, top-1)
EVAL_QUICK_TEST=true EVAL_GRID_SIZE=3 EVAL_TOP_K=1 python3 eval_cell_selection_original_size.py

# Full evaluation with 3x3 grid, top-1
EVAL_GRID_SIZE=3 EVAL_TOP_K=1 python3 eval_cell_selection_original_size.py

# Evaluation with 4x4 grid, top-3
EVAL_GRID_SIZE=4 EVAL_TOP_K=3 python3 eval_cell_selection_original_size.py

# Specific models only
EVAL_MODELS='gpt-5-mini,claude-4-sonnet' EVAL_GRID_SIZE=3 EVAL_TOP_K=1 python3 eval_cell_selection_original_size.py

# Skip zero-shot (only few-shot)
EVAL_SKIP_ZERO_SHOT=true python3 eval_cell_selection_original_size.py

# Skip few-shot (only zero-shot)
EVAL_SKIP_FEW_SHOT=true python3 eval_cell_selection_original_size.py
```

### Environment Variables

- `EVAL_NUM_SAMPLES`: Number of samples to evaluate (default: all)
- `EVAL_MODELS`: Comma-separated list of models (default: all 7 models)
- `EVAL_USE_CACHE`: Whether to use cached responses (default: true)
- `EVAL_QUICK_TEST`: Quick test mode with 5 samples (default: false)
- `EVAL_SKIP_ZERO_SHOT`: Skip zero-shot evaluation (default: false)
- `EVAL_SKIP_FEW_SHOT`: Skip few-shot evaluation (default: false)
- `EVAL_GRID_SIZE`: Grid size - 3 or 4 (default: 3)
- `EVAL_TOP_K`: Maximum cells to predict - 1 or 3 (default: 1)

## Grid Configuration

### 3×3 Grid (G=3)
- 9 cells total: A1, A2, A3, B1, B2, B3, C1, C2, C3
- For 224×224 images: each cell is ~74×74 pixels
- Good for: General organ detection, balanced precision/recall

### 4×4 Grid (G=4)
- 16 cells total: A1-A4, B1-B4, C1-C4, D1-D4
- For 224×224 images: each cell is 56×56 pixels
- Good for: More precise localization, smaller organs

### Top-K Selection
- K=1: Single cell prediction (highest confidence)
- K=3: Up to 3 cells (for larger or elongated organs)

## JSON Response Format

Models are expected to return JSON in this format:
```json
{
  "name": "Liver",
  "present": 1,
  "cells": ["B2", "B3"]
}
```

Rules:
- `present`: 0 or 1 (binary detection)
- `cells`: List of cell labels where organ is located
- If `present=0`, then `cells=[]` (empty)
- If `present=1`, then `1 ≤ |cells| ≤ K`

## Metrics

### Primary Metrics
- **Presence Accuracy**: Binary detection accuracy
- **Cell@K**: Proportion where at least one predicted cell overlaps with ground truth
- **Gated Accuracy**: Combined presence + localization accuracy

### Secondary Metrics
- **Cell Precision**: |predicted ∩ ground_truth| / |predicted|
- **Cell Recall**: |predicted ∩ ground_truth| / |ground_truth|
- **Cell F1**: Harmonic mean of precision and recall
- **False Positive Cells**: Cells predicted when organ is absent

## Output Structure

Results are saved to:
```
results/cell_selection_G{grid}_K{topk}_YYYYMMDD_HHMMSS/
├── zero_shot/
│   └── {model}/
│       └── cholecseg8k_cell_selection/
│           └── sample_NNNN.json
├── fewshot_standard/
│   └── {model}/
│       └── cholecseg8k_cell_selection/
│           └── sample_NNNN.json
├── fewshot_hard_negatives/
│   └── {model}/
│       └── cholecseg8k_cell_selection/
│           └── sample_NNNN.json
├── metrics_comparison.txt    # Human-readable comparison table
└── metrics_comparison.json   # Machine-readable metrics
```

## Testing

Run the test suite to verify the implementation:
```bash
cd notebooks_py/
python3 test_cell_selection.py
```

This tests:
- Ground truth computation
- Metrics calculation
- JSON parsing
- Prompt generation
- Coordinate conversions

## Comparison with Pointing

| Aspect | Pointing (x,y) | Cell Selection |
|--------|---------------|----------------|
| Output | Exact coordinates | Cell label(s) |
| Precision | High (pixel-level) | Medium (cell-level) |
| Robustness | Lower (exact match needed) | Higher (cell overlap) |
| Complexity | Harder for VLMs | Easier for VLMs |
| Best for | Large, clear organs | Small or ambiguous organs |

## Implementation Notes

1. **Grid Alignment**: Cells are aligned to the canvas (224×224 for original size)
2. **Edge Cells**: Last row/column cells may be slightly larger to cover full canvas
3. **Ground Truth**: Computed from binary masks with configurable min_pixels threshold
4. **Dominant Cell**: The cell with the most organ pixels (for strict evaluation)
5. **Caching**: Reuses existing LLM response caching infrastructure

## Future Improvements

- [ ] Support for adaptive grid sizes based on organ size
- [ ] Multi-scale evaluation (coarse-to-fine)
- [ ] Confidence scores per cell
- [ ] Visualization of predictions overlaid on images
- [ ] Integration with existing enhanced evaluator metrics