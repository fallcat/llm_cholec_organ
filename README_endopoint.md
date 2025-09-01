# EndoPoint: Large Vision-Language Models for Fine-Grained Organ Detection

A modular Python package for organ detection in endoscopic videos using Large Vision-Language Models, with enhanced evaluation framework for comprehensive pointing metrics.

## Installation

### Required Dependencies

```bash
# Core dependencies
pip install numpy torch pandas tqdm datasets pillow

# Install the package
pip install -e .
```

### API Keys Setup

Create `API_KEYS2.json` in the root directory:
```json
{
    "OPENAI_API_KEY": "your-openai-key",
    "ANTHROPIC_API_KEY": "your-anthropic-key", 
    "GOOGLE_API_KEY": "your-google-key"
}
```

Or set environment variables:
```bash
export OPENAI_API_KEY=your-key
export ANTHROPIC_API_KEY=your-key
export GOOGLE_API_KEY=your-key
```

## Quick Start

### 1. Prepare Dataset

```bash
# Build presence cache for CholecSeg8k
endopoint-build-cache --dataset cholecseg8k --split train
```

### 2. Run Enhanced Pointing Evaluation

```bash
cd notebooks_py

# Quick test with 5 samples
EVAL_QUICK_TEST=true python3 eval_pointing.py

# Full evaluation
python3 eval_pointing.py
```

## Enhanced Pointing Evaluation

The enhanced evaluation system provides comprehensive metrics beyond simple accuracy, including hit detection and gated metrics.

### Running Enhanced Evaluation

```bash
cd notebooks_py

# Environment variables for configuration
EVAL_NUM_SAMPLES=10              # Number of samples (default: all)
EVAL_MODELS='gpt-4o-mini'       # Comma-separated models
EVAL_USE_CACHE=true              # Use cached responses (default: true)
EVAL_USE_ENHANCED=true           # Use enhanced metrics (default: true)
EVAL_QUICK_TEST=true             # Quick 5-sample test

# Examples
python3 eval_pointing.py                                    # Full evaluation
EVAL_NUM_SAMPLES=20 python3 eval_pointing.py               # 20 samples
EVAL_QUICK_TEST=true python3 eval_pointing.py              # Quick test
EVAL_USE_CACHE=false python3 eval_pointing.py              # No cache
```

### Comprehensive Metrics

The enhanced evaluator calculates:

#### Per-Organ Metrics
- **Confusion Matrix**: TP, FN, TN, FP for each organ
- **Presence Accuracy**: (TP + TN) / Total - How well the model detects organ presence
- **Hit@Point|Present**: Percentage of correct localizations when organ is detected
- **Gated Metrics**: Combined detection and pointing accuracy
- **F1 Score**: Harmonic mean of precision and recall

#### Example Output

```
Model: gpt-4o-mini | Prompt: zero_shot | Split: train | Examples used: 10
ID  Label                     TP   FN   TN   FP   Pres  Abs   Tot   PresenceAcc   Hit@Pt|Pres   gTP  gFN  gTN  gFP   GatedAcc
 1  Abdominal Wall              4    4    1    1      8    2    10    50.00%      0.00%     0    8    2    0    20.00%
 2  Liver                      10    0    0    0     10    0    10   100.00%     40.00%     4    6    0    0    40.00%
 3  Gastrointestinal Tract      2    4    1    3      6    4    10    30.00%      0.00%     0    6    4    0    40.00%
...

Totals across organs:
TP=52  FN=10  TN=17  FP=41  Present=62  Absent=58  Total=120
Macro PresenceAcc= 57.50%   Macro Hit@Point|Present= 14.94%   Macro GatedAcc= 55.00%   Macro F1= 62.45%
```

### Output Directory Structure

```
results/pointing_YYYYMMDD_HHMMSS/
├── zero_shot/
│   ├── gpt-4o-mini/
│   │   └── cholecseg8k_pointing/
│   │       ├── train_00000.json    # Per-sample results
│   │       ├── train_00001.json
│   │       └── metrics_summary_train.json
│   └── claude-3-5-sonnet-20241022/
│       └── ...
├── fewshot_standard/
│   └── ...
├── fewshot_hard_negatives/
│   └── ...
├── raw_results.pkl                 # Complete results
├── summary.csv                     # Summary statistics
└── metrics_comparison.txt          # Full comparison
```

### Analyzing Results

```bash
# Analyze latest results
python3 notebooks_py/eval_pointing_analyze.py --latest

# Analyze specific directory
python3 notebooks_py/eval_pointing_analyze.py results/pointing_20250901_041511

# In Python/Jupyter
import json
with open('results/pointing_*/zero_shot/*/cholecseg8k_pointing/train_00000.json') as f:
    result = json.load(f)
print(f"Ground Truth: {result['y_true']}")
print(f"Predictions: {result['y_pred']}")
print(f"Hit Detection: {result['hits']}")
```

## Architecture

The package is organized into modular components:

- **datasets/**: Dataset adapters with a common protocol
- **models/**: LLM model adapters (OpenAI, Anthropic, Google)
- **prompts/**: Prompt builders and registry
- **geometry/**: Canvas and coordinate transformations
- **data/**: Data processing utilities (presence cache)
- **eval/**: Evaluation runners and parsers
- **cli/**: Command-line tools

## Adding a New Dataset

1. Create a new adapter in `src/endopoint/datasets/`:

```python
from .base import register_dataset

class MyDatasetAdapter:
    # Implement the DatasetAdapter protocol
    ...

@register_dataset("mydataset")
def build_mydataset(**cfg):
    return MyDatasetAdapter(**cfg)
```

2. Create a config file in `configs/`:

```yaml
name: mydataset
root: /path/to/dataset
recommended_canvas: [768, 768]
```

3. Use with any CLI tool:

```bash
endopoint-build-cache --dataset mydataset
```

## Development Status

This is a refactored version of the research codebase. Core modules implemented:

- ✅ Dataset adapter protocol and CholecSeg8k implementation
- ✅ Model adapters for OpenAI, Anthropic, Google
- ✅ Prompt management system
- ✅ Geometry utilities
- ✅ Presence cache
- ✅ JSON parser with fallbacks
- ✅ Enhanced pointing evaluator with comprehensive metrics
- ✅ Hit detection for pointing accuracy
- ⏳ Selection algorithms
- ⏳ Few-shot learning improvements
- ⏳ Visualization tools
- ⏳ Full CLI suite

See `claude_code_commands/01-refactor-code.md` for the complete refactoring plan.

## Key Features of Enhanced Evaluation

### Differences from Standard Evaluation

| Feature | Standard | Enhanced |
|---------|----------|----------|
| Hit Detection | ❌ | ✅ Validates points within organ masks |
| Gated Metrics | ❌ | ✅ Combined detection + pointing accuracy |
| Per-Sample JSON | ❌ | ✅ Detailed results for each image |
| Comprehensive Table | Basic | Full confusion matrix + all metrics |
| Notebook Format | ❌ | ✅ Matches research notebook output |

### Metrics Explained

1. **Presence Accuracy**: How well the model detects if an organ is present
2. **Hit@Point|Present**: Among organs correctly detected as present, how often does the predicted point actually fall within the organ boundary?
3. **Gated Accuracy**: Stricter metric where a prediction is only correct if both detection and pointing are accurate
4. **Macro vs Micro**: Macro averages across organs equally, Micro weights by total predictions

## Troubleshooting

### Missing Dependencies

```bash
# If you see ModuleNotFoundError
pip install numpy torch pandas tqdm datasets pillow
```

### Using Jupyter Environment

If packages are installed in Jupyter but not terminal:

```python
# In Jupyter notebook
import sys
sys.path.append('../src')
%run ../notebooks_py/eval_pointing.py
```

### Cache Issues

```bash
# Force fresh API calls
EVAL_USE_CACHE=false python3 eval_pointing.py
```

### Memory Issues

```bash
# Evaluate in smaller batches
EVAL_NUM_SAMPLES=50 python3 eval_pointing.py
```