# EndoPoint: Large Vision-Language Models for Fine-Grained Organ Detection

A modular Python package for organ detection in endoscopic videos using Large Vision-Language Models, with enhanced evaluation framework for comprehensive pointing metrics.

## Quick Start: Generate All Few-Shot Examples

Generate both separate (per-organ) and combined (multi-organ) few-shot examples for all datasets with a single command:

```bash
cd notebooks_py
./generate_few_shot_all.sh
```

This will:
- Process all 3 datasets (cholecseg8k_local, cholec_organs, cholec_gonogo)
- Generate **separate** few-shot examples (one per organ)
- Generate **combined** few-shot examples using greedy set cover (minimum 3 examples covering all organs)
- Save outputs to `data_info/{dataset_name}_balanced_200/`

### Key Innovation: Combined Mode
The combined mode uses a greedy set cover algorithm to select the minimum number of training images that cover all organs:
- **CholecSeg8k**: Only 3 images needed to cover all 12 organs (75% reduction vs per-organ approach)
- **Efficiency**: Reduces API calls and improves context understanding
- **Diversity**: Ensures minimum 3 examples for robust few-shot learning

### Advanced Options

```bash
# Force regenerate (clear cache)
./generate_few_shot_all.sh --force

# Use 5 minimum combined examples instead of 3
./generate_few_shot_all.sh --min-combined 5

# Generate only combined mode
./generate_few_shot_all.sh --mode combined

# Process specific dataset
./generate_few_shot_all.sh --datasets cholecseg8k_local
```

### Output Files
Each dataset gets:
- `fewshot_plan_bbox_combined_greedy.json` - Combined bounding box examples (NEW)
- `fewshot_plan_bbox_200.json` - Separate bbox examples per organ
- `fewshot_plan_pointing_200.json` - Separate pointing examples
- `presence_matrix_train.npy` - Binary organ presence matrix

For detailed documentation, see `claude/claude_code_commands/15-few-shot-generation-system.md`

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

## All experiments

1. Prepare few-shot examples (alternative - old method)

```
python notebooks_py/prepare_fewshot_examples.py
```

2. Evaluate pointing

```
EVAL_NUM_SAMPLES=20 python3 eval_pointing_original_size.py
```