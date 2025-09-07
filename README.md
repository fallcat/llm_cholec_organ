# llm_cholec_organ
Using Large Vision-Language Models for Fine-Grained Organ Detection in Laparoscopic Cholecystectomy

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

## All experiments

1. Prepare few-shot examples (alternative - old method)

```
python notebooks_py/prepare_fewshot_examples.py
```

2. Evaluate pointing

```
EVAL_NUM_SAMPLES=20 python3 eval_pointing_original_size.py
```