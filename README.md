# llm_cholec_organ
Using Large Vision-Language Models for Fine-Grained Organ Detection in Laparoscopic Cholecystectomy

## Quick Start: Generate Balanced Test Set and Few-Shot Examples

The primary notebook for data preparation is:

```
notebooks/test_unified_fewshot_simplified.ipynb
```

This notebook will:
- Generate 200 balanced test samples with 30% minimum quota for rare classes
- Create few-shot example plans for both pointing and bounding box tasks
- Save all outputs to `data_info/{dataset_name}_balanced_200/`

### Key Features:
- **Balanced Selection**: Boosts rare classes (Blood, Cystic Duct, Hepatic Vein, Liver Ligament) to ≥30% prevalence
- **Video-level Splits**: Prevents frame-level leakage with max 2 frames per video
- **No Data Contamination**: Few-shot examples strictly exclude the 200 test samples
- **Cached Results**: Automatic caching for reproducibility

To regenerate with new parameters, run cells with `force_regenerate=True`.

## All experiments

1. Prepare few-shot examples (alternative Python script)

```
python notebooks_py/prepare_fewshot_examples.py
```

2. Evaluate pointing

```
EVAL_NUM_SAMPLES=20 python3 eval_pointing_original_size.py
```