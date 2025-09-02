# Experiment Pipeline

## Overview
This document describes the complete pipeline for running organ detection and pointing experiments on laparoscopic surgery images.

## Main Entry Point
`notebooks_py/eval_pointing_original_size.py`

## Pipeline Flow

### 1. Data Preparation (One-time setup)
```bash
# These files are pre-generated and stored in data_info/cholecseg8k/
```
- **Balanced indices**: `balanced_indices_train_100_cap70_seed7.json`
- **Few-shot plans**: 
  - `fewshot_plan_train_pos1_neg1_mp50_seed123_excl100.json` (standard)
  - `fewshot_plan_train_pos1_neg1_hneg1_mp50_seed123_excl100.json` (hard negatives)
- **Presence matrix**: `presence_matrix_train_8080.npz`

### 2. Configuration
- **API Keys**: `API_KEYS2.json` (OpenAI, Anthropic, Google)
- **Environment Variables**:
  - `EVAL_NUM_SAMPLES`: Number of samples to evaluate
  - `EVAL_MODELS`: Comma-separated list of models
  - `EVAL_USE_CACHE`: Whether to use cached responses
  - `EVAL_SKIP_ZERO_SHOT`: Skip zero-shot evaluation
  - `EVAL_SKIP_FEW_SHOT`: Skip few-shot evaluation
  - `EVAL_QUICK_TEST`: Quick test with 5 samples

### 3. Core Modules Used

#### From `src/endopoint/`:
- **datasets/**
  - `cholecseg8k.py`: Dataset loader and adapter
  - `base.py`: Dataset adapter protocol
  
- **eval/**
  - `enhanced_evaluator.py`: Main evaluation class
  - `evaluator.py`: Base evaluator with model loading
  - `pointing.py`: Pointing task implementation
  - `pointing_metrics.py`: Metrics calculation
  
- **models/**
  - `base.py`: Base model classes
  - `anthropic.py`: Claude models
  - `google.py`: Gemini models
  - `openai.py`: GPT models
  - `vllm.py`: Open VLMs (LLaVA, Qwen, Pixtral, DeepSeek)
  - `utils.py`: Model utilities
  
- **prompts/**
  - `builders.py`: Prompt construction
  - `registry.py`: Prompt registration
  
- **utils/**
  - `cache.py`: Response caching
  - `geometry.py`: Coordinate transformations
  - `io.py`: File I/O utilities
  - `logging.py`: Logging setup
  - `rng.py`: Random seed management

#### Supporting Files:
- `qwen_vl_utils.py`: Qwen2.5-VL vision processing helper

### 4. Execution Flow

1. **Load Configuration**
   - Read API keys from `API_KEYS2.json`
   - Parse environment variables
   - Set random seeds

2. **Initialize Evaluator**
   ```python
   evaluator = EnhancedPointingEvaluator(
       models=models_to_eval,
       output_dir=output_dir,
       use_cache=use_cache
   )
   ```

3. **Load Test Data**
   - Load balanced indices from `data_info/cholecseg8k/balanced_indices_train_100_cap70_seed7.json`
   - Load CholecSeg8k dataset via HuggingFace

4. **Load Few-shot Plans**
   - Standard: `fewshot_plan_train_pos1_neg1_mp50_seed123_excl100.json`
   - Hard negatives: `fewshot_plan_train_pos1_neg1_hneg1_mp50_seed123_excl100.json`

5. **Run Evaluations**
   - Zero-shot (if not skipped)
   - Few-shot standard
   - Few-shot with hard negatives

6. **Generate Results**
   - Per-sample JSON files: `results/pointing_original_*/[mode]/[model]/cholecseg8k_pointing/train_*.json`
   - Metrics comparison: `metrics_comparison.txt` and `metrics_comparison.json`
   - Summary statistics: `summary.csv`

### 5. Analysis Tools
- `tests/analyze_metrics_json.py`: Analyze JSON metrics
- `tests/test_json_metrics.py`: Test metrics functionality

## Running Experiments

### Basic Run (all models, all modes)
```bash
python3 notebooks_py/eval_pointing_original_size.py
```

### Quick Test
```bash
EVAL_QUICK_TEST=true python3 notebooks_py/eval_pointing_original_size.py
```

### Specific Model
```bash
EVAL_MODELS='gpt-5-mini' python3 notebooks_py/eval_pointing_original_size.py
```

### Skip Zero-shot
```bash
EVAL_SKIP_ZERO_SHOT=true python3 notebooks_py/eval_pointing_original_size.py
```

### Multiple VLMs
```bash
EVAL_MODELS='llava-hf/llava-v1.6-mistral-7b-hf,mistralai/Pixtral-12B-2409' python3 notebooks_py/eval_pointing_original_size.py
```

## Output Structure
```
results/
└── pointing_original_YYYYMMDD_HHMMSS/
    ├── zero_shot/
    │   └── [model]/
    │       └── cholecseg8k_pointing/
    │           ├── train_00000.json
    │           ├── train_00001.json
    │           └── metrics_summary_train.json
    ├── fewshot_standard/
    │   └── [model]/...
    ├── fewshot_hard_negatives/
    │   └── [model]/...
    ├── metrics_comparison.txt
    ├── metrics_comparison.json
    └── summary.csv
```