# Paper & Code Checklist

**Main Entry Point**: `notebooks_py/eval_pointing_original_size.py`

## Current Status Summary
- ✅ **Core infrastructure**: Modular `endopoint` package with dataset adapters, evaluation pipeline, prompt builders
- ✅ **Data preparation**: Balanced subsets, few-shot plans with hard negatives, visualizations
- ✅ **Models integrated**: GPT-4.1, Claude-Sonnet-4, Gemini-2.0, + 3 open VLMs (LLaVA, Qwen2.5-VL, Pixtral)
- ✅ **Evaluation modes**: Zero-shot, few-shot, few-shot with hard negatives
- ✅ **Metrics**: Comprehensive metrics with JSON output for analysis
- ✅ **Visualization**: Complete LaTeX table and PDF figure generation pipeline
- ✅ **Documentation**: Comprehensive guidelines for formatting and visualization
- ⏳ **TODO**: Config system, environment file, SSG-VQA dataset, reasoning prompts

## Prep & repo hygiene
- [ ] **Pin env**: Python + torch/transformers/VLM deps; write `environment.yml` or `requirements.txt`.
- [ ] **Config system**: single `configs/*.yaml` with keys  
  `{dataset, split, indices_file, canvas_mode, min_pixels, n_select, seed, prompt_name, fewshot_plan, models[]}`.
- [x] **Folder layout**
  - `src/endopoint/datasets/cholecseg8k.py` (loader, `ID2LABEL`, `LABEL_IDS`, adapter with `example_to_tensors`)
  - `src/endopoint/eval/` (pointing runner, enhanced_evaluator)
  - `src/endopoint/prompts/` (prompt builders)
  - `results/pointing_original_<timestamp>/<prompt>/<model>/...`
  - `data_info/cholecseg8k/` (balanced indices, fewshot plans)
  - `cache/` (for LLM response caching)
- [x] **Determinism**: set global seeds in `src/endopoint/utils/rng.py`.
- [x] **Canvas mode**: using `original` size (224x224 for CholecSeg8k) as default in `notebooks_py/eval_pointing_original_size.py`.

## Data selection (CholecSeg8k – main ablations)
- [x] **Presence matrix cache**: saved to `data_info/cholecseg8k/presence_matrix_train_8080.npz`.
- [x] **Balanced subset**: saved to  
  `data_info/cholecseg8k/balanced_indices_train_100_cap70_seed7.json` and other variants.
- [x] **Audit plots**: saved as `fewshot_statistics.png`, `fewshot_examples_visualization.png`, `nearmiss_visualization.png`.

## Few-shot exemplars
- [x] **Plan (no overlap with eval)**: saved as  
  `fewshot_plan_train_pos1_neg1_mp50_seed123_excl100.json` (standard)  
  `fewshot_plan_train_pos1_neg1_hneg1_mp50_seed123_excl100.json` (with hard negatives).
- [x] **Positive point policy**: using **centroid** policy; recorded as `point_original`.
- [x] **Hard negatives**: implemented with  
  - **near-miss points** inside adjacent class masks  
  - saved in plans with `hneg` or `nearmiss` in filename.
- [x] **Visualization**: exported as `fewshot_examples_visualization.png` and `nearmiss_visualization.png`.

## Core experiments to run (CholecSeg8k)
- [x] **Tasks**
  - [ ] Presence-only (baseline).
  - [x] Presence+Pointing (main) - implemented in `notebooks_py/eval_pointing_original_size.py`.
  - [x] **Presence+Cell Selection** (alternative localization) - implemented in `notebooks_py/eval_cell_selection_original_size.py`:
    - [x] Grid G=3 (3×3 grid, 9 cells total)
    - [x] Grid G=4 (4×4 grid, 16 cells total)  
    - [x] Top-K=1 (single cell prediction)
    - [x] Top-K=3 (up to 3 cells for larger organs)
  - [ ] **Presence+Bounding Box** (new localization method):
    - [ ] Convert `notebooks_py/eval_pointing_original_size.py` to generate bounding boxes
    - [ ] Add baselines that generate bounding boxes
    - [ ] Evaluate bounding boxes using IoU (Intersection over Union) metric
- [x] **Shot regimes**
  - [x] Zero-shot.
  - [x] Few-shot (1 pos / 1 neg) using saved plan.
  - [x] Few-shot + **hard negatives** (using `fewshot_plan_train_pos1_neg1_hneg1_mp50_seed123_excl100.json`).
- [x] **Prompting**
  - [x] **Vanilla JSON** (strict schema) - implemented in prompt builders.
  - [ ] **One reasoning variant** (CoT *or* Socratic, not both).
- [x] **Models** (expanded set)
  - [x] GPT-4o-mini (OpenAI)
  - [x] Claude-3.5-Sonnet-20241022 (Anthropic)
  - [x] Gemini-2.0-Flash-Exp (Google)
  - [x] **Open VLMs**: 
    - LLaVA-v1.6-Mistral-7B (via vLLM)
    - Qwen2.5-VL-7B-Instruct (via transformers)
    - Pixtral-12B-2409 (via vLLM)
    - DeepSeek-VL2 (via transformers)
- [x] **Canvas**
  - [x] `original` size (224x224) as primary in `eval_pointing_original_size.py`.

## Confirmatory experiments (SSG-VQA)
- [ ] **Dataset integration**: Add SSG-VQA dataset adapter with spatial relation questions
  - [ ] Dataset class with the same API: `example_to_tensors`, `ID2LABEL`, `LABEL_IDS`
  - [ ] Handle VQA format: questions about spatial relations between organs
  - [ ] Map question types to evaluation metrics
- [ ] **Evaluation setup**:
  - [ ] Adapt prompts for VQA-style spatial reasoning tasks
  - [ ] Recompute **presence cache** and **balanced selection** if applicable
  - [ ] Run **only**: best config (few-shot + hard-neg + chosen reasoning) **and** zero-shot baseline
- [ ] **Spatial relation focus**:
  - [ ] Test model's understanding of: above/below, left/right, near/far relations
  - [ ] Compare performance on spatial vs. simple presence questions
- [ ] Optional: replicate **one** small ablation (e.g., shots) on a smaller subset

## Cell Selection Implementation ✅
- [x] **Prompt builders** (`src/endopoint/prompts/builders.py`):
  - [x] `build_cell_selection_system_prompt(canvas_w, canvas_h, grid_size, top_k)`
  - [x] `build_cell_selection_user_prompt(organ_name)` 
  - [x] Cell labeling: A1-C3 (G=3) or A1-D4 (G=4) format
- [x] **Ground truth computation** (`src/endopoint/eval/cell_selection.py`):
  - [x] `compute_cell_ground_truth(mask, grid_size, min_pixels)` → cell set S_k
  - [x] `compute_dominant_cell(mask, grid_size)` → single cell d_k  
  - [x] Handle canvas coordinates (respecting letterbox if used)
- [x] **Parser updates** (`src/endopoint/eval/parser.py`):
  - [x] `parse_cell_selection_json()` with validation
  - [x] Enforce: present=0 → cells=[], present=1 → 1≤|cells|≤K
  - [x] Cell label validation (e.g., A1-C3 for G=3)
- [x] **Metrics** (`src/endopoint/eval/cell_selection.py`):
  - [x] Cell@1 accuracy (hit if pred_cells ∩ S_k ≠ ∅)
  - [x] Cell@K accuracy for K>1
  - [x] Cell Precision/Recall/F1 
  - [x] Penalize non-empty cells when GT absent
- [x] **Evaluation script** (`notebooks_py/eval_cell_selection_original_size.py`):
  - [x] Based on `eval_pointing_original_size.py` structure
  - [x] Config for G∈{3,4}, K∈{1,3}
  - [x] Run on same balanced subset as pointing experiments
  - [x] **Fixed cache key collision** between zero-shot and few-shot
  - [x] **Added persistent directory support** with `EVAL_PERSISTENT_DIR=true`
- [x] **Batch evaluation scripts**:
  - [x] `eval_both_persistent.sh` - Simple script for both tasks
  - [x] `eval_both_advanced.sh` - Advanced script with CLI options
- [ ] There might be some bugs with few-shot outputting the same thing as zero-shot. Need to check.

## Evaluation & logging
- [x] **Per-image JSON** cache with:  
  `{sample_idx,y_true,y_pred,hits,rows,prompt}` saved to `results/pointing_original_*/[prompt]/[model]/cholecseg8k_pointing/train_*.json`.
- [x] **Metrics** per organ and macro: PresenceAcc, Hit@Point|Present, **GatedAcc**, Precision, Recall, F1.
- [x] **Cell Selection metrics**: Cell@1, Cell@3, Cell Precision/Recall, stored in `results/cell_selection_*/`.
- [x] **Tables**: `print_metrics_table` produces paper-ready text tables; stored as `metrics_comparison.txt`.
- [x] **JSON metrics**: saved as `metrics_comparison.json` for programmatic analysis.
- [x] **Metrics regeneration script**: `notebooks_py/regenerate_metrics_comparison.py`
  - [x] Handles LLaVA coordinate normalization issue with `--fix-llava` flag
  - [x] Regenerates metrics from saved results
- [ ] **Curves (optional)**: presence ROC/AUPRC per organ for presence-only.
- [ ] **Failure mining**: thumbnails for FP (pred=1, miss-hit), FN (missed presence), and off-organ clicks.

## Figures for the paper
- [x] **Visualization pipeline**: Complete LaTeX table and PDF figure generation system
  - [x] LaTeX tables with bold/italic formatting for best/second-best values
  - [x] Bar charts with black stars marking highest performance
  - [x] Consistent model ordering (APIs first, then open-source)
  - [x] Proper two-column paper formatting (table*, figure*, [t] placement)
- [x] **Figure generation script**: `src/endopoint/vis/generate_paper_figures.py`
  - [x] Generates comprehensive metrics table with column grouping
  - [x] Creates individual model performance charts
  - [x] Produces metric comparison charts across models
  - [x] Generates 2x2 grid overall comparison
- [x] **Documentation**: Created comprehensive guidelines in `/guidelines/` folder
  - [x] `latex_formatting.md` - LaTeX best practices
  - [x] `figure_generation.md` - Matplotlib configuration
  - [x] `package_requirements.md` - Dependencies
  - [x] `visualization_best_practices.md` - Design principles
- [ ] **Balanced selection**: stacked present/absent bars (pool vs selected).
- [x] **Per-class pointing**: bar of Hit@Point|Present by model (implemented in bar_charts.py).
- [ ] **Qualitative**: 6–8 examples with predicted point (✓/✗), GT mask outline, per-organ rows.

## Ablation matrix (scope control)
- [ ] On **CholecSeg8k** only: shots (0 / few / few+hard-neg), vanilla vs 1 reasoning prompt, canvas sensitivity (original vs 768) *once*.
- [ ] **Cell Selection vs Pointing vs Bounding Box comparison**:
  - [ ] Same models, same test set
  - [ ] Compare Hit@Point|Present vs Cell@1/Cell@3 vs IoU@0.5/IoU@0.75
  - [ ] Analyze robustness: which approach handles small organs better?
  - [ ] Table showing precision/robustness/computational trade-offs
  - [ ] **Bounding Box specific analysis**:
    - [ ] IoU distribution per organ class
    - [ ] Failure modes: over-prediction vs under-prediction
    - [ ] Correlation between organ size and IoU performance
- [ ] On **EndoScape**: best config + zero-shot baseline (and at most one extra small ablation).

## Reproducibility
- [ ] Save **config JSON** alongside each run directory.
- [x] Store **few-shot plan** and **balanced indices** files in `data_info/cholecseg8k/`.
- [x] Record model identifiers in code (see DEFAULT_MODELS in `eval_pointing_original_size.py`).
- [x] Pin random seeds via `src/endopoint/utils/rng.py`.
- [ ] Include commit hash in `agg.json`.

## Sanity checks (tick before big runs)
- [x] Few-shot exemplars **do not overlap** eval indices (plans use `excl100` suffix).
- [x] Positive few-shot points land **inside** the organ mask (verified in visualizations).
- [x] Hard-neg points are **outside** the organ but **near/visually similar** (see `nearmiss_visualization.png`).
- [x] For classes always present (e.g., Liver=100%), handled in balanced selection.
- [x] Presence vector `min_pixels` = 50 in few-shot plans.

## Packaging for the paper
- [ ] **Main table**: best config on both datasets + zero-shot baseline.
- [ ] **Ablation table**: CholecSeg8k only (compact).
- [ ] **Localization comparison table**: Pointing vs Cell Selection vs Bounding Box
  - [ ] Metrics: Hit@Point, Cell@K, IoU@thresholds
  - [ ] Computational cost analysis
  - [ ] Qualitative comparison of outputs
- [ ] **Appendix**: prompt strings, few-shot policy, selection algorithm box, dataset balance plots.
- [ ] **Release artifacts**: indices, plans, and scripts to reproduce evaluation.

## Bounding Box Implementation (TODO)
- [ ] **Script conversion** (`notebooks_py/eval_bbox_original_size.py`):
  - [ ] Adapt from `eval_pointing_original_size.py` 
  - [ ] Modify prompts to request bounding box coordinates [x1, y1, x2, y2]
  - [ ] Update JSON response format: `{"name": "Liver", "present": 1, "bbox": [x1, y1, x2, y2]}`
  
- [ ] **Baseline methods**:
  - [ ] Point-to-box conversion: Convert point predictions to fixed-size boxes
  - [ ] Cell-to-box conversion: Convert cell selection to bounding boxes
  - [ ] MedSAM integration: Use MedSAM to generate region proposals as baseline
  
- [ ] **Ground truth computation**:
  - [ ] Extract tight bounding box from binary masks
  - [ ] Handle multiple connected components (use largest or union)
  - [ ] Store GT boxes in evaluation cache
  
- [ ] **IoU metrics**:
  - [ ] IoU@0.5 (standard detection threshold)
  - [ ] IoU@0.75 (strict threshold)
  - [ ] Mean IoU across all predictions
  - [ ] IoU histogram for error analysis
  
- [ ] **Comparison framework**:
  - [ ] Compare bbox vs pointing vs cell selection
  - [ ] Analyze trade-offs: precision vs computational cost
  - [ ] Visualize predictions with GT overlays

## Cell Selection Implementation Notes
### Integration with existing code:
1. **Reuse existing infrastructure**: 
   - Use same `ModelAdapter` classes for API calls
   - Leverage existing caching system in `src/endopoint/models/base.py`
   - Use same balanced indices and few-shot plans
   
2. **Minimal changes needed**:
   - New prompt builders in `src/endopoint/prompts/builders.py`
   - New module `src/endopoint/eval/cell_selection.py` for GT computation
   - Update parser in `src/endopoint/eval/parser.py` 
   - New evaluation script based on `eval_pointing_original_size.py`

3. **Grid coordinate mapping** (for 224×224 canvas):
   - G=3: cell_width=74, cell_height=74 (last col/row slightly larger)
   - G=4: cell_width=56, cell_height=56
   - Cell (r,c) → label: chr(65+r) + str(c+1) (e.g., A1, B2, C3)

4. **JSON response format**:
   ```json
   {"name": "Liver", "present": 1, "cells": ["B2", "B3"]}
   ```

5. **Metrics computation**:
   - Reuse confusion matrix logic from `enhanced_evaluator.py`
   - Add Cell@K as parallel to Hit@Point|Present
   - Track cell-level TP/FP/FN for precision/recall
