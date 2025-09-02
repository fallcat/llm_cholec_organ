# Archive Summary

Date: 2025-09-02

## Files Reorganization Summary

Based on the analysis in `claude/status_tracking/UNUSED_FILES.md`, the following files have been archived to clean up the codebase and keep only the essential components for the main experiment pipeline.

### Core Files Kept (Main Pipeline)
- `notebooks_py/eval_pointing_original_size.py` - Main experiment entry point
- `notebooks_py/prepare_fewshot_examples.py` - Few-shot data preparation
- `src/endopoint/` - Core package (except archived modules)
- `data_info/` - Pre-computed data and configurations
- `qwen_vl_utils.py` - Qwen model helper utilities
- `tests/analyze_metrics_json.py` - Results analysis tool
- `tests/test_json_metrics.py` - Metrics validation

### Archived Files

#### 1. Legacy Implementation (`archived/legacy_src/`)
- `src/cholec.py` - Old main evaluation script
- `src/cholecseg8k_utils.py` - Old dataset utilities
- `src/llms.py` - Old LLM interface
- `src/few_shot_selection.py` - Old few-shot selection
- `src/prompts/explanations.py` - Old prompt templates

#### 2. Alternative Evaluation Scripts (`archived/alternative_eval/`)
- `notebooks_py/eval_pointing.py`
- `notebooks_py/eval_pointing_standalone.py`
- `notebooks_py/comprehensive_evaluation.py`
- `notebooks_py/simple_evaluation.py`
- `notebooks_py/simple_eval_v2.py`
- `notebooks/evaluate_models_cholec.py`
- `notebooks/evaluate_endopoint_models.py`

#### 3. Debug/Diagnostic Scripts (`archived/debug_scripts/`)
- `notebooks_py/debug_cache.py`
- `notebooks_py/diagnose_fewshot.py`
- `notebooks_py/test_cache_keys.py`
- `notebooks_py/test_coordinate_system.py`
- `notebooks_py/test_fewshot_loading.py`
- `notebooks_py/test_pointing_imports.py`
- `notebooks_py/trace_fewshot.py`
- `notebooks_py/verify_fewshot.py`
- `notebooks_py/clear_cache.py`
- `notebooks_py/analyze_results_simple.py`
- `notebooks_py/eval_pointing_analyze.py`
- `notebooks_py/reanalyze_pickle.py`

#### 4. VLM Test Scripts (`archived/vlm_tests/`)
- `notebooks/test_vlm_cholec.py`
- `notebooks/test_vlm_simple.py`
- `notebooks/test_all_models_cholec.py`
- `notebooks/test_endopoint_simple.py`

#### 5. Unused Endopoint Modules (`archived/unused_endopoint/`)
- `src/endopoint/models/anthropic_claude.py` - Replaced by anthropic.py
- `src/endopoint/models/google_gemini.py` - Replaced by google.py
- `src/endopoint/models/openai_gpt.py` - Replaced by openai.py
- `src/endopoint/datasets/endoscape.py` - Not yet implemented

#### 6. CLI Tools (`archived/cli_tools/`)
- `cli/` directory - Not used in current workflow

## Restoration

If you need to restore any archived files, they are preserved in the `archived/` directory with the following structure:
```
archived/
├── legacy_src/       # Old implementation files
├── alternative_eval/ # Alternative evaluation scripts
├── debug_scripts/    # Debug and diagnostic tools
├── vlm_tests/       # VLM testing scripts
├── unused_endopoint/ # Unused endopoint modules
└── cli_tools/       # CLI utilities
```

## Next Steps

The codebase is now streamlined to focus on the core experiment pipeline. The main entry points are:
1. `notebooks_py/eval_pointing_original_size.py` - Run main experiments
2. `notebooks_py/prepare_fewshot_examples.py` - Prepare few-shot data
3. `tests/analyze_metrics_json.py` - Analyze results

All legacy and experimental code has been preserved in the archive for reference.