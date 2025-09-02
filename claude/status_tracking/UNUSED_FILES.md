# Unused Files Analysis

Based on the main experiment pipeline (`notebooks_py/eval_pointing_original_size.py`), here are files that are NOT actively used:

## 1. Legacy/Old Implementation Files (replaced by endopoint package)
- `src/cholec.py` - Old main evaluation script (replaced by endopoint package)
- `src/cholecseg8k_utils.py` - Old dataset utilities (replaced by endopoint/datasets/cholecseg8k.py)
- `src/llms.py` - Old LLM interface (replaced by endopoint/models/)
- `src/prompts/explanations.py` - Old prompt templates (replaced by endopoint/prompts/builders.py)
- `src/few_shot_selection.py` - Old few-shot selection (data already prepared in data_info/)

## 2. Alternative/Experimental Evaluation Scripts
These are variations or older versions of the main evaluation script:
- `notebooks_py/eval_pointing.py` - Older version
- `notebooks_py/eval_pointing_standalone.py` - Standalone version
- `notebooks_py/comprehensive_evaluation.py` - Alternative comprehensive version
- `notebooks_py/simple_evaluation.py` - Simplified version
- `notebooks_py/simple_eval_v2.py` - Another simplified version
- `notebooks/evaluate_models_cholec.py` - Notebook version
- `notebooks/evaluate_endopoint_models.py` - Alternative endopoint evaluation

## 3. Debug/Diagnostic Scripts
These were used during development but not needed for running experiments:
- `notebooks_py/debug_cache.py` - Cache debugging
- `notebooks_py/diagnose_fewshot.py` - Few-shot debugging
- `notebooks_py/test_cache_keys.py` - Cache key testing
- `notebooks_py/test_coordinate_system.py` - Coordinate system testing
- `notebooks_py/test_fewshot_loading.py` - Few-shot loading test
- `notebooks_py/test_pointing_imports.py` - Import testing
- `notebooks_py/trace_fewshot.py` - Few-shot tracing
- `notebooks_py/verify_fewshot.py` - Few-shot verification

## 4. Utility Scripts (not part of main pipeline)
- `notebooks_py/clear_cache.py` - Cache clearing utility
- `notebooks_py/analyze_results_simple.py` - Simple analysis (replaced by analyze_metrics_json.py)
- `notebooks_py/eval_pointing_analyze.py` - Analysis script
- `notebooks_py/reanalyze_pickle.py` - Pickle reanalysis

## 5. Test Files (used for testing, not production)
All files in `tests/` directory are for testing/validation:
- `tests/test_*.py` - Various test scripts
- These are useful for development but not part of the main experiment pipeline

## 6. VLM Test Scripts
- `notebooks/test_vlm_cholec.py` - VLM testing
- `notebooks/test_vlm_simple.py` - Simple VLM test
- `notebooks/test_all_models_cholec.py` - All models test
- `notebooks/test_endopoint_simple.py` - Simple endopoint test

## 7. Incomplete/Placeholder Modules
- `src/endopoint/data/__init__.py` - Empty/placeholder
- `src/endopoint/data/presence.py` - Not actively used (presence data pre-computed)
- `src/endopoint/datasets/endoscape.py` - EndoScape dataset (not yet implemented)
- `src/endopoint/geometry/` - Geometry modules (canvas.py, __init__.py) - not directly used
- `src/endopoint/utils/io.py` - I/O utilities (minimal usage)
- `src/endopoint/utils/logging.py` - Logging setup (could be used but isn't)
- `src/endopoint/eval/parser.py` - JSON parser (functionality integrated elsewhere)

## 8. CLI Tools (not used in current workflow)
- `cli/build_presence_cache.py` - CLI for building presence cache
- `cli/__init__.py` - CLI module init

## 9. Model-specific Files (replaced by integrated versions)
- `src/endopoint/models/anthropic_claude.py` - Replaced by anthropic.py
- `src/endopoint/models/google_gemini.py` - Replaced by google.py
- `src/endopoint/models/openai_gpt.py` - Replaced by openai.py

## Summary

### Keep (Core Pipeline):
- `notebooks_py/eval_pointing_original_size.py` - Main entry point
- `notebooks_py/prepare_fewshot_examples.py` - Few-shot preparation
- `src/endopoint/` package (most modules)
- `data_info/` - Pre-computed data
- `qwen_vl_utils.py` - Qwen helper
- `tests/analyze_metrics_json.py` - Analysis tool
- `tests/test_json_metrics.py` - Metrics testing

### Consider Removing:
1. **Legacy src/ files** (cholec.py, llms.py, etc.) - replaced by endopoint
2. **Alternative evaluation scripts** - keep only the main one
3. **Debug/diagnostic scripts** - unless needed for troubleshooting
4. **Unused endopoint modules** - geometry/, data/, some models/

### Archive/Document:
- Test files - useful for validation
- VLM test scripts - useful for model testing
- CLI tools - might be useful in future