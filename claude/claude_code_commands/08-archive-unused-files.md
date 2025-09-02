# Archive Unused Files

Date: 2025-09-02

## Objective
Clean up the codebase by archiving unused files identified in `claude/status_tracking/UNUSED_FILES.md`, keeping only the essential components for the main experiment pipeline.

## Files Organization

### Core Files Kept (Active Pipeline)
- `notebooks_py/eval_pointing_original_size.py` - Main experiment entry point
- `notebooks_py/prepare_fewshot_examples.py` - Few-shot data preparation
- `src/endopoint/` - Core package (active modules only)
- `data_info/` - Pre-computed data and configurations
- `qwen_vl_utils.py` - Qwen model helper utilities
- `tests/analyze_metrics_json.py` - Results analysis tool
- `tests/test_json_metrics.py` - Metrics validation

### Archived Structure
```
archived/
├── legacy_src/          # Old implementation (replaced by endopoint)
│   ├── cholec.py
│   ├── cholecseg8k_utils.py
│   ├── llms.py
│   ├── few_shot_selection.py
│   └── explanations.py
├── alternative_eval/    # Alternative evaluation scripts
│   ├── eval_pointing.py
│   ├── eval_pointing_standalone.py
│   ├── comprehensive_evaluation.py
│   ├── simple_evaluation.py
│   ├── simple_eval_v2.py
│   ├── evaluate_models_cholec.py
│   └── evaluate_endopoint_models.py
├── debug_scripts/       # Debug and diagnostic tools
│   ├── debug_cache.py
│   ├── diagnose_fewshot.py
│   ├── test_cache_keys.py
│   ├── test_coordinate_system.py
│   ├── test_fewshot_loading.py
│   ├── test_pointing_imports.py
│   ├── trace_fewshot.py
│   ├── verify_fewshot.py
│   ├── clear_cache.py
│   ├── analyze_results_simple.py
│   ├── eval_pointing_analyze.py
│   └── reanalyze_pickle.py
├── vlm_tests/          # VLM testing scripts
│   ├── test_vlm_cholec.py
│   ├── test_vlm_simple.py
│   ├── test_all_models_cholec.py
│   └── test_endopoint_simple.py
├── unused_endopoint/   # Deprecated endopoint modules
│   ├── anthropic_claude.py
│   ├── google_gemini.py
│   ├── openai_gpt.py
│   └── endoscape.py
└── cli_tools/          # Unused CLI utilities
    └── cli/
```

## Commands Executed

```bash
# Create archive directories
mkdir -p archived/legacy_src archived/alternative_eval archived/debug_scripts \
         archived/vlm_tests archived/unused_endopoint archived/cli_tools

# Move legacy src files
mv src/cholec.py src/cholecseg8k_utils.py src/llms.py \
   src/few_shot_selection.py archived/legacy_src/
mv src/prompts/explanations.py archived/legacy_src/

# Move alternative evaluation scripts
mv notebooks_py/eval_pointing.py notebooks_py/eval_pointing_standalone.py \
   notebooks_py/comprehensive_evaluation.py notebooks_py/simple_evaluation.py \
   notebooks_py/simple_eval_v2.py archived/alternative_eval/
mv notebooks/evaluate_models_cholec.py notebooks/evaluate_endopoint_models.py \
   archived/alternative_eval/

# Move debug/diagnostic scripts
mv notebooks_py/debug_cache.py notebooks_py/diagnose_fewshot.py \
   notebooks_py/test_cache_keys.py notebooks_py/test_coordinate_system.py \
   notebooks_py/test_fewshot_loading.py notebooks_py/test_pointing_imports.py \
   notebooks_py/trace_fewshot.py notebooks_py/verify_fewshot.py \
   archived/debug_scripts/
mv notebooks_py/clear_cache.py notebooks_py/analyze_results_simple.py \
   notebooks_py/eval_pointing_analyze.py notebooks_py/reanalyze_pickle.py \
   archived/debug_scripts/

# Move VLM test scripts
mv notebooks/test_vlm_cholec.py notebooks/test_vlm_simple.py \
   notebooks/test_all_models_cholec.py notebooks/test_endopoint_simple.py \
   archived/vlm_tests/

# Move unused endopoint modules
mv src/endopoint/models/anthropic_claude.py \
   src/endopoint/models/google_gemini.py \
   src/endopoint/models/openai_gpt.py archived/unused_endopoint/
mv src/endopoint/datasets/endoscape.py archived/unused_endopoint/

# Archive CLI tools
mv cli/ archived/cli_tools/
```

## Results

### Benefits
1. **Cleaner codebase** - Only active components remain visible
2. **Clear pipeline** - Main experiment flow is now obvious
3. **Preserved history** - All code archived for reference
4. **Reduced confusion** - No multiple versions of evaluation scripts

### Main Entry Points After Cleanup
1. **Run experiments**: `python notebooks_py/eval_pointing_original_size.py`
2. **Prepare data**: `python notebooks_py/prepare_fewshot_examples.py`
3. **Analyze results**: `python tests/analyze_metrics_json.py`

### Restoration
All archived files are preserved and can be restored from `archived/` if needed.

## Related Files
- `claude/status_tracking/UNUSED_FILES.md` - Original analysis
- `ARCHIVE_SUMMARY.md` - Detailed archive documentation