# Recap
This file contains recap for next steps.

- [x] Saved few-shot examples.
- [x] Written experiments for pointing.
- [x] Add open-source models such as Llava, Qwen2.5-vl
  - Implemented LLaVA 1.6 (LLaVA-NeXT) with vLLM support
  - Implemented Qwen-2.5-VL with vLLM support
  - Implemented Pixtral-12B with vLLM support
  - Implemented DeepSeek-VL2 with transformers support
  - Created test scripts and integrated into llms.py
- [ ] Improve pointing by changing how pointing works.
- [ ] Add experiments for getting presence result only.

Current problems:
1. For pointing experiments, few-shot improves checking if something is present, but doesn't improve for pointing at the specific organ. Need to have better way to eliciting response from LLM.

## Recent Updates (2025-01-09)

### VLM Model Implementation
- Created `src/endopoint/models/vllm.py` with 4 VLM model implementations:
  - LLaVA 1.6 (LLaVA-NeXT) with vLLM support
  - Qwen-2.5-VL with vLLM support  
  - Pixtral-12B with vLLM support
  - DeepSeek-VL2 with transformers
- Updated `src/endopoint/models/utils.py` with `to_pil_image` helper function
- Modified `src/llms.py` to load all models from endopoint package
- Created evaluation scripts using only endopoint modules:
  - `notebooks/evaluate_endopoint_models.py` - Full evaluation
  - `notebooks/test_endopoint_simple.py` - Simple testing
- **FIXED vLLM API calls** - Now uses official dictionary format:
  - Single image: `{"prompt": prompt, "multi_modal_data": {"image": image}}`
  - Based on official vLLM LLaVA example provided by user
- Fixed dataset split issue (CholecSeg8k only has 'train' split, not 'test')
- Fixed LLaVA-NeXT compatibility:
  - Use correct model classes (LlavaNextForConditionalGeneration vs LlavaForConditionalGeneration)
  - Proper conversation format for LLaVA-NeXT models
  - Handle multiprocessing 'spawn' requirement for vLLM with CUDA
- All models now integrate seamlessly with the endopoint architecture

### Testing Commands:
```bash
# Test LLaVA-NeXT directly
python test_llava_next.py

# Test with transformers (no vLLM)
python test_llava_simple.py

# Full evaluation
python notebooks/evaluate_endopoint_models.py --samples 3 --vlm-only

# Pointing evaluation with LLaVA
EVAL_MODELS='llava-hf/llava-v1.6-mistral-7b-hf' python3 notebooks_py/eval_pointing_original_size.py

# Skip zero-shot (few-shot only)
EVAL_SKIP_ZERO_SHOT=true python3 notebooks_py/eval_pointing_original_size.py
```

## Latest Updates (2025-01-09 Evening)

### LLaVA Integration into Pointing Evaluation
- Added LLaVA to `notebooks_py/eval_pointing_original_size.py` evaluation pipeline
- Fixed vLLM prompt formatting with proper `<image>` token placement
- Increased max_model_len from 4096 to 8192 for few-shot learning
- Added evaluation control flags:
  - `EVAL_SKIP_ZERO_SHOT`: Skip zero-shot evaluation
  - `EVAL_SKIP_FEW_SHOT`: Skip few-shot evaluation
- Fixed multi-image handling for few-shot examples
- Enabled verbose debugging for vLLM issues
- See `claude_code_commands/06-llava-integration.md` for full details

### JSON Metrics and VLM Fixes (2025-01-09 Late)
- Added JSON output for metrics comparison alongside text output
- Fixed DeepSeek-VL2 dtype mismatches (BFloat16 vs Half)
- Organized all test files into `tests/` directory
- Created analysis tools for JSON metrics
- See `claude_code_commands/07-json-metrics-and-fixes.md` for full details

## Latest Updates (2025-09-02)

### Codebase Cleanup and Archival
- Archived unused files based on analysis in `claude/status_tracking/UNUSED_FILES.md`
- Organized files into clear categories:
  - **Kept**: Core pipeline files (eval_pointing_original_size.py, endopoint package, data_info/)
  - **Archived**: Legacy implementations, alternative scripts, debug tools, unused modules
- Created organized archive structure in `archived/` directory:
  - `legacy_src/` - Old implementation files replaced by endopoint
  - `alternative_eval/` - Various evaluation script versions
  - `debug_scripts/` - Debug and diagnostic utilities
  - `vlm_tests/` - VLM testing scripts
  - `unused_endopoint/` - Deprecated endopoint modules
  - `cli_tools/` - Unused CLI utilities
- Benefits: Cleaner codebase, clear main pipeline, preserved history
- See `claude_code_commands/08-archive-unused-files.md` for full details
- See `ARCHIVE_SUMMARY.md` for restoration guide
