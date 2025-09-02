# Project Recap

## Project Overview
This is a medical computer vision research project that uses Large Vision-Language Models (LLMs) for fine-grained organ detection in laparoscopic cholecystectomy procedures. The project evaluates various LLMs (GPT, Claude, Gemini) on organ presence detection and localization tasks using the CholecSeg8k dataset.

## Current Status (as of 2025-09-02)

### Recently Completed:
- **Cell Selection Implementation**: Added grid-based organ localization as alternative to exact (x,y) pointing
  - Supports 3×3 and 4×4 grids with top-1 or top-3 cell selection
  - Comprehensive metrics: Cell@K, precision, recall, F1, gated accuracy
  - Full evaluation pipeline with zero-shot and few-shot configurations
  - See `02-cell-selection-implementation.md` for details
  
- **Cache Fix**: Resolved cache key collision issue between zero-shot and few-shot configurations
  - Fixed by computing actual ground truth for few-shot examples
  - Ensured examples are included in prompt tuple for proper cache key generation

- **Persistent Directory Feature**: Added evaluation-level persistence
  - `EVAL_PERSISTENT_DIR=true` uses fixed directories without timestamps
  - Allows resuming interrupted evaluations and skipping completed samples
  - Independent from API-level caching

### Previous Work:
- **Refactoring to `endopoint` package**: Partially complete
  - ✅ Package scaffold with pyproject.toml
  - ✅ Dataset adapter protocol and CholecSeg8k implementation
  - ✅ Utility modules (io, logging, rng)
  - ✅ Geometry/canvas transformations
  - ✅ Prompt builders and registry
  - ✅ Model adapters (OpenAI, Anthropic, Google)
  - ✅ JSON parser with fallbacks
  - ✅ Basic CLI structure
  - 🔄 Evaluation runner (in progress)
  - 🔄 Complete CLI suite (in progress)

## Key Files and Locations

### Evaluation Scripts:
- `notebooks_py/eval_pointing_original_size.py` - Pointing evaluation
- `notebooks_py/eval_cell_selection_original_size.py` - Cell selection evaluation
- `notebooks_py/eval_both_persistent.sh` - Run both evaluations
- `notebooks_py/eval_both_advanced.sh` - Advanced script with options

### Core Modules:
- `src/endopoint/eval/cell_selection.py` - Cell selection computation
- `src/endopoint/eval/pointing.py` - Pointing and cell selection runners
- `src/endopoint/prompts/builders.py` - Prompt templates
- `src/endopoint/models/` - Model adapters for different APIs

### Results:
- `results/pointing_original/` - Persistent pointing results
- `results/cell_selection_G3_K1/` - Persistent cell selection results (3×3, top-1)

## Quick Start Commands

```bash
# Change to notebooks directory
cd /shared_data0/weiqiuy/llm_cholec_organ/notebooks_py

# Quick test both tasks
./eval_both_advanced.sh --quick

# Full evaluation with persistent directories
EVAL_PERSISTENT_DIR=true EVAL_USE_CACHE=false python3 eval_cell_selection_original_size.py

# Resume interrupted evaluation
EVAL_PERSISTENT_DIR=true EVAL_MODELS='gpt-5-mini' python3 eval_pointing_original_size.py
```

## Next Steps
1. Complete visualization tools for cell selection results
2. Finish refactoring evaluation runners
3. Add comprehensive testing
4. Document API costs and optimization strategies