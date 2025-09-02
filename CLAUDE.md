# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based medical computer vision research project that uses Large Vision-Language Models (LLMs) for fine-grained organ detection in laparoscopic cholecystectomy procedures. The project evaluates various LLMs (GPT, Claude, Gemini) on organ presence detection and localization tasks using the CholecSeg8k dataset.

## Key Commands

### Running the Main Code

```bash
# Run the main evaluation pipeline (expensive - runs all models and baselines)
cd src
python cholec.py

# Run dataset utilities demo
cd src
python cholecseg8k_utils.py

# Run Jupyter notebooks for experiments
jupyter notebook
# Then open notebooks/ directory
```

### API Setup Required

Create `API_KEYS2.json` in the root directory with:
```json
{
    "OPENAI_API_KEY": "your-key",
    "ANTHROPIC_API_KEY": "your-key", 
    "GOOGLE_API_KEY": "your-key"
}
```

### Current Development Workflow

1. Most development happens in Jupyter notebooks under `notebooks/`
2. Core modules are in `src/` - modify these for shared functionality
3. Results are cached in `cache/` and saved to `results/`
4. Visualizations are saved to `vis/`

## Architecture & Code Structure

### Core Modules

- **`src/llms.py`** - Unified interface for LLM models (OpenAI, Anthropic, Google)
  - Contains `LLMModel` class with caching and retry logic
  - Handles API calls and response parsing

- **`src/cholec.py`** - Main dataset handling and evaluation logic
  - `CholecExample` class for dataset samples
  - `run_baselines()` function for model evaluation
  - Organ detection and pointing task implementations

- **`src/cholecseg8k_utils.py`** - CholecSeg8k dataset utilities
  - Dataset loading from HuggingFace
  - Balanced sampling algorithms
  - Presence matrix computation

- **`src/prompts/explanations.py`** - Prompt templates for LLM queries
  - Different prompt strategies (base, strict, qna, fewshot)
  - Organ detection prompts with visual explanations

### Key Concepts

1. **Organ Classes**: 12 classes including Liver, Gallbladder, Hepatocystic Triangle, Fat, Grasper, etc.

2. **Tasks**:
   - **Existence**: Binary classification of organ presence
   - **Pointing**: Localization with (x,y) coordinates when present
   - **Few-shot**: Using positive/negative examples for better performance

3. **Caching**: All LLM responses are cached to disk using hash-based keys to avoid redundant API calls

4. **Balanced Sampling**: Greedy algorithm ensures balanced representation of all organ types in training/evaluation sets

### Refactoring Plans

The project is planned to be refactored into a modular package called `endopoint` (see `claude_code_commands/01-refactor-code.md`). Key changes will include:
- Protocol-based adapters for multiple datasets
- Proper CLI tools
- Testing infrastructure with pytest
- Type hints throughout
- Configuration-driven experiments

## Development Notes

- **No formal testing/linting setup yet** - code quality relies on manual review
- **Research-oriented codebase** - prioritizes experimentation over production readiness
- **Heavy API usage** - be mindful of costs when running full evaluations
- **Large dataset** - CholecSeg8k requires significant disk space
- **Jupyter-centric workflow** - most experiments start in notebooks before moving to modules

When making changes:
1. Test in notebooks first for rapid iteration
2. Move stable code to src/ modules
3. Maintain backward compatibility with existing notebooks
4. Use type hints for new code
5. Document any new prompt strategies or model configurations

## Refactoring Status

The codebase is being refactored into a modular package called `endopoint`. Progress:

### Completed
- ✅ Package scaffold with pyproject.toml
- ✅ Dataset adapter protocol and CholecSeg8k implementation
- ✅ Utility modules (io, logging, rng)
- ✅ Geometry/canvas transformations
- ✅ Prompt builders and registry
- ✅ Model adapters (OpenAI, Anthropic, Google)
- ✅ JSON parser with fallbacks
- ✅ Presence cache implementation
- ✅ Basic CLI structure
- ✅ **Cell selection implementation** for grid-based localization (alternative to pointing)

### In Progress
- 🔄 Evaluation runner
- 🔄 Selection algorithms
- 🔄 Few-shot learning
- 🔄 Metrics implementation
- 🔄 Complete CLI suite

### Cell Selection (New Feature)
Cell selection provides a discrete alternative to exact (x,y) pointing:
- **Module**: `src/endopoint/eval/cell_selection.py` - ground truth and metrics
- **Evaluation**: `notebooks_py/eval_cell_selection_original_size.py` - full pipeline
- **Configuration**: Supports G∈{3,4} grids and K∈{1,3} top cells
- **Documentation**: See `notebooks_py/CELL_SELECTION_README.md` for usage

### Usage
The new package can be installed with:
```bash
pip install -e .
```

And used via CLI tools:
```bash
endopoint-build-cache --dataset cholecseg8k
```

See `claude_code_commands/01-refactor-code.md` for the full refactoring plan.

### Recap
Whenever you start looking at the project again, you should check this file for the current progress, and then update it concisely after you finish every day's work.
`claude_code_commands/00-recap.md`