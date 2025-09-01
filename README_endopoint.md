# EndoPoint: Large Vision-Language Models for Fine-Grained Organ Detection

A modular Python package for organ detection in endoscopic videos using Large Vision-Language Models.

## Installation

```bash
pip install -e .
```

## Quick Start

1. Set up API keys:
```bash
export OPENAI_API_KEY=your-key
export ANTHROPIC_API_KEY=your-key
export GOOGLE_API_KEY=your-key
```

2. Build presence cache:
```bash
endopoint-build-cache --dataset cholecseg8k --split train
```

## Architecture

The package is organized into modular components:

- **datasets/**: Dataset adapters with a common protocol
- **models/**: LLM model adapters (OpenAI, Anthropic, Google)
- **prompts/**: Prompt builders and registry
- **geometry/**: Canvas and coordinate transformations
- **data/**: Data processing utilities (presence cache)
- **eval/**: Evaluation runners and parsers
- **cli/**: Command-line tools

## Adding a New Dataset

1. Create a new adapter in `src/endopoint/datasets/`:

```python
from .base import register_dataset

class MyDatasetAdapter:
    # Implement the DatasetAdapter protocol
    ...

@register_dataset("mydataset")
def build_mydataset(**cfg):
    return MyDatasetAdapter(**cfg)
```

2. Create a config file in `configs/`:

```yaml
name: mydataset
root: /path/to/dataset
recommended_canvas: [768, 768]
```

3. Use with any CLI tool:

```bash
endopoint-build-cache --dataset mydataset
```

## Development Status

This is a refactored version of the research codebase. Core modules implemented:

- ✅ Dataset adapter protocol and CholecSeg8k implementation
- ✅ Model adapters for OpenAI, Anthropic, Google
- ✅ Prompt management system
- ✅ Geometry utilities
- ✅ Presence cache
- ✅ JSON parser with fallbacks
- ⏳ Evaluation runner
- ⏳ Selection algorithms
- ⏳ Few-shot learning
- ⏳ Visualization tools
- ⏳ Full CLI suite

See `claude_code_commands/01-refactor-code.md` for the complete refactoring plan.