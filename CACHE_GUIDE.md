# Cache Management Guide

## Overview

The codebase uses multiple cache mechanisms to avoid redundant API calls and computations. This guide documents all cache types, locations, and how to manage them.

## Cache Types and Locations

### 1. LLM Response Cache (Primary Issue)

**Purpose**: Caches API responses from OpenAI, Anthropic, and Google models to avoid repeated API calls.

**Locations**:
- **Old implementation** (`src/llms.py`): 
  - Location: `src/.llms.py.cache/`
  - Type: DiskCache database
  - Key: SHA256 hash of (model_name, system_prompt, user_prompt)

- **New implementation** (`src/endopoint/models/`):
  - Location: `~/.cache/endopoint/models/`
  - Type: DiskCache database
  - Key: SHA256 hash of (model_name, system_prompt, user_prompt)

**Why it's a problem**: 
- If you change prompt formatting but the content hashes to the same value, you get old responses
- Cache persists across runs unless explicitly cleared
- Different cache locations for old vs new code

### 2. Presence Matrix Cache

**Purpose**: Caches organ presence matrices for dataset samples to speed up data loading.

**Location**: `/shared_data0/weiqiuy/llm_cholec_organ/cache/presence/`

**Files**:
- `presence_CholecSeg8k_train_N{num}_mp50_{hash}.npz` - Numpy arrays
- `presence_CholecSeg8k_train_N{num}_mp50_{hash}.meta.json` - Metadata

**Generally not problematic** unless dataset changes.

### 3. HuggingFace Dataset Cache

**Purpose**: Caches downloaded CholecSeg8k dataset.

**Location**: `~/.cache/huggingface/datasets/`

**Generally not problematic** - managed by HuggingFace.

## How to Clear Caches

### Option 1: Disable Cache in Code

When running evaluation:
```bash
# Disable cache for fresh API calls
EVAL_USE_CACHE=false python3 notebooks_py/eval_pointing.py
```

### Option 2: Clear Cache Directories

```bash
# Clear old LLM cache (if it exists)
rm -rf /shared_data0/weiqiuy/llm_cholec_organ/src/.llms.py.cache

# Clear new LLM cache
rm -rf ~/.cache/endopoint/models/

# Clear presence cache (rarely needed)
rm -rf /shared_data0/weiqiuy/llm_cholec_organ/cache/presence/

# Clear HuggingFace cache (rarely needed)
rm -rf ~/.cache/huggingface/datasets/minwoosun___cholec_seg8k/
```

### Option 3: Programmatic Cache Control

In Python scripts:
```python
# Old implementation
from llms import MyOpenAIModel
model = MyOpenAIModel(use_cache=False)  # Disable cache

# New implementation
from endopoint.models import OpenAIAdapter
model = OpenAIAdapter(use_cache=False)  # Disable cache

# Clear cache programmatically
import diskcache
cache = diskcache.Cache("src/.llms.py.cache")
cache.clear()  # Clear all entries
```

### Option 4: Selective Cache Clearing

Clear only specific model caches:
```python
import diskcache
import hashlib
import pickle

# Open cache
cache = diskcache.Cache("src/.llms.py.cache")

# List all keys (warning: can be many)
for key in list(cache.iterkeys())[:10]:  # First 10
    print(key)

# Clear entries for specific model
model_name = "gpt-4o-mini"
for key in list(cache.iterkeys()):
    # Keys are SHA256 hashes, so we can't easily filter
    # Best to clear all or use use_cache=False
    pass
```

## Debugging Cache Issues

### 1. Check if cache is being used

Add debug prints to see cache hits:
```python
# In src/llms.py around line 148
if self.use_cache:
    ret = cache.get(get_cache_key(self.model_name, prompt, system_prompt))
    if ret is not None and ret != "":
        print(f"CACHE HIT for {self.model_name}")  # Add this
        return ret
    print(f"CACHE MISS for {self.model_name}")  # Add this
```

### 2. Verify cache location

```bash
# Check which cache directories exist
ls -la src/.llms.py.cache 2>/dev/null && echo "Old cache exists"
ls -la ~/.cache/endopoint/models/ 2>/dev/null && echo "New cache exists"
```

### 3. Monitor cache size

```bash
# Check cache size
du -sh src/.llms.py.cache 2>/dev/null
du -sh ~/.cache/endopoint/models/ 2>/dev/null
```

## Best Practices

1. **During Development**: Always use `use_cache=False` or `EVAL_USE_CACHE=false` when testing prompt changes.

2. **Clear Cache After**:
   - Changing prompt templates
   - Updating model parameters
   - Switching between model versions
   - Debugging unexpected responses

3. **Keep Cache For**:
   - Production runs with stable prompts
   - Large-scale evaluations
   - When API costs are a concern

## Quick Commands

```bash
# Complete cache reset
rm -rf src/.llms.py.cache ~/.cache/endopoint/models/

# Run evaluation without cache
EVAL_USE_CACHE=false python3 notebooks_py/eval_pointing.py

# Run with cache (default)
python3 notebooks_py/eval_pointing.py
```

## Common Issues and Solutions

### Issue: Getting old responses despite prompt changes
**Solution**: Clear cache or run with `use_cache=False`

### Issue: Evaluation results don't match expectations
**Solution**: 
1. Clear all caches
2. Run with `EVAL_USE_CACHE=false`
3. Verify prompts are being constructed correctly

### Issue: Running out of disk space
**Solution**: Clear old cache directories, especially `src/.llms.py.cache` which can grow large

### Issue: Can't find cache directory
**Solution**: The cache directory is created on first use. If it doesn't exist, the cache hasn't been used yet.

## Cache Key Generation

The cache key is generated from:
1. Model name (e.g., "gpt-4o-mini")
2. System prompt (if provided)
3. User prompt (text and/or images)

All these are serialized with pickle and hashed with SHA256. This means:
- Same inputs always produce the same cache key
- Slightly different prompts produce completely different keys
- Image prompts are base64-encoded before hashing