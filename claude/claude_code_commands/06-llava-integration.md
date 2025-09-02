# LLaVA Integration and vLLM Fixes

## Date: 2025-01-09

## Overview
Successfully integrated LLaVA-1.6 (LLaVA-NeXT) model into the pointing evaluation pipeline with vLLM support, fixing multiple issues related to multi-modal prompt formatting and context length limitations.

## Changes Made

### 1. Added LLaVA to Evaluation Script
**File**: `notebooks_py/eval_pointing_original_size.py`
- Added `llava-hf/llava-v1.6-mistral-7b-hf` to DEFAULT_MODELS list
- Added environment variables for evaluation control:
  - `EVAL_SKIP_ZERO_SHOT`: Skip zero-shot evaluation
  - `EVAL_SKIP_FEW_SHOT`: Skip few-shot evaluation
- Updated documentation with usage examples

### 2. Updated Evaluator to Support LLaVA
**File**: `src/endopoint/eval/evaluator.py`
- Modified `load_model()` to recognize and load LLaVA models
- Added LLaVA model initialization with vLLM as default backend
- Added `skip_zero_shot` parameter to evaluation methods
- Enabled verbose mode for debugging

### 3. Fixed vLLM API Calls
**File**: `src/endopoint/models/vllm.py`

#### Issue 1: Missing Image Tokens
- **Problem**: vLLM requires explicit `<image>` tokens in prompts
- **Solution**: Modified prompt building to insert `<image>` tokens for each image
  - Single image: One `<image>` token
  - Multiple images (few-shot): Multiple `<image>` tokens interleaved with text

#### Issue 2: Incorrect vLLM generate() API
- **Problem**: Used keyword arguments instead of dictionary format
- **Solution**: Updated to official vLLM dictionary format:
```python
outputs = self.model.generate(
    {
        "prompt": prompt,
        "multi_modal_data": {"image": image}
    },
    sampling_params=self.sampling_params
)
```

#### Issue 3: Context Length Exceeded
- **Problem**: Few-shot learning with 3 images exceeded 4096 token limit
- **Solution**: Increased `max_model_len` from 4096 to 8192 tokens

#### Issue 4: Better Error Handling
- Added verbose logging for debugging
- Added warnings for empty responses
- Improved error messages with attempt counts
- Graceful fallback to transformers if vLLM fails

### 4. Enhanced Evaluation Control
**Files**: `src/endopoint/eval/evaluator.py`, `src/endopoint/eval/enhanced_evaluator.py`
- Added `skip_zero_shot` parameter to both base and enhanced evaluators
- Conditional execution of evaluation types based on flags
- Proper handling of empty fewshot_plans when few-shot is skipped

## Usage Examples

### Run only few-shot evaluation (skip zero-shot):
```bash
EVAL_SKIP_ZERO_SHOT=true python3 eval_pointing_original_size.py
```

### Quick test with LLaVA only:
```bash
EVAL_QUICK_TEST=true EVAL_MODELS='llava-hf/llava-v1.6-mistral-7b-hf' python3 eval_pointing_original_size.py
```

### Few-shot only with specific models:
```bash
EVAL_SKIP_ZERO_SHOT=true EVAL_MODELS='gpt-4o-mini,llava-hf/llava-v1.6-mistral-7b-hf' python3 eval_pointing_original_size.py
```

### Test with reduced samples:
```bash
EVAL_NUM_SAMPLES=5 EVAL_SKIP_ZERO_SHOT=true EVAL_MODELS='llava-hf/llava-v1.6-mistral-7b-hf' python3 eval_pointing_original_size.py
```

## Technical Details

### Image Token Handling
For vLLM, the prompt must contain `<image>` tokens that indicate where images should be processed:
- Text: "What organ is this?" + Image → Prompt: "What organ is this? <image>"
- Text + Image + Text + Image → Prompt: "Example: <image> Response: ... Query: <image>"

### Context Length Calculation
- Each image: ~1500-2000 tokens (depending on resolution)
- Few-shot with 3 images: ~4500-6000 tokens + text
- New limit (8192) provides comfortable headroom

### vLLM vs Transformers Backend
- vLLM: Faster inference, requires spawn multiprocessing, explicit image tokens
- Transformers: More stable, automatic image handling, slower inference
- Default: vLLM with automatic fallback to transformers on error

## Testing Status
- ✅ Zero-shot evaluation works with vLLM
- ✅ Few-shot evaluation works with increased context length
- ✅ Proper error handling and logging
- ✅ Environment variable controls working
- ✅ Model successfully generates responses (not empty)

## Known Limitations
1. vLLM requires multiprocessing spawn method (handled automatically)
2. Maximum 10 images per prompt (configured limit)
3. Context length still limited to 8192 tokens
4. vLLM may fail on some systems (automatic fallback to transformers)

## Future Improvements
1. Dynamic context length based on number of images
2. Optimize prompt length for few-shot examples
3. Add support for other VLM models (Qwen, Pixtral, DeepSeek)
4. Implement prompt compression for long contexts
5. Add model-specific configurations

## Files Modified
- `/notebooks_py/eval_pointing_original_size.py`
- `/src/endopoint/eval/evaluator.py`
- `/src/endopoint/eval/enhanced_evaluator.py`
- `/src/endopoint/models/vllm.py`

## Related Documentation
- `VLM_IMPLEMENTATION_SUMMARY.md`: Overall VLM implementation details
- `VLM_VLLM_FIX_SUMMARY.md`: vLLM API fix details
- `claude_code_commands/00-recap.md`: Project status and updates