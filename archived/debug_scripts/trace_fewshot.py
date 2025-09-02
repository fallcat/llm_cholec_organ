#!/usr/bin/env python3
"""
Simple trace to see if few-shot examples are being used.
Add this to eval_pointing.py temporarily to debug.
"""

print("""
============================================================
FEW-SHOT DEBUGGING INSTRUCTIONS
============================================================

The issue is that few-shot and zero-shot are giving the same results.

To debug this, we need to add tracing to the actual code.

STEP 1: Add debug prints to src/endopoint/eval/pointing.py
--------------------------------------------------------

Open src/endopoint/eval/pointing.py and add these lines after line 63:

    if few_shot_examples:
        print(f"[DEBUG] Few-shot: {len(few_shot_examples)} examples for {organ_name}")
    else:
        print(f"[DEBUG] Zero-shot for {organ_name}")

STEP 2: Add debug prints to src/endopoint/eval/evaluator.py
--------------------------------------------------------

Open src/endopoint/eval/evaluator.py and add after line 276:

    print(f"[DEBUG] Organ {organ_name}: {len(organ_examples)} examples")

STEP 3: Check if examples are actually different
--------------------------------------------------------

In src/endopoint/eval/evaluator.py, after line 175 add:

    print(f"[DEBUG] Total few-shot examples prepared: {sum(len(v) for v in few_shot_examples.values())}")

STEP 4: Run a quick test
--------------------------------------------------------

cd notebooks_py
EVAL_USE_CACHE=false EVAL_QUICK_TEST=true python3 eval_pointing.py 2>&1 | grep DEBUG

This will show:
- How many examples are prepared
- Whether they're being passed to the model
- Whether zero-shot and few-shot are actually different

STEP 5: Check the actual prompt content
--------------------------------------------------------

In src/llms.py, add after line 148:

    if self.use_cache:
        cache_key = get_cache_key(self.model_name, prompt, system_prompt)
        print(f"[CACHE_KEY] {self.model_name}: {cache_key[:16]}...")
        ret = cache.get(cache_key)

This will show if zero-shot and few-shot generate different cache keys.

============================================================
ALTERNATIVE: Direct Test
============================================================

If you want to test without modifying code, create this test file:
""")

print("""
# test_fewshot_direct.py

import sys
import os
sys.path.append('../src')

# Test cache key generation
from llms import get_cache_key

# Test 1: Simple strings
key1 = get_cache_key("gpt-4o-mini", "test", "system")
key2 = get_cache_key("gpt-4o-mini", "test", "system")
key3 = get_cache_key("gpt-4o-mini", "test different", "system")

print(f"Same prompt, same key: {key1 == key2}")
print(f"Different prompt, different key: {key1 != key3}")

# Test 2: Tuples (like few-shot)
zero_prompt = ("Is there a Liver?", "image_placeholder")
few_prompt = ("Example 1:", "img1", "Response: yes", "Now: Is there a Liver?", "image_placeholder")

key_zero = get_cache_key("gpt-4o-mini", zero_prompt, "system")
key_few = get_cache_key("gpt-4o-mini", few_prompt, "system")

print(f"Zero-shot vs few-shot different: {key_zero != key_few}")
print(f"  Zero key: {key_zero[:16]}...")
print(f"  Few key:  {key_few[:16]}...")

# If these are the SAME, there's a bug in cache key generation
# If they're DIFFERENT but results are same, the model is ignoring few-shot examples
""")

print("""
============================================================
MOST LIKELY ISSUES
============================================================

1. Few-shot examples not being loaded:
   - Check if fewshot_plan files exist in data_info/cholecseg8k/
   - Verify the plan has entries for each organ

2. Examples prepared but not passed:
   - The fix to evaluator.py line 288 should have fixed this
   - Verify the change is actually applied

3. Cache key collision:
   - If cache keys are the same for different prompts
   - Clear cache and run without cache to verify

4. Model ignoring examples:
   - Some models might not respond differently to few-shot
   - Try with a different model

Run the debug steps above to identify which issue you have.
""")