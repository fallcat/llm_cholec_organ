#!/usr/bin/env python3
"""
Test if cache keys are different for zero-shot vs few-shot.
This is the core issue - if keys are the same, we get cached results.
"""

import sys
import os
import hashlib
import pickle

sys.path.append('../src')

# Import the cache key function
from llms import get_cache_key

print("="*60)
print("TESTING CACHE KEY GENERATION")
print("="*60)

# Test 1: Basic test with strings
print("\n1. Basic string test:")
key1 = get_cache_key("gpt-4o-mini", "test prompt", "system")
key2 = get_cache_key("gpt-4o-mini", "test prompt", "system")
key3 = get_cache_key("gpt-4o-mini", "different prompt", "system")

print(f"   Same prompt → same key: {key1 == key2} {'✓' if key1 == key2 else '✗'}")
print(f"   Different prompt → different key: {key1 != key3} {'✓' if key1 != key3 else '✗'}")

# Test 2: Tuple prompts (like what pointing uses)
print("\n2. Tuple prompt test (simulating zero-shot vs few-shot):")

# Simulate zero-shot: just query and image
zero_shot_prompt = (
    "Is there a Liver in this image? Respond with JSON.",
    "image_placeholder"
)

# Simulate few-shot: examples + query + image
few_shot_prompt = (
    "Here are some examples:\n",
    "\nExample 1: Is there a Liver in this image? Respond with JSON.",
    "example_image_1",
    '\nResponse: {"name":"Liver","present":1,"point_canvas":[400,400]}\n',
    "\nExample 2: Is there a Liver in this image? Respond with JSON.",
    "example_image_2", 
    '\nResponse: {"name":"Liver","present":0,"point_canvas":null}\n',
    "\nNow for the actual query: Is there a Liver in this image? Respond with JSON.",
    "image_placeholder"
)

key_zero = get_cache_key("gpt-4o-mini", zero_shot_prompt, "system_prompt")
key_few = get_cache_key("gpt-4o-mini", few_shot_prompt, "system_prompt")

print(f"   Zero-shot prompt parts: {len(zero_shot_prompt)}")
print(f"   Few-shot prompt parts: {len(few_shot_prompt)}")
print(f"   Different keys: {key_zero != key_few} {'✓' if key_zero != key_few else '✗'}")
print(f"   Zero-shot key: {key_zero[:16]}...")
print(f"   Few-shot key:  {key_few[:16]}...")

# Test 3: Edge case - empty few-shot (should be same as zero-shot)
print("\n3. Edge case - empty few-shot list:")
empty_few_shot_prompt = zero_shot_prompt  # If no examples, should be same as zero-shot
key_empty = get_cache_key("gpt-4o-mini", empty_few_shot_prompt, "system_prompt")

print(f"   Empty few-shot same as zero-shot: {key_empty == key_zero} {'✓' if key_empty == key_zero else '✗'}")

print("\n" + "="*60)
print("CONCLUSION")
print("="*60)

if key_zero != key_few:
    print("✅ Cache keys are DIFFERENT for zero-shot vs few-shot")
    print("   This means the cache system is working correctly.")
    print("\n   If you're still getting same results, the issue is likely:")
    print("   1. Few-shot examples are not being prepared (empty list)")
    print("   2. Few-shot examples are not being passed to the model")
    print("   3. The model is ignoring the few-shot examples")
else:
    print("❌ Cache keys are the SAME for zero-shot vs few-shot")
    print("   This is a BUG in cache key generation!")

print("\nTo check if examples are being prepared, look for debug output:")
print("   [DEBUG FEW-SHOT] Total examples prepared: N")
print("   [DEBUG POINTING] Organ: Using N few-shot examples")
print("\nIf N is 0, the problem is in example preparation.")
print("If N > 0 but results are same, the model might be ignoring examples.")