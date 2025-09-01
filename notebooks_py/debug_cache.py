#!/usr/bin/env python3
"""
Debug script to verify cache behavior and few-shot examples.
"""

import sys
import os
import hashlib
import pickle
from pathlib import Path

# Add src to path
sys.path.append(os.path.join("..", "src"))

def check_cache_locations():
    """Check which cache directories exist."""
    print("=" * 60)
    print("CACHE LOCATIONS CHECK")
    print("=" * 60)
    
    # Old cache location
    old_cache = Path("../src/.llms.py.cache")
    if old_cache.exists():
        print(f"✓ Old cache exists: {old_cache}")
        # Try to count entries
        try:
            import diskcache
            cache = diskcache.Cache(str(old_cache))
            print(f"  Entries: {len(list(cache.iterkeys()))}")
            cache.close()
        except:
            print("  Could not count entries")
    else:
        print(f"✗ Old cache not found: {old_cache}")
    
    # New cache location
    new_cache = Path.home() / ".cache" / "endopoint" / "models"
    if new_cache.exists():
        print(f"✓ New cache exists: {new_cache}")
        try:
            import diskcache
            cache = diskcache.Cache(str(new_cache))
            print(f"  Entries: {len(list(cache.iterkeys()))}")
            cache.close()
        except:
            print("  Could not count entries")
    else:
        print(f"✗ New cache not found: {new_cache}")
    
    print()


def test_cache_key_generation():
    """Test that different prompts generate different cache keys."""
    print("=" * 60)
    print("CACHE KEY GENERATION TEST")
    print("=" * 60)
    
    from llms import get_cache_key
    
    # Test 1: Same prompt should give same key
    key1 = get_cache_key("gpt-4o-mini", "Test prompt", "System prompt")
    key2 = get_cache_key("gpt-4o-mini", "Test prompt", "System prompt")
    print(f"Same prompt, same key: {key1 == key2} ✓" if key1 == key2 else f"ERROR: Keys differ!")
    
    # Test 2: Different prompts should give different keys
    key3 = get_cache_key("gpt-4o-mini", "Different prompt", "System prompt")
    print(f"Different prompt, different key: {key1 != key3} ✓" if key1 != key3 else f"ERROR: Keys same!")
    
    # Test 3: Zero-shot vs few-shot should differ
    zero_shot_prompt = ("Is there a liver?", "image_placeholder")
    few_shot_prompt = ("Example 1: Is there a liver?", "ex_image", "Response: yes", 
                       "Now: Is there a liver?", "image_placeholder")
    
    key_zero = get_cache_key("gpt-4o-mini", zero_shot_prompt, "System")
    key_few = get_cache_key("gpt-4o-mini", few_shot_prompt, "System")
    print(f"Zero-shot vs few-shot, different keys: {key_zero != key_few} ✓" if key_zero != key_few else f"ERROR: Keys same!")
    
    print(f"\nExample keys:")
    print(f"  Zero-shot: {key_zero[:16]}...")
    print(f"  Few-shot:  {key_few[:16]}...")
    print()


def check_few_shot_plan():
    """Check if few-shot plans are being loaded correctly."""
    print("=" * 60)
    print("FEW-SHOT PLAN CHECK")
    print("=" * 60)
    
    data_dir = Path("../data_info/cholecseg8k")
    
    # Check for few-shot plan files
    plan_files = {
        "standard": data_dir / "fewshot_plan_train_pos1_neg1_seed43_excl100.json",
        "hard_negatives": data_dir / "fewshot_plan_train_pos1_neg1_nearmiss1_seed45_excl100.json",
    }
    
    for plan_name, plan_file in plan_files.items():
        if plan_file.exists():
            print(f"✓ {plan_name} plan exists: {plan_file.name}")
            
            # Load and check content
            import json
            with open(plan_file) as f:
                plan = json.load(f)
            
            # Check a sample organ
            sample_organ = "Liver"
            if sample_organ in plan:
                organ_plan = plan[sample_organ]
                print(f"  {sample_organ}:")
                print(f"    Positive examples: {len(organ_plan.get('positive', []))}")
                print(f"    Negative (easy): {len(organ_plan.get('negative_easy', []))}")
                print(f"    Negative (hard): {len(organ_plan.get('negative_hard', []))}")
        else:
            print(f"✗ {plan_name} plan not found: {plan_file}")
    
    print()


def test_model_with_cache():
    """Test a simple model call with and without cache."""
    print("=" * 60)
    print("MODEL CACHE BEHAVIOR TEST")
    print("=" * 60)
    
    # Check if API keys are available
    api_keys_file = Path("../API_KEYS2.json")
    if not api_keys_file.exists():
        print("✗ API_KEYS2.json not found - cannot test model calls")
        return
    
    import json
    with open(api_keys_file) as f:
        api_keys = json.load(f)
    
    os.environ['OPENAI_API_KEY'] = api_keys.get('OPENAI_API_KEY', '')
    
    if not os.environ['OPENAI_API_KEY']:
        print("✗ No OpenAI API key found")
        return
    
    try:
        from llms import MyOpenAIModel
        
        # Test with cache disabled
        print("\n1. Testing with cache DISABLED:")
        model_no_cache = MyOpenAIModel(
            model_name="gpt-4o-mini",
            use_cache=False,
            max_tokens=10
        )
        
        test_prompt = "Say 'test' and nothing else"
        response1 = model_no_cache(test_prompt)
        print(f"   First call: '{response1}'")
        
        response2 = model_no_cache(test_prompt)
        print(f"   Second call: '{response2}'")
        print(f"   Same response: {response1 == response2}")
        
        # Test with cache enabled
        print("\n2. Testing with cache ENABLED:")
        model_with_cache = MyOpenAIModel(
            model_name="gpt-4o-mini",
            use_cache=True,
            max_tokens=10
        )
        
        response3 = model_with_cache(test_prompt)
        print(f"   First call: '{response3}'")
        
        response4 = model_with_cache(test_prompt)
        print(f"   Second call: '{response4}'")
        print(f"   Same response (should be True): {response3 == response4}")
        
    except Exception as e:
        print(f"✗ Error testing model: {e}")
    
    print()


def main():
    """Run all debug checks."""
    print("\n" + "=" * 60)
    print("CACHE DEBUG REPORT")
    print("=" * 60 + "\n")
    
    # Check cache locations
    check_cache_locations()
    
    # Test cache key generation
    test_cache_key_generation()
    
    # Check few-shot plans
    check_few_shot_plan()
    
    # Test model with cache
    test_model_with_cache()
    
    print("=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    print("""
1. If zero-shot and few-shot give same results with cache:
   - Clear all caches: rm -rf ../src/.llms.py.cache ~/.cache/endopoint/models/
   - Run with EVAL_USE_CACHE=false to verify they differ
   - Check that few-shot examples are actually being passed

2. To verify few-shot is working:
   - Add debug prints in src/endopoint/eval/pointing.py line 63
   - Print len(few_shot_examples) to ensure they exist
   - Print the prompt_parts to see what's being sent

3. For production:
   - Always use EVAL_USE_CACHE=false during development
   - Only enable cache for final evaluation runs
""")


if __name__ == "__main__":
    main()