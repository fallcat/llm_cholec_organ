#!/usr/bin/env python3
"""Test script to verify the GoNoGoNet refactoring is complete and working."""

import sys
import os
from pathlib import Path

print("="*80)
print("TESTING GONOGONET REFACTORING")
print("="*80)

# Add src to path
sys.path.insert(0, '/shared_data0/weiqiuy/llm_cholec_organ/src')

# Test 1: Check file renaming
print("\n1. Checking file renaming:")
print("-" * 40)

old_files = [
    Path('/shared_data0/weiqiuy/llm_cholec_organ/src/endopoint/models/gonogo.py'),
    Path('/shared_data0/weiqiuy/llm_cholec_organ/src/endopoint/models/gonogo_adapter.py')
]

new_files = [
    Path('/shared_data0/weiqiuy/llm_cholec_organ/src/endopoint/models/gonogonet.py'),
    Path('/shared_data0/weiqiuy/llm_cholec_organ/src/endopoint/models/gonogonet_adapter.py')
]

for old_file in old_files:
    if old_file.exists():
        print(f"  ✗ Old file still exists: {old_file.name}")
    else:
        print(f"  ✓ Old file removed: {old_file.name}")

for new_file in new_files:
    if new_file.exists():
        print(f"  ✓ New file exists: {new_file.name}")
    else:
        print(f"  ✗ New file missing: {new_file.name}")

# Test 2: Import the new modules
print("\n2. Testing module imports:")
print("-" * 40)

try:
    from endopoint.models.gonogonet import GoNoGoNet, load_gonogo_model
    print("  ✓ Successfully imported from gonogonet module")
except ImportError as e:
    print(f"  ✗ Failed to import from gonogonet: {e}")

try:
    from endopoint.models.gonogonet_adapter import GoNoGoNetAdapter
    print("  ✓ Successfully imported GoNoGoNetAdapter")
except ImportError as e:
    print(f"  ✗ Failed to import GoNoGoNetAdapter: {e}")

# Test 3: Check class instantiation
print("\n3. Testing class instantiation:")
print("-" * 40)

try:
    # Test that GoNoGoNetAdapter can be instantiated
    import inspect
    from endopoint.models.gonogonet_adapter import GoNoGoNetAdapter
    
    sig = inspect.signature(GoNoGoNetAdapter.__init__)
    model_name_default = sig.parameters['model_name'].default
    
    if model_name_default == "gonogonet":
        print(f"  ✓ GoNoGoNetAdapter default model_name: '{model_name_default}'")
    else:
        print(f"  ✗ GoNoGoNetAdapter default model_name: '{model_name_default}' (should be 'gonogonet')")
    
    # Try to create an instance (will fail if model weights aren't available, but that's OK)
    try:
        adapter = GoNoGoNetAdapter(use_cache=False, verbose=False)
        print("  ✓ GoNoGoNetAdapter instance created successfully")
    except FileNotFoundError as e:
        # This is expected if model weights aren't downloaded
        if "checkpoint" in str(e).lower() or "model" in str(e).lower():
            print("  ✓ GoNoGoNetAdapter instantiation works (model weights not found, which is expected)")
        else:
            print(f"  ✗ Unexpected error: {e}")
    except Exception as e:
        print(f"  ✗ Error creating GoNoGoNetAdapter: {e}")
        
except Exception as e:
    print(f"  ✗ Error in instantiation test: {e}")

# Test 4: Check model creation factory
print("\n4. Testing model creation factory:")
print("-" * 40)

try:
    from endopoint.models import create_model
    
    # Test that gonogonet model ID creates the right adapter
    model = create_model("gonogonet", use_cache=False, verbose=False)
    
    if model.__class__.__name__ == "GoNoGoNetAdapter":
        print(f"  ✓ create_model('gonogonet') returns GoNoGoNetAdapter")
    else:
        print(f"  ✗ create_model('gonogonet') returns {model.__class__.__name__}")
        
except Exception as e:
    print(f"  ✗ Error testing create_model: {e}")

# Test 5: Check __all__ exports
print("\n5. Testing module exports:")
print("-" * 40)

try:
    import endopoint.models as models
    
    if "GoNoGoNetAdapter" in models.__all__:
        print("  ✓ GoNoGoNetAdapter is in __all__ exports")
    else:
        print("  ✗ GoNoGoNetAdapter not in __all__ exports")
        
    if "GoNoGoAdapter" in models.__all__:
        print("  ✗ Old GoNoGoAdapter still in __all__ exports")
    else:
        print("  ✓ Old GoNoGoAdapter removed from __all__ exports")
        
except Exception as e:
    print(f"  ✗ Error checking exports: {e}")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print("""
The refactoring changes:
1. gonogo.py → gonogonet.py
2. gonogo_adapter.py → gonogonet_adapter.py  
3. GoNoGoAdapter → GoNoGoNetAdapter
4. All imports and references updated
5. Model name defaults to 'gonogonet'

Next steps:
1. Run the full evaluation: ./run_cholenet_gonogo_full.sh
2. Results will save to 'gonogonet/' directories
3. Old 'gonogo/' directories can be renamed or removed
""")

print("\nTo test the full pipeline:")
print("  EVAL_MODEL=gonogonet EVAL_DATASET=cholec_gonogo EVAL_NUM_SAMPLES=2 python notebooks_py/eval_bbox_unified.py")