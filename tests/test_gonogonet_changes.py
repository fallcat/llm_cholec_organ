#!/usr/bin/env python3
"""Test script to verify gonogonet name changes are complete."""

import sys
from pathlib import Path

print("="*80)
print("TESTING GONOGONET NAME CHANGES")
print("="*80)

# Check results directories
print("\n1. Checking existing result directories:")
print("-" * 40)

model_dirs = [
    Path('/shared_data0/weiqiuy/llm_cholec_organ/results/bbox_cholecseg8k_local_quick/zeroshot_combined'),
    Path('/shared_data0/weiqiuy/llm_cholec_organ/results/bbox_cholec_organs_quick/zeroshot_combined'),
    Path('/shared_data0/weiqiuy/llm_cholec_organ/results/bbox_cholec_gonogo_quick/zeroshot_combined')
]

for model_dir in model_dirs:
    if model_dir.exists():
        dataset_name = model_dir.parent.name
        print(f'\n{dataset_name}:')
        
        # Check for both gonogo and gonogonet directories
        gonogo_dir = model_dir / 'gonogo'
        gonogonet_dir = model_dir / 'gonogonet'
        
        if gonogo_dir.exists():
            test_files = list(gonogo_dir.glob('test_*.json'))
            print(f'  ✗ Found "gonogo" directory with {len(test_files)} test files')
            print(f'    Path: {gonogo_dir}')
        
        if gonogonet_dir.exists():
            test_files = list(gonogonet_dir.glob('test_*.json'))
            print(f'  ✓ Found "gonogonet" directory with {len(test_files)} test files')
            print(f'    Path: {gonogonet_dir}')
        
        if not gonogo_dir.exists() and not gonogonet_dir.exists():
            print(f'  - No gonogo or gonogonet directories found')

# Check model adapter
print("\n2. Checking GoNoGoNetAdapter default model name:")
print("-" * 40)

sys.path.insert(0, '/shared_data0/weiqiuy/llm_cholec_organ/src')

try:
    from endopoint.models.gonogonet_adapter import GoNoGoNetAdapter
    import inspect
    
    # Get the default value for model_name parameter
    sig = inspect.signature(GoNoGoNetAdapter.__init__)
    model_name_default = sig.parameters['model_name'].default
    
    if model_name_default == "gonogonet":
        print(f"  ✓ GoNoGoNetAdapter default model_name: '{model_name_default}'")
    else:
        print(f"  ✗ GoNoGoNetAdapter default model_name: '{model_name_default}' (should be 'gonogonet')")
    
    # Check cache directory
    adapter = GoNoGoNetAdapter()
    if "gonogonet" in str(adapter.cache_dir):
        print(f"  ✓ Default cache directory: {adapter.cache_dir}")
    else:
        print(f"  ✗ Default cache directory: {adapter.cache_dir} (should contain 'gonogonet')")
        
except Exception as e:
    print(f"  ✗ Error loading GoNoGoNetAdapter: {e}")

# Check shell scripts
print("\n3. Checking shell scripts for 'gonogo' references:")
print("-" * 40)

shell_scripts = [
    '/shared_data0/weiqiuy/llm_cholec_organ/run_cholenet_gonogo_full.sh',
    '/shared_data0/weiqiuy/llm_cholec_organ/run_all_cholenet_gonogo_eval.sh'
]

import subprocess

for script in shell_scripts:
    script_path = Path(script)
    if script_path.exists():
        # Check for 'gonogo' without 'gonogonet'
        result = subprocess.run(
            f"grep -c 'gonogo' {script} | grep -v 'gonogonet' || true",
            shell=True, capture_output=True, text=True
        )
        
        # Also check what's actually in EVAL_BATCH_MODELS
        models_result = subprocess.run(
            f"grep 'EVAL_BATCH_MODELS' {script} || true",
            shell=True, capture_output=True, text=True
        )
        
        print(f"\n  {script_path.name}:")
        if 'gonogonet' in models_result.stdout and 'gonogo"' not in models_result.stdout:
            print(f"    ✓ Uses 'gonogonet' in EVAL_BATCH_MODELS")
        else:
            print(f"    ✗ Still references 'gonogo' in EVAL_BATCH_MODELS")
        
        if models_result.stdout:
            print(f"    Found: {models_result.stdout.strip()}")

# Check Python files
print("\n4. Checking Python files for model name references:")
print("-" * 40)

python_files = [
    '/shared_data0/weiqiuy/llm_cholec_organ/notebooks_py/eval_bbox_unified.py',
    '/shared_data0/weiqiuy/llm_cholec_organ/run_cholenet_gonogo_with_summary.py'
]

for py_file in python_files:
    py_path = Path(py_file)
    if py_path.exists():
        with open(py_path, 'r') as f:
            content = f.read()
        
        print(f"\n  {py_path.name}:")
        
        # Check for gonogonet references
        if 'gonogonet' in content.lower():
            if 'elif model == "gonogonet"' in content:
                print(f"    ✓ Uses 'gonogonet' in model comparisons")
            if '"gonogonet"' in content or "'gonogonet'" in content:
                print(f"    ✓ Contains 'gonogonet' string references")
        
        # Check for old gonogo references (excluding comments and dataset names)
        lines_with_gonogo = []
        for i, line in enumerate(content.split('\n'), 1):
            if 'gonogo' in line.lower() and 'gonogonet' not in line.lower() and 'cholec_gonogo' not in line.lower():
                if not line.strip().startswith('#'):
                    lines_with_gonogo.append((i, line.strip()))
        
        if lines_with_gonogo:
            print(f"    ✗ Still has 'gonogo' references (not 'gonogonet'):")
            for line_no, line in lines_with_gonogo[:3]:  # Show first 3
                print(f"      Line {line_no}: {line[:80]}...")
        else:
            print(f"    ✓ No standalone 'gonogo' references found")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print("""
Expected state after changes:
1. All model references should use 'gonogonet' not 'gonogo'
2. Results should save to 'gonogonet/' directories
3. GoNoGoNetAdapter should default to model_name='gonogonet'
4. Shell scripts should use EVAL_BATCH_MODELS with 'gonogonet'
5. Python evaluation scripts should compare with 'gonogonet'

If you see any ✗ marks above, those items still need to be fixed.
""")

print("Next steps:")
print("1. If there are existing 'gonogo' directories with results, they may need to be renamed")
print("2. Run: mv results/*/zeroshot_combined/gonogo results/*/zeroshot_combined/gonogonet")
print("3. Then run your evaluation script to generate new results in 'gonogonet' directories")