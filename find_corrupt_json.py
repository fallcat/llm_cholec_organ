#!/usr/bin/env python3
"""Find corrupt or empty JSON files in the results directory."""

import json
from pathlib import Path

# Check CholecGoNoGo results
base_dir = Path("/shared_data0/weiqiuy/llm_cholec_organ/results/bbox_cholec_gonogo_quick")
mode = "zeroshot_combined"

mode_dir = base_dir / mode
if not mode_dir.exists():
    print(f"Directory not found: {mode_dir}")
    exit(1)

print(f"Checking JSON files in: {mode_dir}")
print("-" * 60)

corrupt_files = []
empty_files = []
valid_files = 0

# Check all model directories
for model_dir in sorted(mode_dir.iterdir()):
    if not model_dir.is_dir():
        continue
    
    print(f"\nChecking model: {model_dir.name}")
    
    # Check all test files
    test_files = sorted(model_dir.glob("test_*.json"))
    
    for test_file in test_files:
        try:
            # Check if file is empty
            if test_file.stat().st_size == 0:
                empty_files.append(test_file)
                print(f"  ❌ EMPTY: {test_file.name}")
                continue
            
            # Try to parse JSON
            with open(test_file, 'r') as f:
                json.load(f)
            valid_files += 1
            
        except json.JSONDecodeError as e:
            corrupt_files.append((test_file, str(e)))
            print(f"  ❌ CORRUPT: {test_file.name} - {e}")
        except Exception as e:
            corrupt_files.append((test_file, str(e)))
            print(f"  ❌ ERROR: {test_file.name} - {e}")
    
    if not test_files:
        print(f"  ⚠️  No test files found")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Valid files: {valid_files}")
print(f"Empty files: {len(empty_files)}")
print(f"Corrupt files: {len(corrupt_files)}")

if empty_files:
    print(f"\nEmpty files ({len(empty_files)}):")
    for f in empty_files[:5]:  # Show first 5
        print(f"  - {f}")
    if len(empty_files) > 5:
        print(f"  ... and {len(empty_files) - 5} more")

if corrupt_files:
    print(f"\nCorrupt files ({len(corrupt_files)}):")
    for f, err in corrupt_files[:5]:  # Show first 5
        print(f"  - {f}")
        print(f"    Error: {err}")
    if len(corrupt_files) > 5:
        print(f"  ... and {len(corrupt_files) - 5} more")

# Check for specific models that might be problematic
print("\n" + "=" * 60)
print("CHECKING SPECIFIC MODELS")
print("=" * 60)

models_to_check = ["CholeNet", "GoNoGoNet", "llava-hf_llava-v1.6-mistral-7b-hf"]
for model_name in models_to_check:
    model_dir = mode_dir / model_name
    if model_dir.exists():
        test_files = list(model_dir.glob("test_*.json"))
        print(f"{model_name}: {len(test_files)} files")
        
        # Check a sample file
        if test_files:
            sample_file = test_files[0]
            try:
                with open(sample_file, 'r') as f:
                    data = json.load(f)
                print(f"  ✓ Sample file valid, has keys: {list(data.keys())[:5]}")
            except Exception as e:
                print(f"  ✗ Sample file error: {e}")
    else:
        print(f"{model_name}: Directory not found")