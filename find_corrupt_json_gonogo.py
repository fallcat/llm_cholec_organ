#!/usr/bin/env python3
"""Find corrupted JSON files in CholecGoNoGo results."""

import json
from pathlib import Path

results_dir = Path("/shared_data0/weiqiuy/llm_cholec_organ/results/bbox_cholec_gonogo_quick/zeroshot_combined")

corrupt_files = []
empty_files = []
valid_files = 0

# Check all JSON files in the directory
for model_dir in results_dir.iterdir():
    if not model_dir.is_dir():
        continue
    
    print(f"\nChecking {model_dir.name}...")
    
    for json_file in model_dir.glob("test_*.json"):
        try:
            # Check if file is empty
            if json_file.stat().st_size == 0:
                empty_files.append(str(json_file))
                print(f"  EMPTY: {json_file.name}")
                continue
            
            # Try to load the JSON
            with open(json_file, 'r') as f:
                content = f.read()
                if not content.strip():
                    empty_files.append(str(json_file))
                    print(f"  EMPTY (whitespace only): {json_file.name}")
                else:
                    data = json.loads(content)
                    valid_files += 1
        except json.JSONDecodeError as e:
            corrupt_files.append((str(json_file), str(e)))
            print(f"  CORRUPT: {json_file.name} - {e}")
        except Exception as e:
            corrupt_files.append((str(json_file), str(e)))
            print(f"  ERROR: {json_file.name} - {e}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Valid files: {valid_files}")
print(f"Empty files: {len(empty_files)}")
print(f"Corrupt files: {len(corrupt_files)}")

if empty_files:
    print("\nEmpty files:")
    for f in empty_files[:10]:  # Show first 10
        print(f"  {f}")
    if len(empty_files) > 10:
        print(f"  ... and {len(empty_files) - 10} more")

if corrupt_files:
    print("\nCorrupt files:")
    for f, error in corrupt_files[:10]:  # Show first 10
        print(f"  {f}")
        print(f"    Error: {error}")
    if len(corrupt_files) > 10:
        print(f"  ... and {len(corrupt_files) - 10} more")