#!/usr/bin/env python3
"""Check Hepatic Vein ground truth distribution in test files."""

import json
from pathlib import Path
from collections import Counter

# Directory to check - using the correct path
results_dir = Path("/shared_data0/weiqiuy/llm_cholec_organ/results/bbox_cholecseg8k_local_quick/zeroshot_combined")

# Get one model's results
model_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
if not model_dirs:
    print("No model directories found")
    exit(1)

# Use the first model's test files
model_dir = model_dirs[0]
print(f"Checking test files in: {model_dir.name}")

# Count Hepatic Vein ground truth values
hepatic_vein_gt = Counter()
total_files = 0
files_with_hepatic_vein = []

for test_file in model_dir.glob("test_*.json"):
    total_files += 1
    try:
        with open(test_file, 'r') as f:
            data = json.load(f)
        
        for organ in data.get('organs', []):
            organ_name = organ.get('organ_name') or organ.get('name') or organ.get('organ')
            if organ_name == 'Hepatic Vein':
                gt_value = organ.get('ground_truth_present', 0)
                hepatic_vein_gt[gt_value] += 1
                if gt_value == 1:
                    files_with_hepatic_vein.append(test_file.name)
                break
    except Exception as e:
        print(f"Error reading {test_file}: {e}")
        continue

print(f"\nTotal test files: {total_files}")
print(f"Hepatic Vein ground truth distribution:")
print(f"  GT=0 (absent): {hepatic_vein_gt[0]}")
print(f"  GT=1 (present): {hepatic_vein_gt[1]}")

if files_with_hepatic_vein:
    print(f"\nFiles with Hepatic Vein present (GT=1):")
    for f in files_with_hepatic_vein[:10]:  # Show first 10
        print(f"  {f}")
    if len(files_with_hepatic_vein) > 10:
        print(f"  ... and {len(files_with_hepatic_vein) - 10} more")
else:
    print("\nNo files have Hepatic Vein present in ground truth!")

# Check all organs' presence distribution
print("\n" + "="*60)
print("Checking all organs' ground truth distribution...")
organ_gt_counts = Counter()

for test_file in model_dir.glob("test_*.json"):
    try:
        with open(test_file, 'r') as f:
            data = json.load(f)
        
        for organ in data.get('organs', []):
            organ_name = organ.get('organ_name') or organ.get('name') or organ.get('organ')
            if organ_name:
                gt_value = organ.get('ground_truth_present', 0)
                organ_gt_counts[(organ_name, gt_value)] += 1
    except Exception as e:
        continue

# Print organ presence statistics
print("\nOrgan presence statistics (number of test samples):")
print(f"{'Organ':<30} {'Present':<10} {'Absent':<10} {'Total':<10}")
print("-" * 60)

all_organs = set(organ for organ, _ in organ_gt_counts.keys())
for organ in sorted(all_organs):
    present = organ_gt_counts.get((organ, 1), 0)
    absent = organ_gt_counts.get((organ, 0), 0)
    total = present + absent
    print(f"{organ:<30} {present:<10} {absent:<10} {total:<10}")