#!/usr/bin/env python3
"""Test script to verify model discovery across all datasets."""

import json
from pathlib import Path

# Define result directories for all three datasets
results_dirs = {
    'CholecSeg8k': Path('/shared_data0/weiqiuy/llm_cholec_organ/results/bbox_cholecseg8k_local_quick'),
    'CholecOrgans': Path('/shared_data0/weiqiuy/llm_cholec_organ/results/bbox_cholec_organs_quick'),
    'CholecGoNoGo': Path('/shared_data0/weiqiuy/llm_cholec_organ/results/bbox_cholec_gonogo_quick')
}

# Extended mapping from directory names to display names
MODEL_NAME_MAPPING = {
    # Naive baselines
    "naive_baseline_perfect": "Native Baseline",
    "naive_baseline_all": "Naive (All)",
    "naive_baseline_all_full": "Naive (Full Box)",
    "naive_baseline_all_random": "Naive (Random Box)",
    
    # Commercial LVLMs
    "gpt-4.1": "GPT-4.1",
    "gemini-2.0-flash": "Gemini-2.0-Flash",
    "claude-sonnet-4-20250514": "Claude-Sonnet-4",
    
    # Open-source LVLMs
    "llava-hf_llava-v1.6-mistral-7b-hf": "Llava-v1.6-Mistral-7B",
    "Qwen_Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL-7B",
    "mistralai_Pixtral-12B-2409": "Pixtral-12B",
    
    # CLIP-based models
    "peskavlp": "PeskaVLP",
    "raso": "RASO",
    
    # Task-specific models
    "cholenet": "CholeNet",
    "gonogonet": "GoNoGoNet",
    "gonogonet": "GoNoGoNet"
}

print("="*80)
print("MODEL DISCOVERY TEST")
print("="*80)

# Collect all unique models across all datasets
all_models = set()
dataset_models = {}

for dataset, results_dir in results_dirs.items():
    mode_dir = results_dir / 'zeroshot_combined'
    dataset_models[dataset] = []
    
    if mode_dir.exists():
        models = [d.name for d in mode_dir.iterdir() if d.is_dir()]
        dataset_models[dataset] = models
        all_models.update(models)
        
        print(f'\n{dataset}: {len(models)} models found')
        for model in sorted(models):
            display_name = MODEL_NAME_MAPPING.get(model, model)
            if display_name != model:
                print(f'  - {model} -> {display_name}')
            else:
                print(f'  - {model}')
    else:
        print(f'\n{dataset}: Directory not found at {mode_dir}')

print("\n" + "="*80)
print("UNIQUE MODELS ACROSS ALL DATASETS")
print("="*80)
print(f"Total unique models: {len(all_models)}")

# Categorize models
categories = {
    "Baselines": [],
    "Commercial LVLMs": [],
    "Open-Source LVLMs": [],
    "CLIP-based Models": [],
    "Task-Specific Models": []
}

for model in all_models:
    model_lower = model.lower()
    display_name = MODEL_NAME_MAPPING.get(model, model)
    
    # Baselines
    if "naive" in model_lower or "baseline" in model_lower:
        categories["Baselines"].append(display_name)
    # Commercial LVLMs
    elif any(x in model_lower for x in ["gpt", "gemini", "claude"]):
        categories["Commercial LVLMs"].append(display_name)
    # Open-source LVLMs
    elif any(x in model_lower for x in ["llava", "qwen", "pixtral"]):
        categories["Open-Source LVLMs"].append(display_name)
    # CLIP-based models
    elif any(x in model_lower for x in ["peskavlp", "raso"]):
        categories["CLIP-based Models"].append(display_name)
    # Task-specific models
    elif any(x in model_lower for x in ["cholenet", "gonogonet"]):
        categories["Task-Specific Models"].append(display_name)
    else:
        categories["Task-Specific Models"].append(display_name)

print("\nCategorized models:")
for category, models in categories.items():
    if models:
        print(f"\n{category}: {len(models)} models")
        for model in sorted(set(models)):  # Use set to remove duplicates
            print(f"  - {model}")

# Check for unmapped models
print("\n" + "="*80)
print("UNMAPPED MODEL NAMES")
print("="*80)
unmapped = [m for m in all_models if m not in MODEL_NAME_MAPPING]
if unmapped:
    print("The following model directory names are not in MODEL_NAME_MAPPING:")
    for model in sorted(unmapped):
        print(f"  - {model}")
else:
    print("All model directory names are mapped to display names.")

# Show model availability matrix
print("\n" + "="*80)
print("MODEL AVAILABILITY MATRIX")
print("="*80)
print(f"{'Model':<40} {'CholecSeg8k':<15} {'CholecOrgans':<15} {'CholecGoNoGo':<15}")
print("-" * 85)

for model in sorted(all_models):
    display_name = MODEL_NAME_MAPPING.get(model, model)
    row = f"{display_name:<40}"
    for dataset in ['CholecSeg8k', 'CholecOrgans', 'CholecGoNoGo']:
        if model in dataset_models[dataset]:
            row += f"{'✓':<15}"
        else:
            row += f"{'-':<15}"
    print(row)