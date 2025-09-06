#!/usr/bin/env python
"""
Cell Selection Evaluation Pipeline - FIXED VERSION with proper organ-specific few-shot examples
"""

import os
import sys
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import pandas as pd

# Add src to path
ROOT_DIR = ".."
sys.path.append(os.path.join(ROOT_DIR, 'src'))

# Load API keys
API_KEYS_FILE = os.path.join(ROOT_DIR, "API_KEYS2.json")
if os.path.exists(API_KEYS_FILE):
    with open(API_KEYS_FILE, 'r') as f:
        API_KEYS = json.load(f)
        os.environ["OPENAI_API_KEY"] = API_KEYS.get("OPENAI_API_KEY", "")
        os.environ["ANTHROPIC_API_KEY"] = API_KEYS.get("ANTHROPIC_API_KEY", "")
        os.environ["GOOGLE_API_KEY"] = API_KEYS.get("GOOGLE_API_KEY", "")

# Import modules
from endopoint.datasets.cholecseg8k import CholecSeg8kAdapter, ID2LABEL, LABEL2ID
from endopoint.models import create_model
from endopoint.prompts.builders import (
    build_cell_selection_system_prompt,
    build_cell_selection_system_prompt_strict,
    build_cell_selection_user_prompt
)
from endopoint.eval.parser import parse_cell_selection_json, validate_cell_selection_response
from endopoint.eval.cell_selection import (
    compute_cell_ground_truth,
    compute_cell_metrics,
    get_cell_labels,
    point_to_cell
)
from endopoint.eval.pointing import run_cell_selection_on_canvas
from endopoint.utils.io import ensure_dir
from endopoint.utils.logging import get_logger

logger = get_logger(__name__)

# Default models to evaluate
DEFAULT_MODELS = [
    "gpt-5-mini",
    "claude-sonnet-4-20250514", 
    "gemini-2.5-flash",
    "llava-hf/llava-v1.6-mistral-7b-hf",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "mistralai/Pixtral-12B-2409",
    "deepseek-ai/deepseek-vl2",
    "nvidia/Llama-3.2-11B-Vision-Surgical-CholecT50"
]


def prepare_organ_specific_few_shot_examples(
    dataset_adapter,
    fewshot_plan: Dict,
    dataset_split: str = "train",
    grid_size: int = 3,
    min_pixels: int = 50
) -> Dict[str, List[Tuple[torch.Tensor, Dict]]]:
    """Prepare organ-specific few-shot examples for cell selection.
    
    This follows the same pattern as the pointing evaluation where each organ
    gets its own positive and negative examples.
    
    Args:
        dataset_adapter: Dataset adapter
        fewshot_plan: Few-shot plan dictionary
        dataset_split: Dataset split to use
        grid_size: Grid size for cell selection
        min_pixels: Minimum pixels for presence
        
    Returns:
        Dictionary mapping organ names to list of (image, response) tuples
    """
    from datasets import load_dataset
    dataset = load_dataset("minwoosun/CholecSeg8k")
    
    few_shot_examples = {}
    
    # Handle different plan formats
    if 'plan' in fewshot_plan:
        # New format: {'plan': {'1': {...}, '2': {...}}}
        actual_plan = fewshot_plan['plan']
    else:
        # Old format: {'organ_name': {...}}
        actual_plan = fewshot_plan
    
    for organ_name, organ_id in LABEL2ID.items():
        if organ_id == 0:  # Skip background
            continue
            
        examples = []
        
        # Try to find the organ plan by ID or name
        organ_id_str = str(organ_id)
        
        # First try by ID (new format)
        if organ_id_str in actual_plan:
            organ_data = actual_plan[organ_id_str]
            # Extract examples from new format
            positives = organ_data.get('positives', [])
            pos_indices = [item['idx'] if isinstance(item, dict) else item for item in positives]
            
            negatives_easy = organ_data.get('negatives_easy', [])
            neg_easy_indices = [item['idx'] if isinstance(item, dict) else item for item in negatives_easy]
            
            negatives_hard = organ_data.get('negatives_hard', [])
            neg_hard_indices = [item['idx'] if isinstance(item, dict) else item for item in negatives_hard]
            
            organ_plan = {
                'positive': pos_indices,
                'negative_easy': neg_easy_indices,
                'negative_hard': neg_hard_indices
            }
        # Fallback to name (old format)
        elif organ_name in actual_plan:
            organ_plan = actual_plan[organ_name]
        else:
            organ_plan = {}
        
        # Add positive examples
        for idx in organ_plan.get("positive", []):
            example = dataset[dataset_split][idx]
            img_t, lab_t = dataset_adapter.example_to_tensors(example)
            
            # Compute ground truth for this specific organ
            organ_mask = (lab_t == organ_id).numpy().astype(np.uint8)
            gt_info = compute_cell_ground_truth(organ_mask, grid_size, min_pixels)
            
            if gt_info['present']:
                response = {
                    "name": organ_name,
                    "present": 1,
                    "cells": list(gt_info['cells'])[:1]  # Use dominant cell for example
                }
                examples.append((img_t, response))
        
        # Add negative examples (easy)
        for idx in organ_plan.get("negative_easy", []):
            example = dataset[dataset_split][idx]
            img_t, lab_t = dataset_adapter.example_to_tensors(example)
            
            response = {
                "name": organ_name,
                "present": 0,
                "cells": []
            }
            examples.append((img_t, response))
        
        # Add hard negative examples
        for idx in organ_plan.get("negative_hard", []):
            example = dataset[dataset_split][idx]
            img_t, lab_t = dataset_adapter.example_to_tensors(example)
            
            response = {
                "name": organ_name,
                "present": 0,
                "cells": []
            }
            examples.append((img_t, response))
        
        few_shot_examples[organ_name] = examples
    
    return few_shot_examples


def evaluate_cell_selection_fixed(
    dataset_adapter,
    model,
    indices: List[int],
    grid_size: int = 3,
    top_k: int = 1,
    prompt_style: str = "standard",
    few_shot_examples: Optional[Dict[str, List]] = None,  # Organ-specific examples
    use_cache: bool = True,
    results_dir: Optional[Path] = None,
    config_name: str = "default"
) -> Dict:
    """Evaluate cell selection with proper organ-specific few-shot examples.
    
    Args:
        dataset_adapter: Dataset adapter
        model: Model adapter
        indices: Sample indices to evaluate
        grid_size: Grid size
        top_k: Maximum cells to predict
        prompt_style: Prompt style
        few_shot_examples: Dictionary mapping organ names to their few-shot examples
        use_cache: Whether to use cache
        results_dir: Directory to save results
        
    Returns:
        Dictionary with evaluation results
    """
    from datasets import load_dataset
    dataset = load_dataset("minwoosun/CholecSeg8k")
    
    all_results = []
    
    # Process each sample
    for idx in tqdm(indices, desc=f"Evaluating {model.model_id}"):
        # Check if result already exists
        if results_dir:
            result_file = results_dir / f"sample_{idx:04d}.json"
            if result_file.exists():
                # Load existing result
                with open(result_file, 'r') as f:
                    sample_results = json.load(f)
                all_results.append(sample_results)
                continue  # Skip to next sample
        
        # Get example from dataset
        example = dataset['train'][idx]
        # Convert to tensors
        img_tensor, lab_tensor = dataset_adapter.example_to_tensors(example)
        
        # Get canvas dimensions
        H, W = lab_tensor.shape
        
        # Evaluate each organ
        sample_results = {'sample_idx': idx, 'organs': {}}
        
        for organ_id, organ_name in ID2LABEL.items():
            if organ_id == 0:  # Skip background
                continue
            
            # Get organ-specific few-shot examples
            organ_few_shot = None
            if few_shot_examples and organ_name in few_shot_examples:
                organ_few_shot = few_shot_examples[organ_name]
            
            # Run cell selection
            result = run_cell_selection_on_canvas(
                model=model,
                img_t=img_tensor,
                lab_t=lab_tensor,
                organ_name=organ_name,
                grid_size=grid_size,
                top_k=top_k,
                canvas_width=W,
                canvas_height=H,
                prompt_style=prompt_style,
                few_shot_examples=organ_few_shot,  # Use organ-specific examples
                min_pixels=50
            )
            
            # Add organ_id to result
            result['organ_id'] = organ_id
            sample_results['organs'][organ_name] = result
        
        all_results.append(sample_results)
        
        # Save intermediate results if directory provided
        if results_dir:
            result_file = results_dir / f"sample_{idx:04d}.json"
            with open(result_file, 'w') as f:
                json.dump(sample_results, f, indent=2)
    
    return {'results': all_results, 'grid_size': grid_size, 'top_k': top_k}


def main():
    """Main evaluation function."""
    
    # Configuration from environment variables
    NUM_SAMPLES = os.environ.get('EVAL_NUM_SAMPLES', None)
    if NUM_SAMPLES:
        NUM_SAMPLES = int(NUM_SAMPLES)
    
    MODELS = os.environ.get('EVAL_MODELS', None)
    if MODELS:
        MODELS = MODELS.split(',')
    else:
        MODELS = DEFAULT_MODELS
    
    USE_CACHE = os.environ.get('EVAL_USE_CACHE', 'true').lower() != 'false'
    USE_PERSISTENT = os.environ.get('EVAL_PERSISTENT_DIR', 'false').lower() == 'true'
    SKIP_ZERO_SHOT = os.environ.get('EVAL_SKIP_ZERO_SHOT', 'false').lower() == 'true'
    SKIP_FEW_SHOT = os.environ.get('EVAL_SKIP_FEW_SHOT', 'false').lower() == 'true'
    GRID_SIZE = int(os.environ.get('EVAL_GRID_SIZE', '3'))
    TOP_K = int(os.environ.get('EVAL_TOP_K', '1'))
    
    print("\n" + "="*60)
    print("Cell Selection Evaluation (FIXED with organ-specific few-shot)")
    print("="*60)
    print(f"Grid: {GRID_SIZE}×{GRID_SIZE}, Top-K: {TOP_K}")
    print(f"Models: {', '.join(MODELS)}")
    print(f"Samples: {NUM_SAMPLES if NUM_SAMPLES else 'all'}")
    
    # Load dataset adapter
    dataset_adapter = CholecSeg8kAdapter()
    
    # Load test indices
    data_dir = Path(ROOT_DIR) / "data_info" / "cholecseg8k"
    test_indices_file = data_dir / "balanced_indices_train_100.json"
    
    with open(test_indices_file, 'r') as f:
        indices_data = json.load(f)
        # Handle both dict format {"indices": [...]} and list format [...]
        if isinstance(indices_data, dict):
            test_indices = indices_data.get('indices', indices_data)
        else:
            test_indices = indices_data
    
    if NUM_SAMPLES and NUM_SAMPLES < len(test_indices):
        import numpy as np
        selected_idx = np.linspace(0, len(test_indices) - 1, NUM_SAMPLES, dtype=int)
        test_indices = [test_indices[i] for i in selected_idx]
    
    print(f"Using {len(test_indices)} test samples")
    
    # Load few-shot plans
    fewshot_plans = {}
    if not SKIP_FEW_SHOT:
        fewshot_plan_files = {
            "standard": data_dir / "fewshot_plan_train_pos1_neg1_seed43_excl100.json",
            "hard_negatives": data_dir / "fewshot_plan_train_pos1_neg1_nearmiss1_seed45_excl100.json",
        }
        
        for plan_name, plan_file in fewshot_plan_files.items():
            if plan_file.exists():
                with open(plan_file, 'r') as f:
                    fewshot_plans[plan_name] = json.load(f)
                print(f"Loaded few-shot plan: {plan_name}")
    
    # Create output directory
    if USE_PERSISTENT:
        output_dir = Path(ROOT_DIR) / "results" / f"cell_selection_G{GRID_SIZE}_K{TOP_K}_fixed"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(ROOT_DIR) / "results" / f"cell_selection_G{GRID_SIZE}_K{TOP_K}_fixed_{timestamp}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate each model
    all_results = {}
    
    for model_name in MODELS:
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")
        
        # Load model
        model = create_model(model_name, use_cache=USE_CACHE)
        model_results = {}
        
        # Zero-shot evaluation
        if not SKIP_ZERO_SHOT:
            print("\n📊 Zero-shot evaluation...")
            eval_dir = output_dir / "zero_shot" / model_name.replace('/', '_') / "cholecseg8k_cell_selection"
            eval_dir.mkdir(parents=True, exist_ok=True)
            
            results = evaluate_cell_selection_fixed(
                dataset_adapter,
                model,
                test_indices,
                grid_size=GRID_SIZE,
                top_k=TOP_K,
                prompt_style="standard",
                few_shot_examples=None,  # No few-shot
                use_cache=USE_CACHE,
                results_dir=eval_dir
            )
            model_results['zero_shot'] = results
        
        # Few-shot evaluation
        if not SKIP_FEW_SHOT:
            for plan_name, fewshot_plan in fewshot_plans.items():
                print(f"\n📊 Few-shot ({plan_name}) evaluation...")
                
                # Prepare organ-specific few-shot examples
                organ_few_shot_examples = prepare_organ_specific_few_shot_examples(
                    dataset_adapter,
                    fewshot_plan,
                    dataset_split="train",
                    grid_size=GRID_SIZE,
                    min_pixels=50
                )
                
                eval_dir = output_dir / f"fewshot_{plan_name}" / model_name.replace('/', '_') / "cholecseg8k_cell_selection"
                eval_dir.mkdir(parents=True, exist_ok=True)
                
                results = evaluate_cell_selection_fixed(
                    dataset_adapter,
                    model,
                    test_indices,
                    grid_size=GRID_SIZE,
                    top_k=TOP_K,
                    prompt_style="standard",
                    few_shot_examples=organ_few_shot_examples,  # Organ-specific examples
                    use_cache=USE_CACHE,
                    results_dir=eval_dir
                )
                model_results[f'fewshot_{plan_name}'] = results
        
        all_results[model_name] = model_results
    
    print("\n✨ Evaluation complete!")
    print(f"Results saved to: {output_dir}")
    
    return all_results


if __name__ == "__main__":
    main()