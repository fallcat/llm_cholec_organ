#!/usr/bin/env python
"""
Cell Selection Evaluation Pipeline - Using Grid-based Localization
This version uses cell selection instead of exact (x,y) pointing.

USAGE:
    # Quick test with 5 samples
    EVAL_QUICK_TEST=true python3 eval_cell_selection_original_size.py
    
    # Full evaluation with all test samples
    python3 eval_cell_selection_original_size.py
    
    # Evaluate specific grid size and top-k
    EVAL_GRID_SIZE=3 EVAL_TOP_K=1 python3 eval_cell_selection_original_size.py
    
    # Test with 4x4 grid and top-3 cells
    EVAL_GRID_SIZE=4 EVAL_TOP_K=3 python3 eval_cell_selection_original_size.py
    
    # Use specific models
    EVAL_MODELS='gpt-5-mini,claude-4-sonnet' python3 eval_cell_selection_original_size.py
    
    # Combine options
    EVAL_NUM_SAMPLES=10 EVAL_GRID_SIZE=3 EVAL_TOP_K=1 python3 eval_cell_selection_original_size.py
    
    # Use persistent directory to resume evaluation (skip already evaluated)
    EVAL_PERSISTENT_DIR=true EVAL_USE_CACHE=false python3 eval_cell_selection_original_size.py
    
    # Resume evaluation for specific models only
    EVAL_PERSISTENT_DIR=true EVAL_MODELS='gpt-5-mini' python3 eval_cell_selection_original_size.py

ENVIRONMENT VARIABLES:
    EVAL_NUM_SAMPLES    - Number of samples to evaluate (default: all)
    EVAL_MODELS         - Comma-separated list of models (default: all 7 models)
    EVAL_USE_CACHE      - Whether to use cached API responses (default: true)
    EVAL_PERSISTENT_DIR - Use persistent dir to skip evaluated samples (default: false)
    EVAL_QUICK_TEST     - Quick test mode with 5 samples (default: false)
    EVAL_SKIP_ZERO_SHOT - Skip zero-shot evaluation (default: false)
    EVAL_SKIP_FEW_SHOT  - Skip few-shot evaluation (default: false)
    EVAL_GRID_SIZE      - Grid size (3 or 4, default: 3)
    EVAL_TOP_K          - Max cells to predict (1 or 3, default: 1)

OUTPUT:
    Results are saved to: results/cell_selection_G{grid}_K{topk}_YYYYMMDD_HHMMSS/
    - zero_shot/MODEL/cholecseg8k_cell_selection/*.json
    - fewshot_standard/MODEL/cholecseg8k_cell_selection/*.json
    - fewshot_hard_negatives/MODEL/cholecseg8k_cell_selection/*.json
    - metrics_comparison.txt
    - metrics_comparison.json

METRICS:
    - Presence Accuracy: How well organs are detected
    - Cell@K: Cell selection accuracy when organ is detected
    - Cell Precision/Recall/F1: Fine-grained cell-level metrics
    - Gated Accuracy: Combined detection + cell selection performance
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
from endopoint.datasets.cholecseg8k import CholecSeg8kAdapter, ID2LABEL
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
from endopoint.utils.io import ensure_dir
from endopoint.utils.logging import get_logger

logger = get_logger(__name__)

# Default models to evaluate
DEFAULT_MODELS = [
    "gpt-5-mini",
    "claude-4-sonnet", 
    "gemini-2.5-flash",
    "llava-hf/llava-v1.6-mistral-7b-hf",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "mistralai/Pixtral-12B-2409",
    "deepseek-ai/deepseek-vl2"
]


def run_cell_selection_for_organ(
    model,
    img_tensor: torch.Tensor,
    lab_tensor: torch.Tensor,
    organ_name: str,
    organ_id: int,
    grid_size: int = 3,
    top_k: int = 1,
    prompt_style: str = "standard",
    few_shot_examples: Optional[List] = None,
    min_pixels: int = 50,
    config_name: str = "default"
) -> Dict:
    """Run cell selection for a single organ.
    
    Uses the run_cell_selection_on_canvas function from pointing.py
    which properly handles cache keys through the prompt content.
    
    Args:
        model: Model adapter
        img_tensor: Image tensor [3,H,W]
        lab_tensor: Label tensor [H,W]
        organ_name: Organ name
        organ_id: Organ ID
        grid_size: Grid size (3 or 4)
        top_k: Maximum cells to predict
        prompt_style: "standard" or "strict"
        few_shot_examples: Optional few-shot examples
        min_pixels: Minimum pixels for presence
        config_name: Configuration name (not needed for caching anymore)
        
    Returns:
        Result dictionary with predictions and ground truth
    """
    from endopoint.eval.pointing import run_cell_selection_on_canvas
    
    # Convert few-shot examples to the format expected by run_cell_selection_on_canvas
    formatted_examples = None
    if few_shot_examples:
        formatted_examples = []
        for ex in few_shot_examples:
            # Each example is a tuple of (image_tensor, response_dict)
            formatted_examples.append((ex['image'], ex['response']))
    
    # Get canvas dimensions from tensor shape
    H, W = lab_tensor.shape
    
    # Call the function from pointing.py
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
        few_shot_examples=formatted_examples,
        min_pixels=min_pixels
    )
    
    # The result from run_cell_selection_on_canvas already includes all we need
    # Just add the organ_id and rename some fields for consistency
    return {
        'organ_name': result['organ'],
        'organ_id': organ_id,
        'pred_present': result['present'],
        'pred_cells': result['cells'],
        'gt_present': result['gt_present'],
        'gt_cells': result['gt_cells'],
        'gt_dominant_cell': result.get('gt_dominant_cell'),
        'gt_pixel_counts': {},  # Not included in the base function
        'metrics': result['metrics'],
        'raw_response': result['raw'],
        'grid_size': result['grid_size'],
        'top_k': result['top_k']
    }


def evaluate_cell_selection(
    dataset_adapter,
    model,
    indices: List[int],
    grid_size: int = 3,
    top_k: int = 1,
    prompt_style: str = "standard",
    few_shot_plan: Optional[Dict] = None,
    use_cache: bool = True,
    results_dir: Optional[Path] = None,
    config_name: str = "default"
) -> Dict:
    """Evaluate cell selection on a dataset.
    
    Args:
        dataset_adapter: Dataset adapter
        model: Model adapter
        indices: Sample indices to evaluate
        grid_size: Grid size
        top_k: Maximum cells to predict
        prompt_style: Prompt style
        few_shot_plan: Optional few-shot plan
        use_cache: Whether to use cache
        results_dir: Directory to save results
        
    Returns:
        Dictionary with evaluation results
    """
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
        example = dataset_adapter.get_example('train', idx)
        # Convert to tensors
        img_tensor, lab_tensor = dataset_adapter.example_to_tensors(example)
        
        # Get few-shot examples for this sample if plan provided
        few_shot_base_examples = None
        if few_shot_plan and str(idx) in few_shot_plan:
            # Load few-shot example images and labels
            few_shot_base_examples = []
            for ex_idx in few_shot_plan[str(idx)].get('positive', []):
                ex_example = dataset_adapter.get_example('train', ex_idx)
                ex_img_tensor, ex_lab_tensor = dataset_adapter.example_to_tensors(ex_example)
                few_shot_base_examples.append({
                    'image': ex_img_tensor,
                    'label': ex_lab_tensor,
                    'is_positive': True
                })
            # Also load negative examples if they exist
            for ex_idx in few_shot_plan[str(idx)].get('negative', []):
                ex_example = dataset_adapter.get_example('train', ex_idx)
                ex_img_tensor, ex_lab_tensor = dataset_adapter.example_to_tensors(ex_example)
                few_shot_base_examples.append({
                    'image': ex_img_tensor,
                    'label': ex_lab_tensor,
                    'is_positive': False
                })
        
        # Evaluate each organ
        sample_results = {'sample_idx': idx, 'organs': {}}
        
        for organ_id, organ_name in ID2LABEL.items():
            if organ_id == 0:  # Skip background
                continue
            
            # Create organ-specific few-shot examples if available
            organ_few_shot = None
            if few_shot_base_examples:
                organ_few_shot = []
                for ex in few_shot_base_examples:
                    # Compute actual ground truth for this organ in the example
                    ex_organ_mask = (ex['label'] == organ_id).numpy().astype(np.uint8)
                    ex_gt_info = compute_cell_ground_truth(ex_organ_mask, grid_size, min_pixels=50)
                    
                    # Create proper response based on ground truth
                    organ_few_shot.append({
                        'image': ex['image'],
                        'response': {
                            'name': organ_name,
                            'present': 1 if ex_gt_info['present'] else 0,
                            'cells': list(ex_gt_info['cells']) if ex_gt_info['present'] else []
                        }
                    })
            
            result = run_cell_selection_for_organ(
                model,
                img_tensor,
                lab_tensor,
                organ_name,
                organ_id,
                grid_size,
                top_k,
                prompt_style,
                organ_few_shot,
                config_name=config_name
            )
            
            sample_results['organs'][organ_name] = result
        
        all_results.append(sample_results)
        
        # Save intermediate results if directory provided
        if results_dir:
            result_file = results_dir / f"sample_{idx:04d}.json"
            with open(result_file, 'w') as f:
                json.dump(sample_results, f, indent=2)
    
    return {'results': all_results, 'grid_size': grid_size, 'top_k': top_k}


def compute_aggregate_metrics(results: List[Dict]) -> Dict:
    """Compute aggregate metrics across all samples.
    
    Args:
        results: List of sample results
        
    Returns:
        Dictionary with aggregate metrics
    """
    # Collect metrics per organ
    organ_metrics = {}
    
    for sample_result in results:
        for organ_name, organ_result in sample_result['organs'].items():
            if organ_name not in organ_metrics:
                organ_metrics[organ_name] = {
                    'presence_correct': [],
                    'cell_hits': [],
                    'cell_precisions': [],
                    'cell_recalls': [],
                    'cell_f1s': [],
                    'false_positives': []
                }
            
            metrics = organ_result['metrics']
            
            # Presence accuracy
            presence_correct = int(organ_result['pred_present'] == organ_result['gt_present'])
            organ_metrics[organ_name]['presence_correct'].append(presence_correct)
            
            # Cell metrics (only if organ present in GT)
            if organ_result['gt_present']:
                organ_metrics[organ_name]['cell_hits'].append(metrics['cell_hit'])
                organ_metrics[organ_name]['cell_precisions'].append(metrics['cell_precision'])
                organ_metrics[organ_name]['cell_recalls'].append(metrics['cell_recall'])
                organ_metrics[organ_name]['cell_f1s'].append(metrics['cell_f1'])
            
            # False positives
            if not organ_result['gt_present']:
                organ_metrics[organ_name]['false_positives'].append(metrics['false_positive_cells'])
    
    # Compute averages
    aggregate = {}
    for organ_name, metrics in organ_metrics.items():
        aggregate[organ_name] = {
            'presence_acc': np.mean(metrics['presence_correct']) * 100,
            'cell_hit_rate': np.mean(metrics['cell_hits']) * 100 if metrics['cell_hits'] else 0,
            'cell_precision': np.mean(metrics['cell_precisions']) * 100 if metrics['cell_precisions'] else 0,
            'cell_recall': np.mean(metrics['cell_recalls']) * 100 if metrics['cell_recalls'] else 0,
            'cell_f1': np.mean(metrics['cell_f1s']) * 100 if metrics['cell_f1s'] else 0,
            'avg_false_positive_cells': np.mean(metrics['false_positives']) if metrics['false_positives'] else 0,
            'n_samples': len(metrics['presence_correct'])
        }
    
    # Compute macro averages
    macro_metrics = {
        'macro_presence_acc': np.mean([m['presence_acc'] for m in aggregate.values()]),
        'macro_cell_hit_rate': np.mean([m['cell_hit_rate'] for m in aggregate.values()]),
        'macro_cell_precision': np.mean([m['cell_precision'] for m in aggregate.values()]),
        'macro_cell_recall': np.mean([m['cell_recall'] for m in aggregate.values()]),
        'macro_cell_f1': np.mean([m['cell_f1'] for m in aggregate.values()])
    }
    
    return {'per_organ': aggregate, 'macro': macro_metrics}


def main():
    """Main evaluation pipeline."""
    # Parse environment variables
    quick_test = os.getenv('EVAL_QUICK_TEST', 'false').lower() == 'true'
    num_samples = int(os.getenv('EVAL_NUM_SAMPLES', '0'))
    grid_size = int(os.getenv('EVAL_GRID_SIZE', '3'))
    top_k = int(os.getenv('EVAL_TOP_K', '1'))
    use_cache = os.getenv('EVAL_USE_CACHE', 'true').lower() == 'true'
    skip_zero_shot = os.getenv('EVAL_SKIP_ZERO_SHOT', 'false').lower() == 'true'
    skip_few_shot = os.getenv('EVAL_SKIP_FEW_SHOT', 'false').lower() == 'true'
    use_persistent_dir = os.getenv('EVAL_PERSISTENT_DIR', 'false').lower() == 'true'
    
    # Parse models
    models_str = os.getenv('EVAL_MODELS', '')
    if models_str:
        model_ids = [m.strip() for m in models_str.split(',')]
    else:
        model_ids = DEFAULT_MODELS
    
    # Validate parameters
    assert grid_size in [3, 4], f"Grid size must be 3 or 4, got {grid_size}"
    assert top_k in [1, 3], f"Top-K must be 1 or 3, got {top_k}"
    
    # Quick test override
    if quick_test:
        num_samples = 5
        logger.info("Quick test mode: evaluating 5 samples")
    
    # Create results directory
    if use_persistent_dir:
        # Use a persistent directory without timestamp
        results_root = Path(f"../results/cell_selection_G{grid_size}_K{top_k}")
        logger.info(f"Using persistent directory: {results_root}")
    else:
        # Use timestamped directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_root = Path(f"../results/cell_selection_G{grid_size}_K{top_k}_{timestamp}")
    results_root.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    logger.info("Loading CholecSeg8k dataset...")
    dataset = CholecSeg8kAdapter()
    
    # Load balanced indices
    indices_file = Path("../data_info/cholecseg8k/balanced_indices_train_100_cap70_seed7.json")
    if indices_file.exists():
        with open(indices_file, 'r') as f:
            indices = json.load(f)['indices']
        logger.info(f"Loaded {len(indices)} balanced indices")
    else:
        # Fallback to first N samples
        indices = list(range(min(100, len(dataset))))
        logger.warning(f"Using first {len(indices)} samples (balanced indices not found)")
    
    # Limit samples if specified
    if num_samples > 0:
        indices = indices[:num_samples]
    logger.info(f"Evaluating {len(indices)} samples")
    
    # Load few-shot plans
    few_shot_plans = {}
    if not skip_few_shot:
        # Standard few-shot
        standard_plan_file = Path("../data_info/cholecseg8k/fewshot_plan_train_pos1_neg1_mp50_seed123_excl100.json")
        if standard_plan_file.exists():
            with open(standard_plan_file, 'r') as f:
                few_shot_plans['standard'] = json.load(f)
        
        # Hard negatives few-shot
        hard_neg_plan_file = Path("../data_info/cholecseg8k/fewshot_plan_train_pos1_neg1_hneg1_mp50_seed123_excl100.json")
        if hard_neg_plan_file.exists():
            with open(hard_neg_plan_file, 'r') as f:
                few_shot_plans['hard_negatives'] = json.load(f)
    
    # Evaluation configurations
    configs = []
    if not skip_zero_shot:
        configs.append(('zero_shot', None))
    if not skip_few_shot and 'standard' in few_shot_plans:
        configs.append(('fewshot_standard', few_shot_plans['standard']))
    if not skip_few_shot and 'hard_negatives' in few_shot_plans:
        configs.append(('fewshot_hard_negatives', few_shot_plans['hard_negatives']))
    
    # Store all results for comparison
    all_evaluation_results = {}
    
    # Evaluate each model and configuration
    for model_id in model_ids:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating model: {model_id}")
        logger.info(f"{'='*60}")
        
        try:
            # Create model
            model = create_model(model_id, use_cache=use_cache)
            
            for config_name, few_shot_plan in configs:
                logger.info(f"\nConfiguration: {config_name}")
                
                # Create results directory for this config
                config_dir = results_root / config_name / model_id.replace('/', '_') / 'cholecseg8k_cell_selection'
                config_dir.mkdir(parents=True, exist_ok=True)
                
                # Run evaluation
                results = evaluate_cell_selection(
                    dataset,
                    model,
                    indices,
                    grid_size=grid_size,
                    top_k=top_k,
                    prompt_style="strict" if "gpt" in model_id.lower() else "standard",
                    few_shot_plan=few_shot_plan,
                    use_cache=use_cache,
                    results_dir=config_dir,
                    config_name=config_name
                )
                
                # Compute aggregate metrics
                metrics = compute_aggregate_metrics(results['results'])
                
                # Store results
                key = f"{model_id}_{config_name}"
                all_evaluation_results[key] = {
                    'model': model_id,
                    'config': config_name,
                    'metrics': metrics,
                    'grid_size': grid_size,
                    'top_k': top_k,
                    'n_samples': len(indices)
                }
                
                # Print summary
                logger.info(f"\nResults for {model_id} - {config_name}:")
                logger.info(f"Grid: {grid_size}×{grid_size}, Top-K: {top_k}")
                logger.info(f"Macro Presence Acc: {metrics['macro']['macro_presence_acc']:.1f}%")
                logger.info(f"Macro Cell Hit Rate: {metrics['macro']['macro_cell_hit_rate']:.1f}%")
                logger.info(f"Macro Cell F1: {metrics['macro']['macro_cell_f1']:.1f}%")
                
        except Exception as e:
            logger.error(f"Error evaluating {model_id}: {e}")
            continue
    
    # Save comparison results
    comparison_file = results_root / "metrics_comparison.json"
    with open(comparison_file, 'w') as f:
        json.dump(all_evaluation_results, f, indent=2)
    
    # Generate comparison table
    comparison_text = generate_comparison_table(all_evaluation_results)
    comparison_text_file = results_root / "metrics_comparison.txt"
    with open(comparison_text_file, 'w') as f:
        f.write(comparison_text)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluation complete!")
    logger.info(f"Results saved to: {results_root}")
    logger.info(f"{'='*60}")
    
    # Print final comparison
    print("\n" + comparison_text)


def generate_comparison_table(results: Dict) -> str:
    """Generate a formatted comparison table.
    
    Args:
        results: Dictionary of evaluation results
        
    Returns:
        Formatted table string
    """
    lines = []
    lines.append("="*80)
    lines.append("CELL SELECTION EVALUATION RESULTS")
    lines.append("="*80)
    
    if not results:
        lines.append("No results to display")
        return "\n".join(lines)
    
    # Get grid and top-k from first result
    first_result = next(iter(results.values()))
    grid_size = first_result['grid_size']
    top_k = first_result['top_k']
    
    lines.append(f"Grid Size: {grid_size}×{grid_size}")
    lines.append(f"Top-K: {top_k}")
    lines.append(f"Samples: {first_result['n_samples']}")
    lines.append("")
    
    # Create table
    headers = ["Model", "Config", "Presence Acc", f"Cell@{top_k}", "Cell Prec", "Cell Rec", "Cell F1"]
    col_widths = [20, 20, 12, 10, 10, 10, 10]
    
    # Header row
    header_row = "|"
    for header, width in zip(headers, col_widths):
        header_row += f" {header:^{width-2}} |"
    lines.append(header_row)
    
    # Separator
    sep_row = "|"
    for width in col_widths:
        sep_row += "-" * width + "|"
    lines.append(sep_row)
    
    # Data rows
    for key, result in sorted(results.items()):
        model_name = result['model'].split('/')[-1][:18]  # Truncate long names
        config_name = result['config'].replace('_', ' ')[:18]
        metrics = result['metrics']['macro']
        
        row = "|"
        row += f" {model_name:<{col_widths[0]-2}} |"
        row += f" {config_name:<{col_widths[1]-2}} |"
        row += f" {metrics['macro_presence_acc']:>{col_widths[2]-3}.1f}% |"
        row += f" {metrics['macro_cell_hit_rate']:>{col_widths[3]-3}.1f}% |"
        row += f" {metrics['macro_cell_precision']:>{col_widths[4]-3}.1f}% |"
        row += f" {metrics['macro_cell_recall']:>{col_widths[5]-3}.1f}% |"
        row += f" {metrics['macro_cell_f1']:>{col_widths[6]-3}.1f}% |"
        
        lines.append(row)
    
    lines.append("="*80)
    
    return "\n".join(lines)


if __name__ == "__main__":
    main()