#!/usr/bin/env python3
"""
Unified Bounding Box Evaluation Script for all datasets
Supports: CholecSeg8k, CholecOrgans, CholecGoNoGo
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
import time

# Add src to path
sys.path.insert(0, '/shared_data0/weiqiuy/llm_cholec_organ/src')

# Load API keys
api_keys_file = Path("/shared_data0/weiqiuy/llm_cholec_organ/API_KEYS2.json")
if api_keys_file.exists():
    with open(api_keys_file, "r") as f:
        api_keys = json.load(f)
    
    os.environ['OPENAI_API_KEY'] = api_keys.get('OPENAI_API_KEY', '')
    os.environ['ANTHROPIC_API_KEY'] = api_keys.get('ANTHROPIC_API_KEY', '')
    os.environ['GOOGLE_API_KEY'] = api_keys.get('GOOGLE_API_KEY', '')

from endopoint.eval.bbox_evaluator import BoundingBoxEvaluator


def load_dataset_adapter(dataset_name):
    """Load the appropriate dataset adapter based on dataset name."""
    
    if dataset_name == "cholecseg8k":
        from endopoint.datasets.cholecseg8k_local import CholecSeg8kLocalAdapter
        data_dir = "/shared_data0/weiqiuy/datasets/cholecseg8k"
        return CholecSeg8kLocalAdapter(data_dir=data_dir)
    
    elif dataset_name == "cholec_organs":
        from endopoint.datasets.cholec_organs import CholecOrgansAdapter
        # Uses default directory: /shared_data0/weiqiuy/real_drs/data/abdomen_exlib
        return CholecOrgansAdapter()
    
    elif dataset_name == "cholec_gonogo":
        from endopoint.datasets.cholec_gonogo import CholecGoNoGoAdapter
        # Uses default directory: /shared_data0/weiqiuy/real_drs/data/abdomen_exlib
        return CholecGoNoGoAdapter()
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: cholecseg8k, cholec_organs, cholec_gonogo")


def get_dataset_display_name(dataset_name):
    """Get display name for dataset."""
    display_names = {
        "cholecseg8k": "CHOLECSEG8K",
        "cholec_organs": "CHOLEC ORGANS",
        "cholec_gonogo": "CHOLEC GONOGO"
    }
    return display_names.get(dataset_name, dataset_name.upper())


def get_results_dir(dataset_name):
    """Get the results directory for a dataset."""
    if dataset_name == "cholecseg8k":
        return "bbox_cholecseg8k_local_quick"
    elif dataset_name == "cholec_organs":
        return "bbox_cholec_organs_quick"
    elif dataset_name == "cholec_gonogo":
        return "bbox_cholec_gonogo_quick"
    else:
        return f"bbox_{dataset_name}_quick"


def run_single_evaluation(dataset_name, model, num_samples=2, use_cache=True, 
                         use_persistent_dir=True, detection_mode='combined', 
                         use_fewshot=False):
    """Run evaluation for a single model-dataset combination.
    
    Returns:
        dict: Results dictionary with metrics
    """
    # Display configuration
    display_name = get_dataset_display_name(dataset_name)
    
    print("=" * 80)
    print(f"BOUNDING BOX EVALUATION - {display_name}")
    print("=" * 80)
    print(f"Dataset: {dataset_name}")
    print(f"Model: {model}")
    print(f"Samples: {num_samples}")
    print(f"Cache: {'enabled' if use_cache else 'disabled'}")
    print(f"Output: {'persistent' if use_persistent_dir else 'timestamped'}")
    print()
    
    # Load dataset adapter
    try:
        dataset_adapter = load_dataset_adapter(dataset_name)
        print(f"✓ Loaded {dataset_name} dataset adapter")
    except Exception as e:
        print(f"✗ Error loading dataset adapter: {e}")
        return None
    
    # Print dataset info
    if hasattr(dataset_adapter, 'total'):
        total_train = dataset_adapter.total("train") if hasattr(dataset_adapter, 'total') else 0
        total_val = dataset_adapter.total("validation") if hasattr(dataset_adapter, 'total') else 0
        total_test = dataset_adapter.total("test") if hasattr(dataset_adapter, 'total') else 0
        
        print(f"{dataset_name} dataset indexed:")
        print(f"  Total examples: {total_train + total_val + total_test}")
        print(f"  Train: {total_train} examples")
        print(f"  Validation: {total_val} examples")
        print(f"  Test: {total_test} examples")
    
    # Load test indices
    indices_dir = Path(f"/shared_data0/weiqiuy/llm_cholec_organ/data_info/{dataset_name}_balanced_200")
    indices_file = indices_dir / "balanced_test_indices_advanced_200.json"
    
    if indices_file.exists():
        with open(indices_file, 'r') as f:
            data = json.load(f)
            # Handle different formats: dict with 'indices' key or direct list
            if isinstance(data, dict) and 'indices' in data:
                test_indices = data['indices']
            else:
                test_indices = data
            test_indices = test_indices[:num_samples]
        print(f"Selected test indices: {test_indices[:5]}{'...' if len(test_indices) > 5 else ''}")
    else:
        print(f"Warning: No balanced test indices found, using first {num_samples} samples")
        test_indices = list(range(num_samples))
    
    # Get image dimensions based on dataset
    if dataset_name == "cholecseg8k":
        image_width, image_height = 854, 480
    elif dataset_name in ["cholec_organs", "cholec_gonogo"]:
        image_width, image_height = 640, 384
    else:
        image_width, image_height = 640, 480  # Default
    
    print(f"Image dimensions: {image_width}x{image_height}")
    
    # Determine output directory
    results_dir_name = get_results_dir(dataset_name)
    
    if use_persistent_dir:
        base_output_dir = Path(f"/shared_data0/weiqiuy/llm_cholec_organ/results/{results_dir_name}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_output_dir = Path(f"/shared_data0/weiqiuy/llm_cholec_organ/results/{results_dir_name}_{timestamp}")
    
    base_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {base_output_dir}")
    
    # Create evaluator
    evaluator = BoundingBoxEvaluator(
        models=[model],  # Pass as list
        dataset=None,  # Not using dataset object, just adapter
        dataset_adapter=dataset_adapter,
        output_dir=base_output_dir,  # Pass as Path object
        use_cache=use_cache,
        dataset_name=dataset_name  # Pass dataset name for RASO model selection
    )
    
    # Determine evaluation type
    if use_fewshot:
        eval_type = f"Few-shot {detection_mode.capitalize()}"
    else:
        eval_type = f"Zero-shot {detection_mode.capitalize()}"
    
    print("\n" + "=" * 80)
    print(f"RUNNING EVALUATION: {eval_type}")
    print("=" * 80)
    print(f"Detection mode: {detection_mode}")
    print(f"Few-shot: {'enabled' if use_fewshot else 'disabled'}")
    print()
    
    # Load few-shot data if needed
    fewshot_plan = None
    fewshot_examples = None
    
    if use_fewshot:
        # Load few-shot plan/examples based on detection mode
        fewshot_dir = Path(f"/shared_data0/weiqiuy/llm_cholec_organ/data_info/{dataset_name}_balanced_200")
        
        if detection_mode == 'separate':
            # Load few-shot plan for separate mode
            plan_file = fewshot_dir / "fewshot_plan_bbox_140.json"
            if plan_file.exists():
                with open(plan_file, 'r') as f:
                    fewshot_plan = json.load(f)
                print(f"Loaded few-shot plan from {plan_file}")
        else:  # combined mode
            # Load few-shot examples for combined mode
            examples_file = fewshot_dir / "fewshot_examples_bbox_10.json"
            if examples_file.exists():
                with open(examples_file, 'r') as f:
                    fewshot_examples = json.load(f)
                print(f"Loaded {len(fewshot_examples)} few-shot examples from {examples_file}")
    
    # Run evaluation
    start_time = time.time()
    
    try:
        results = evaluator.evaluate_model(
            model_name=model,
            test_indices=test_indices,
            detection_mode=detection_mode,
            prompt_type="standard",
            fewshot_plan=fewshot_plan if use_fewshot and detection_mode == 'separate' else None,
            fewshot_examples=fewshot_examples if use_fewshot and detection_mode == 'combined' else None,
            split='test'
        )
        
        elapsed_time = time.time() - start_time
        
        # Print results
        print("\n" + "=" * 80)
        print("EVALUATION COMPLETE")
        print("=" * 80)
        print(f"Model: {model}")
        print(f"Dataset: {dataset_name}")
        print(f"Evaluation: {eval_type}")
        print(f"Samples evaluated: {len(test_indices)}")
        print(f"Results saved to: {base_output_dir}")
        
        # Determine mode string for summary file
        mode_str = "fewshot" if use_fewshot else "zeroshot"
        summary_file = f"summary_{detection_mode}_{mode_str}.json"
        print(f"Summary file: {summary_file}")
        
        if results and 'metrics' in results:
            metrics = results['metrics']
            presence_acc = metrics.get('presence_accuracy', 0)
            mean_iou_bbox = metrics.get('mean_iou_bbox_to_bbox', 0)
            mean_iou_mask = metrics.get('mean_iou_bbox_to_mask', 0)
            
            print("\n📊 RESULTS:")
            print("-" * 60)
            print(f"Presence Accuracy: {presence_acc*100:.1f}%")
            print(f"Bbox-to-Bbox IoU:  {mean_iou_bbox:.3f}")
            print(f"Bbox-to-Mask IoU:  {mean_iou_mask:.3f}")
            print(f"Evaluation Time:   {elapsed_time:.1f}s")
            print("-" * 60)
        
        return results
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run bbox evaluation for specified dataset(s) and model(s)."""
    
    # Configuration from environment variables
    DATASET_NAME = os.environ.get('EVAL_DATASET', 'cholecseg8k')
    MODEL = os.environ.get('EVAL_MODEL', 'gpt-4.1')
    NUM_SAMPLES = int(os.environ.get('EVAL_NUM_SAMPLES', '2'))
    USE_CACHE = os.environ.get('EVAL_USE_CACHE', 'true').lower() != 'false'
    USE_PERSISTENT_DIR = os.environ.get('EVAL_PERSISTENT_DIR', 'true').lower() == 'true'
    
    # Evaluation configuration
    DETECTION_MODE = os.environ.get('EVAL_DETECTION_MODE', 'combined')
    USE_FEWSHOT = os.environ.get('EVAL_USE_FEWSHOT', 'false').lower() == 'true'
    
    # Check for batch mode (multiple models or datasets)
    BATCH_MODE = os.environ.get('EVAL_BATCH_MODE', 'false').lower() == 'true'
    
    if BATCH_MODE:
        # Get models and datasets from environment or use defaults
        BATCH_MODELS = os.environ.get('EVAL_BATCH_MODELS', 'cholenet,gonogo')
        BATCH_DATASETS = os.environ.get('EVAL_BATCH_DATASETS', 'cholecseg8k,cholec_organs,cholec_gonogo')
        
        # Parse comma-separated lists
        models = [m.strip() for m in BATCH_MODELS.split(',')]
        datasets = [d.strip() for d in BATCH_DATASETS.split(',')]
        
        print("=" * 80)
        print(f"BATCH EVALUATION MODE")
        print("=" * 80)
        print(f"Models: {', '.join(models)}")
        print(f"Datasets: {', '.join(datasets)}")
        print(f"Samples per evaluation: {NUM_SAMPLES}")
        print()
        
        all_results = {}
        
        # Build evaluations list dynamically
        evaluations = []
        for model in models:
            for dataset in datasets:
                # Create description based on model and dataset combination
                if model == "cholenet":
                    if dataset == "cholecseg8k":
                        desc = f"{model.upper()} on {dataset.upper()} (3/13 organs)"
                    elif dataset == "cholec_organs":
                        desc = f"{model.upper()} on {dataset.replace('_', ' ').title()} (native)"
                    elif dataset == "cholec_gonogo":
                        desc = f"{model.upper()} on {dataset.replace('_', ' ').title()} (cross-mapped)"
                    else:
                        desc = f"{model.upper()} on {dataset.upper()}"
                elif model == "gonogo":
                    if dataset == "cholecseg8k":
                        desc = f"{model.upper()} on {dataset.upper()} (no organs)"
                    elif dataset == "cholec_organs":
                        desc = f"{model.upper()} on {dataset.replace('_', ' ').title()} (cross-mapped)"
                    elif dataset == "cholec_gonogo":
                        desc = f"{model.upper()} on {dataset.replace('_', ' ').title()} (native)"
                    else:
                        desc = f"{model.upper()} on {dataset.upper()}"
                else:
                    # For other models (GPT, Claude, etc.)
                    desc = f"{model.upper()} on {dataset.replace('_', ' ').title()}"
                
                evaluations.append((dataset, model, desc))
        
        # Run each evaluation
        for dataset, model, description in evaluations:
            print(f"\n{'#'*80}")
            print(f"# {description}")
            print(f"{'#'*80}\n")
            
            results = run_single_evaluation(
                dataset_name=dataset,
                model=model,
                num_samples=NUM_SAMPLES,
                use_cache=USE_CACHE,
                use_persistent_dir=USE_PERSISTENT_DIR,
                detection_mode=DETECTION_MODE,
                use_fewshot=USE_FEWSHOT
            )
            
            all_results[f"{model}_{dataset}"] = results
        
        # Print summary table
        print("\n" + "=" * 100)
        print("BATCH EVALUATION SUMMARY")
        print("=" * 100)
        print(f"{'Model':<12} {'Dataset':<15} {'Presence Acc (%)':<18} {'BBox IoU':<12} {'Mask IoU':<12}")
        print("-" * 100)
        
        for (dataset, model, _) in evaluations:
            key = f"{model}_{dataset}"
            if key in all_results and all_results[key] and 'metrics' in all_results[key]:
                metrics = all_results[key]['metrics']
                presence_acc = metrics.get('presence_accuracy', 0)
                bbox_iou = metrics.get('mean_iou_bbox_to_bbox', 0)
                mask_iou = metrics.get('mean_iou_bbox_to_mask', 0)
                print(f"{model:<12} {dataset:<15} {presence_acc*100:>16.1f}% {bbox_iou:>11.3f} {mask_iou:>11.3f}")
            else:
                print(f"{model:<12} {dataset:<15} {'FAILED':>18} {'N/A':>11} {'N/A':>11}")
        
        print("-" * 100)
        print("\nCross-dataset mappings:")
        print("  • CholeNet → GoNoGo: Hepatocystic Triangle → Go Zone")
        print("  • GoNoGoNet → Organs: Go Zone → Hepatocystic Triangle, NoGo → Background")
        
    else:
        # Single evaluation mode (original behavior)
        results = run_single_evaluation(
            dataset_name=DATASET_NAME,
            model=MODEL,
            num_samples=NUM_SAMPLES,
            use_cache=USE_CACHE,
            use_persistent_dir=USE_PERSISTENT_DIR,
            detection_mode=DETECTION_MODE,
            use_fewshot=USE_FEWSHOT
        )


if __name__ == "__main__":
    sys.exit(main())
