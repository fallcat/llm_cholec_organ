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


def main():
    """Run bbox evaluation for specified dataset."""
    
    # Configuration from environment variables
    DATASET_NAME = os.environ.get('EVAL_DATASET', 'cholecseg8k')
    MODEL = os.environ.get('EVAL_MODEL', 'gpt-4.1')
    NUM_SAMPLES = int(os.environ.get('EVAL_NUM_SAMPLES', '2'))
    USE_CACHE = os.environ.get('EVAL_USE_CACHE', 'true').lower() != 'false'
    USE_PERSISTENT_DIR = os.environ.get('EVAL_PERSISTENT_DIR', 'true').lower() == 'true'
    
    # Evaluation configuration
    DETECTION_MODE = os.environ.get('EVAL_DETECTION_MODE', 'combined')
    USE_FEWSHOT = os.environ.get('EVAL_USE_FEWSHOT', 'false').lower() == 'true'
    
    # Display configuration
    display_name = get_dataset_display_name(DATASET_NAME)
    
    print("=" * 80)
    print(f"BOUNDING BOX EVALUATION - {display_name}")
    print("=" * 80)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Model: {MODEL}")
    print(f"Samples: {NUM_SAMPLES}")
    print(f"Cache: {'enabled' if USE_CACHE else 'disabled'}")
    print(f"Output: {'persistent' if USE_PERSISTENT_DIR else 'timestamped'}")
    print()
    
    # Load dataset adapter
    try:
        dataset_adapter = load_dataset_adapter(DATASET_NAME)
        print(f"✓ Loaded {DATASET_NAME} dataset adapter")
    except Exception as e:
        print(f"✗ Error loading dataset adapter: {e}")
        return 1
    
    # Print dataset info
    if hasattr(dataset_adapter, 'total'):
        total_train = dataset_adapter.total("train") if hasattr(dataset_adapter, 'total') else 0
        total_val = dataset_adapter.total("validation") if hasattr(dataset_adapter, 'total') else 0
        total_test = dataset_adapter.total("test") if hasattr(dataset_adapter, 'total') else 0
        
        print(f"{DATASET_NAME} dataset indexed:")
        print(f"  Total examples: {total_train + total_val + total_test}")
        print(f"  Train: {total_train} examples")
        print(f"  Validation: {total_val} examples")
        print(f"  Test: {total_test} examples")
    
    # Load test indices
    indices_dir = Path(f"/shared_data0/weiqiuy/llm_cholec_organ/data_info/{DATASET_NAME}_balanced_200")
    indices_file = indices_dir / "balanced_test_indices_advanced_200.json"
    
    if indices_file.exists():
        with open(indices_file, 'r') as f:
            data = json.load(f)
            # Handle different formats: dict with 'indices' key or direct list
            if isinstance(data, dict) and 'indices' in data:
                test_indices = data['indices']
            else:
                test_indices = data
            test_indices = test_indices[:NUM_SAMPLES]
        print(f"Selected test indices: {test_indices[:5]}{'...' if len(test_indices) > 5 else ''}")
    else:
        print(f"Warning: No balanced test indices found, using first {NUM_SAMPLES} samples")
        test_indices = list(range(NUM_SAMPLES))
    
    # Get image dimensions based on dataset
    if DATASET_NAME == "cholecseg8k":
        image_width, image_height = 854, 480
    elif DATASET_NAME in ["cholec_organs", "cholec_gonogo"]:
        image_width, image_height = 640, 384
    else:
        image_width, image_height = 640, 480  # Default
    
    print(f"Image dimensions: {image_width}x{image_height}")
    
    # Determine output directory
    results_dir_name = get_results_dir(DATASET_NAME)
    
    if USE_PERSISTENT_DIR:
        base_output_dir = Path(f"/shared_data0/weiqiuy/llm_cholec_organ/results/{results_dir_name}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_output_dir = Path(f"/shared_data0/weiqiuy/llm_cholec_organ/results/{results_dir_name}_{timestamp}")
    
    base_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {base_output_dir}")
    
    # Create evaluator
    evaluator = BoundingBoxEvaluator(
        models=[MODEL],  # Pass as list
        dataset=None,  # Not using dataset object, just adapter
        dataset_adapter=dataset_adapter,
        output_dir=base_output_dir,  # Pass as Path object
        use_cache=USE_CACHE
    )
    
    # Determine evaluation type
    if USE_FEWSHOT:
        eval_type = f"Few-shot {DETECTION_MODE.capitalize()}"
    else:
        eval_type = f"Zero-shot {DETECTION_MODE.capitalize()}"
    
    print("\n" + "=" * 80)
    print(f"RUNNING EVALUATION: {eval_type}")
    print("=" * 80)
    print(f"Detection mode: {DETECTION_MODE}")
    print(f"Few-shot: {'enabled' if USE_FEWSHOT else 'disabled'}")
    print()
    
    # Load few-shot data if needed
    fewshot_plan = None
    fewshot_examples = None
    
    if USE_FEWSHOT:
        # Load few-shot plan/examples based on detection mode
        fewshot_dir = Path(f"/shared_data0/weiqiuy/llm_cholec_organ/data_info/{DATASET_NAME}_balanced_200")
        
        if DETECTION_MODE == 'separate':
            # Load few-shot plan for separate mode
            plan_file = fewshot_dir / "fewshot_plan_bbox_140.json"
            if plan_file.exists():
                with open(plan_file, 'r') as f:
                    fewshot_plan = json.load(f)
                print(f"Loaded few-shot plan from {plan_file}")
        else:  # combined mode
            # Load few-shot examples for combined mode
            examples_file = fewshot_dir / "fewshot_plan_bbox_combined_greedy.json"
            if examples_file.exists():
                with open(examples_file, 'r') as f:
                    fewshot_data = json.load(f)
                    if 'selected_examples' in fewshot_data:
                        fewshot_examples = fewshot_data['selected_examples']
                    else:
                        fewshot_examples = fewshot_data
                print(f"Loaded {len(fewshot_examples) if fewshot_examples else 0} few-shot examples")
    
    # Run evaluation
    start_time = time.time()
    
    try:
        results = evaluator.evaluate_model(
            model_name=MODEL,
            test_indices=test_indices,
            detection_mode=DETECTION_MODE,
            use_fewshot=USE_FEWSHOT,
            fewshot_plan=fewshot_plan if USE_FEWSHOT and DETECTION_MODE == 'separate' else None,
            fewshot_examples=fewshot_examples if USE_FEWSHOT and DETECTION_MODE == 'combined' else None,
            split='test'
        )
        
        elapsed_time = time.time() - start_time
        
        # Print results
        print("\n" + "=" * 80)
        print("EVALUATION COMPLETE")
        print("=" * 80)
        print(f"Model: {MODEL}")
        print(f"Dataset: {DATASET_NAME}")
        print(f"Evaluation: {eval_type}")
        print(f"Samples evaluated: {len(test_indices)}")
        print(f"Results saved to: {base_output_dir}")
        
        # Determine mode string for summary file
        mode_str = "fewshot" if USE_FEWSHOT else "zeroshot"
        summary_file = f"summary_{DETECTION_MODE}_{mode_str}.json"
        print(f"Summary file: {summary_file}")
        
        if results and 'metrics' in results:
            metrics = results['metrics']
            presence_acc = metrics.get('presence_accuracy', 0)
            mean_iou_bbox = metrics.get('mean_iou_bbox_to_bbox', 0)
            mean_iou_mask = metrics.get('mean_iou_bbox_to_mask', 0)
            
            print("\n📊 RESULTS:")
            print("-" * 60)
            print(f"Presence Accuracy: {presence_acc:.1%}")
            print(f"Bbox-to-Bbox IoU:  {mean_iou_bbox:.3f}")
            print(f"Bbox-to-Mask IoU:  {mean_iou_mask:.3f}")
            print(f"Evaluation Time:   {elapsed_time:.1f}s")
            print("-" * 60)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())