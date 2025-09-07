#!/usr/bin/env python3
"""
Quick Bounding Box Evaluation Script - Cholec GoNoGo Dataset
This is a variant of eval_bbox_quick_test.py for the cholec_gonogo dataset.
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

from endopoint.datasets.cholec_gonogo import CholecGoNoGoAdapter
from endopoint.eval.bbox_evaluator import BoundingBoxEvaluator


def main():
    """Run bbox evaluation for cholec_gonogo dataset."""
    
    # Configuration from environment variables
    MODEL = os.environ.get('EVAL_MODEL', 'gpt-4.1')
    NUM_SAMPLES = int(os.environ.get('EVAL_NUM_SAMPLES', '2'))
    USE_CACHE = os.environ.get('EVAL_USE_CACHE', 'true').lower() != 'false'
    USE_PERSISTENT_DIR = os.environ.get('EVAL_PERSISTENT_DIR', 'true').lower() == 'true'
    
    # Dataset configuration
    DATASET_NAME = "cholec_gonogo"
    
    print("=" * 80)
    print("BOUNDING BOX QUICK TEST - CHOLEC GONOGO")
    print("=" * 80)
    print(f"Model: {MODEL}")
    print(f"Samples: {NUM_SAMPLES}")
    print(f"Cache: {'enabled' if USE_CACHE else 'disabled'}")
    print(f"Output: {'persistent' if USE_PERSISTENT_DIR else 'timestamped'}")
    print()
    
    # Load dataset - use default data directory
    dataset_adapter = CholecGoNoGoAdapter()  # Uses default: /shared_data0/weiqiuy/real_drs/data/abdomen_exlib
    
    # Load test indices
    indices_dir = Path(f"/shared_data0/weiqiuy/llm_cholec_organ/data_info/{DATASET_NAME}_balanced_200")
    with open(indices_dir / "balanced_test_indices_advanced_200.json", 'r') as f:
        test_data = json.load(f)
    
    # Select subset of indices
    test_indices = test_data['indices'][:NUM_SAMPLES]
    print(f"Selected test indices: {test_indices}")
    
    # Load few-shot plan (determine based on detection mode)
    DETECTION_MODE = os.environ.get('EVAL_DETECTION_MODE', 'combined')  # combined or separate
    USE_FEWSHOT = os.environ.get('EVAL_USE_FEWSHOT', 'false').lower() == 'true'
    
    fewshot_plan = None
    fewshot_examples = None
    
    if USE_FEWSHOT:
        if DETECTION_MODE == 'combined':
            # Load combined few-shot examples
            combined_file = indices_dir / "fewshot_plan_bbox_combined_greedy.json"
            if combined_file.exists():
                with open(combined_file, 'r') as f:
                    combined_plan = json.load(f)
                fewshot_examples = combined_plan.get('examples', [])
                print(f"Loaded {len(fewshot_examples)} combined few-shot examples")
            else:
                print(f"Warning: Combined few-shot file not found, using zero-shot")
                USE_FEWSHOT = False
        else:
            # Load separate few-shot plan
            separate_file = indices_dir / "fewshot_plan_bbox_200.json"
            if separate_file.exists():
                with open(separate_file, 'r') as f:
                    fewshot_plan = json.load(f)
                print(f"Loaded separate few-shot plan")
            else:
                print(f"Warning: Separate few-shot file not found, using zero-shot")
                USE_FEWSHOT = False
    
    # Get image dimensions
    example = dataset_adapter.get_example('train', 0)
    img_width, img_height = example['image'].size
    print(f"Image dimensions: {img_width}x{img_height}")
    print()
    
    # Create output directory
    if USE_PERSISTENT_DIR:
        output_dir = Path(f"/shared_data0/weiqiuy/llm_cholec_organ/results/bbox_{DATASET_NAME}_quick")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"/shared_data0/weiqiuy/llm_cholec_organ/results/bbox_{DATASET_NAME}_{timestamp}")
    
    # Initialize evaluator
    evaluator = BoundingBoxEvaluator(
        models=[MODEL],  # Single model
        dataset=None,
        dataset_adapter=dataset_adapter,
        canvas_width=img_width,
        canvas_height=img_height,
        output_dir=output_dir,
        use_cache=USE_CACHE,
        min_pixels=50
    )
    
    print(f"Output directory: {evaluator.output_dir}")
    print()
    
    # Evaluation mode already determined above
    
    # Determine evaluation name
    eval_name = f"{'Few-shot' if USE_FEWSHOT else 'Zero-shot'} {DETECTION_MODE.capitalize()}"
    
    print("=" * 80)
    print(f"RUNNING EVALUATION: {eval_name}")
    print("=" * 80)
    print(f"Detection mode: {DETECTION_MODE}")
    print(f"Few-shot: {'enabled' if USE_FEWSHOT else 'disabled'}")
    print()
    
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
        
        elapsed = time.time() - start_time
        
        # Store results
        results_summary = {
            "model": MODEL,
            "dataset": DATASET_NAME,
            "num_samples": NUM_SAMPLES,
            "detection_mode": DETECTION_MODE,
            "use_fewshot": USE_FEWSHOT,
            "evaluation": eval_name,
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "presence_accuracy": results['metrics']['presence_accuracy'],
                "mean_iou_bbox_to_bbox": results['metrics'].get('mean_iou_bbox_to_bbox', 0),
                "mean_iou_bbox_to_mask": results['metrics'].get('mean_iou_bbox_to_mask', 0),
                "elapsed_seconds": elapsed
            }
        }
        
        print(f"✓ Presence Accuracy: {results['metrics']['presence_accuracy']:.1%}")
        print(f"  Bbox-to-Bbox IoU: {results['metrics'].get('mean_iou_bbox_to_bbox', 0):.3f}")
        print(f"  Bbox-to-Mask IoU: {results['metrics'].get('mean_iou_bbox_to_mask', 0):.3f}")
        print(f"  Time: {elapsed:.1f}s")
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        results_summary = {
            "model": MODEL,
            "dataset": DATASET_NAME,
            "num_samples": NUM_SAMPLES,
            "detection_mode": DETECTION_MODE,
            "use_fewshot": USE_FEWSHOT,
            "evaluation": eval_name,
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "elapsed_seconds": time.time() - start_time
        }
    
    # Save summary with mode-specific filename
    summary_filename = f"summary_{DETECTION_MODE}_{'fewshot' if USE_FEWSHOT else 'zeroshot'}.json"
    summary_file = evaluator.output_dir / summary_filename
    with open(summary_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Model: {MODEL}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Evaluation: {eval_name}")
    print(f"Samples evaluated: {NUM_SAMPLES}")
    print(f"Results saved to: {evaluator.output_dir}")
    print(f"Summary file: {summary_filename}")
    print()
    
    # Print summary
    if "error" not in results_summary:
        print("📊 RESULTS:")
        print("-" * 60)
        print(f"Presence Accuracy: {results_summary['metrics']['presence_accuracy']:.1%}")
        print(f"Bbox-to-Bbox IoU:  {results_summary['metrics']['mean_iou_bbox_to_bbox']:.3f}")
        print(f"Bbox-to-Mask IoU:  {results_summary['metrics']['mean_iou_bbox_to_mask']:.3f}")
        print(f"Evaluation Time:   {results_summary['metrics']['elapsed_seconds']:.1f}s")
        print("-" * 60)
    else:
        print(f"❌ Evaluation failed: {results_summary['error']}")


if __name__ == "__main__":
    main()