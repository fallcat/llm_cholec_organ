#!/usr/bin/env python3
"""
Run bounding box evaluation with configurable output directory structure.

This script demonstrates both timestamped and persistent output modes:
- Timestamped: results/bbox_cholecseg8k_local_20250906_123456/ (for experiments)
- Persistent: results/bbox_cholecseg8k_local/ (for production/consistent results)
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

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

from endopoint.datasets.cholecseg8k_local import CholecSeg8kLocalAdapter
from endopoint.eval.bbox_evaluator import BoundingBoxEvaluator

def main():
    """Run bbox evaluation with different output modes."""
    
    # Configuration
    DATASET_NAME = "cholecseg8k_local"
    MODELS = ["gpt-4.1"]
    QUICK_TEST = True  # Set to False for full evaluation
    
    # Load dataset
    data_dir = "/shared_data0/weiqiuy/datasets/cholecseg8k"
    dataset_adapter = CholecSeg8kLocalAdapter(data_dir=data_dir)
    
    # Load test indices
    indices_dir = Path(f"/shared_data0/weiqiuy/llm_cholec_organ/data_info/{DATASET_NAME}_balanced_200")
    with open(indices_dir / "balanced_test_indices_advanced_200.json", 'r') as f:
        test_data = json.load(f)
    test_indices = test_data['indices'][:5] if QUICK_TEST else test_data['indices']
    
    # Load few-shot plan
    with open(indices_dir / "fewshot_plan_bbox_200.json", 'r') as f:
        fewshot_plan = json.load(f)
    
    # Get image dimensions
    example = dataset_adapter.get_example('train', 0)
    img_width, img_height = example['image'].size
    
    print("="*80)
    print("BOUNDING BOX EVALUATION - OUTPUT MODE COMPARISON")
    print("="*80)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Models: {MODELS}")
    print(f"Samples: {len(test_indices)} ({'QUICK TEST' if QUICK_TEST else 'FULL'})")
    print(f"Evaluation modes: Combined + Separate × Zero-shot + Few-shot = 4 combinations each")
    print()
    
    # ========================================================================
    # TIMESTAMPED OUTPUT (All runs use unique timestamps)
    # ========================================================================
    
    print("🕒 TIMESTAMPED OUTPUT")
    print("-" * 50)
    
    evaluator = BoundingBoxEvaluator(
        models=MODELS,
        dataset=None,
        dataset_adapter=dataset_adapter,
        canvas_width=img_width,
        canvas_height=img_height,
        use_cache=True,
        min_pixels=50,
        use_timestamp=True  # Always use timestamp for unique directories
    )
    
    print(f"Output directory: {evaluator.output_dir}")
    
    # Run all evaluation combinations
    eval_configs = [
        {"mode": "combined", "fewshot": False, "name": "Zero-shot Combined"},
        {"mode": "combined", "fewshot": True, "name": "Few-shot Combined"},
        {"mode": "separate", "fewshot": False, "name": "Zero-shot Separate"},
        {"mode": "separate", "fewshot": True, "name": "Few-shot Separate"}
    ]
    
    timestamped_results = {}
    for config in eval_configs:
        print(f"Running {config['name']} evaluation...")
        results = evaluator.evaluate_model(
            model_name=MODELS[0],
            test_indices=test_indices,
            detection_mode=config['mode'],
            use_fewshot=config['fewshot'],
            fewshot_plan=fewshot_plan if config['fewshot'] else None,
            split='test'
        )
        timestamped_results[config['name']] = results
        print(f"   ✓ {config['name']} Presence Accuracy: {results['metrics']['presence_accuracy']:.1%}")
    
    print(f"\n✅ All results saved to: {evaluator.output_dir}")
    for name, results in timestamped_results.items():
        print(f"   {name}: {results['metrics']['presence_accuracy']:.1%}")
    print()
    
    
    print()
    print("="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    print("📈 RESULTS:")
    for name, results in timestamped_results.items():
        metrics = results['metrics']
        print(f"   {name:20s}: {metrics['presence_accuracy']:.1%} presence, "
              f"{metrics.get('mean_iou_bbox_to_bbox', 0):.3f} bbox IoU, "
              f"{metrics.get('mean_iou_bbox_to_mask', 0):.3f} mask IoU")
    
    print(f"\n📊 COMPARISON INSIGHTS:")
    print(f"   • Combined vs Separate: Shows API efficiency vs granular control")
    print(f"   • Zero-shot vs Few-shot: Shows impact of example guidance")
    print(f"   • Dual IoU metrics: Bbox-to-bbox (standard) vs bbox-to-mask (anatomical)")
    print(f"\nIndividual test_XXXXX.json files saved for detailed analysis!")
    print(f"Dataset: {len(test_indices)} samples ({'QUICK TEST' if QUICK_TEST else 'FULL EVALUATION'})")
    print(f"Output: {evaluator.output_dir}")

if __name__ == "__main__":
    main()