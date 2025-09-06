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
    print()
    
    # ========================================================================
    # MODE 1: TIMESTAMPED OUTPUT (for experiments)
    # ========================================================================
    
    print("🕒 MODE 1: TIMESTAMPED OUTPUT (Experiments)")
    print("-" * 50)
    
    evaluator_timestamped = BoundingBoxEvaluator(
        models=MODELS,
        dataset=None,
        dataset_adapter=dataset_adapter,
        canvas_width=img_width,
        canvas_height=img_height,
        use_cache=True,
        min_pixels=50,
        use_timestamp=True  # Each run gets unique timestamp
    )
    
    print(f"Output directory: {evaluator_timestamped.output_dir}")
    
    # Run zero-shot evaluation
    print("Running zero-shot evaluation...")
    zs_results = evaluator_timestamped.evaluate_model(
        model_name=MODELS[0],
        test_indices=test_indices,
        detection_mode="combined",
        use_fewshot=False,
        split='test'
    )
    
    print(f"✅ Timestamped results saved to: {evaluator_timestamped.output_dir}")
    print(f"   Presence Accuracy: {zs_results['metrics']['presence_accuracy']:.1%}")
    print()
    
    # ========================================================================  
    # MODE 2: PERSISTENT OUTPUT (for production/consistent results)
    # ========================================================================
    
    print("📁 MODE 2: PERSISTENT OUTPUT (Production/Consistent)")
    print("-" * 50)
    
    evaluator_persistent = BoundingBoxEvaluator(
        models=MODELS,
        dataset=None,
        dataset_adapter=dataset_adapter,
        canvas_width=img_width,
        canvas_height=img_height,
        use_cache=True,
        min_pixels=50,
        use_timestamp=False  # Same directory across runs
    )
    
    print(f"Output directory: {evaluator_persistent.output_dir}")
    
    # Check if results already exist
    persistent_dir = evaluator_persistent.output_dir / "zeroshot_combined" / MODELS[0]
    existing_files = list(persistent_dir.glob("test_*.json")) if persistent_dir.exists() else []
    
    if existing_files:
        print(f"Found {len(existing_files)} existing prediction files - will reuse them")
        # Load and compute metrics from existing files
        metrics = evaluator_persistent.load_and_compute_metrics(persistent_dir)
        print(f"✅ Loaded existing results from: {persistent_dir}")
        print(f"   Presence Accuracy: {metrics['presence_accuracy']:.1%}")
    else:
        print("No existing results found - running fresh evaluation...")
        # Run evaluation
        persistent_results = evaluator_persistent.evaluate_model(
            model_name=MODELS[0],
            test_indices=test_indices,
            detection_mode="combined",
            use_fewshot=False,
            split='test'
        )
        
        print(f"✅ Persistent results saved to: {evaluator_persistent.output_dir}")
        print(f"   Presence Accuracy: {persistent_results['metrics']['presence_accuracy']:.1%}")
    
    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print("📈 Timestamped Mode:")
    print(f"   • Use for: Experiments, comparisons, ablation studies")
    print(f"   • Output: results/bbox_dataset_YYYYMMDD_HHMMSS/")
    print(f"   • Behavior: Each run creates new directory")
    print()
    print("🏭 Persistent Mode:")
    print(f"   • Use for: Production runs, consistent baselines")
    print(f"   • Output: results/bbox_dataset/")
    print(f"   • Behavior: Reuses existing predictions, avoids redundant API calls")
    print()
    print("Both modes save individual test_XXXXX.json files for granular analysis!")

if __name__ == "__main__":
    main()