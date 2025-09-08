#!/usr/bin/env python3
"""
Test RASO model on all three datasets: CholecSeg8k, CholecOrgans, and CholecGoNoGo.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '/shared_data0/weiqiuy/llm_cholec_organ/src')


def test_dataset(dataset_name, num_samples=5):
    """Test RASO on a specific dataset.
    
    Args:
        dataset_name: Name of the dataset (cholecseg8k, cholec_organs, cholec_gonogo)
        num_samples: Number of samples to test
        
    Returns:
        Dict with test results
    """
    print("\n" + "=" * 80)
    print(f"TESTING RASO ON {dataset_name.upper()}")
    print("=" * 80)
    
    # Set environment variables
    os.environ['EVAL_DATASET'] = dataset_name
    os.environ['EVAL_MODEL'] = 'raso'
    os.environ['EVAL_NUM_SAMPLES'] = str(num_samples)
    os.environ['EVAL_USE_CACHE'] = 'false'  # Disable cache for testing
    os.environ['EVAL_PERSISTENT_DIR'] = 'false'  # Use timestamped dir
    os.environ['EVAL_DETECTION_MODE'] = 'combined'  # Use combined mode
    os.environ['EVAL_USE_FEWSHOT'] = 'false'  # Zero-shot
    
    # Import here to get fresh environment variables
    from notebooks_py.eval_bbox_unified import main, load_dataset_adapter, get_dataset_display_name
    
    print(f"\nConfiguration:")
    print(f"  Dataset: {dataset_name}")
    print(f"  Model: RASO")
    print(f"  Samples: {num_samples}")
    print(f"  Mode: Zero-shot Combined")
    print(f"  Cache: Disabled")
    
    # Load dataset to show info
    try:
        adapter = load_dataset_adapter(dataset_name)
        
        # Get dataset dimensions
        if dataset_name == "cholecseg8k":
            image_width, image_height = 854, 480
            # CholecSeg8k organ classes (from label file)
            organ_classes = [
                "black background", "abdominal wall", "liver", "gastrointestinal tract", 
                "fat", "grasper", "connective tissue", "blood", "cystic duct",
                "l-hook electrocautery", "gallbladder", "hepatic vein", "liver ligament"
            ]
        elif dataset_name == "cholec_organs":
            image_width, image_height = 640, 480
            # CholecOrgans classes (from label file)
            organ_classes = ["background", "liver", "gallbladder", "hepatocystic triangle"]
        elif dataset_name == "cholec_gonogo":
            image_width, image_height = 640, 480
            # CholecGoNoGo classes (from label file)
            organ_classes = ["background", "go (safe to incise)", "nogo (unsafe to incise)"]
        else:
            image_width, image_height = 640, 480
            organ_classes = []
            
        print(f"  Image dimensions: {image_width}x{image_height}")
        print(f"  Organ classes ({len(organ_classes)}): {', '.join(organ_classes[:5])}...")
        
    except Exception as e:
        print(f"Error loading adapter: {e}")
        return {"error": str(e)}
    
    # Run evaluation
    try:
        result_code = main()
        
        # Try to load and display results
        results_dir = Path(f"/shared_data0/weiqiuy/llm_cholec_organ/results")
        latest_dir = None
        
        # Find the most recent results directory for this dataset
        for dir_path in sorted(results_dir.glob(f"*{dataset_name}*"), reverse=True):
            if dir_path.is_dir() and "20" in dir_path.name:  # Has timestamp
                latest_dir = dir_path
                break
        
        if latest_dir:
            summary_file = latest_dir / "summary_combined_zeroshot.json"
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                    
                print(f"\n📊 Results Summary:")
                print(f"  Results directory: {latest_dir.name}")
                
                if 'raso' in summary:
                    metrics = summary['raso']
                    print(f"  Presence Accuracy: {metrics.get('presence_accuracy', 0)*100:.1f}%")
                    print(f"  Bbox-to-Bbox IoU: {metrics.get('bbox_to_bbox_iou', 0):.3f}")
                    print(f"  Bbox-to-Mask IoU: {metrics.get('bbox_to_mask_iou', 0):.3f}")
                    print(f"  Evaluation Time: {metrics.get('eval_time', 0):.1f}s")
                    
                    return {
                        "dataset": dataset_name,
                        "success": True,
                        "presence_accuracy": metrics.get('presence_accuracy', 0),
                        "bbox_iou": metrics.get('bbox_to_bbox_iou', 0),
                        "mask_iou": metrics.get('bbox_to_mask_iou', 0),
                        "eval_time": metrics.get('eval_time', 0),
                        "results_dir": str(latest_dir)
                    }
        
        return {
            "dataset": dataset_name,
            "success": result_code == 0,
            "results_dir": str(latest_dir) if latest_dir else None
        }
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return {
            "dataset": dataset_name,
            "success": False,
            "error": str(e)
        }


def main():
    """Test RASO on all three datasets."""
    
    print("=" * 80)
    print("RASO MODEL EVALUATION ON ALL DATASETS")
    print("=" * 80)
    print(f"Start time: {datetime.now()}")
    
    # Test all three datasets
    datasets = ["cholecseg8k", "cholec_organs", "cholec_gonogo"]
    num_samples = 5  # Use 5 samples for quick testing
    
    results = {}
    
    for dataset in datasets:
        result = test_dataset(dataset, num_samples=num_samples)
        results[dataset] = result
        
        # Print progress
        print(f"\n✓ Completed {dataset}")
    
    # Print final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    print("\n📊 Presence Accuracy Comparison:")
    print("-" * 40)
    
    for dataset in datasets:
        if results[dataset].get('success'):
            acc = results[dataset].get('presence_accuracy', 0) * 100
            print(f"  {dataset:15s}: {acc:5.1f}%")
        else:
            print(f"  {dataset:15s}: FAILED")
    
    print("\n📊 Bounding Box IoU (should be 0 for RASO):")
    print("-" * 40)
    
    for dataset in datasets:
        if results[dataset].get('success'):
            bbox_iou = results[dataset].get('bbox_iou', 0)
            mask_iou = results[dataset].get('mask_iou', 0)
            print(f"  {dataset:15s}: Bbox={bbox_iou:.3f}, Mask={mask_iou:.3f}")
        else:
            print(f"  {dataset:15s}: N/A")
    
    print("\n📊 Evaluation Times:")
    print("-" * 40)
    
    total_time = 0
    for dataset in datasets:
        if results[dataset].get('success'):
            eval_time = results[dataset].get('eval_time', 0)
            total_time += eval_time
            print(f"  {dataset:15s}: {eval_time:6.1f}s")
        else:
            print(f"  {dataset:15s}: N/A")
    
    print(f"  {'Total':15s}: {total_time:6.1f}s")
    
    # Save results to file
    output_file = Path(f"/shared_data0/weiqiuy/llm_cholec_organ/results/raso_all_datasets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    print(f"\nEnd time: {datetime.now()}")
    
    # Check if all succeeded
    all_success = all(r.get('success', False) for r in results.values())
    if all_success:
        print("\n🎉 All dataset evaluations completed successfully!")
    else:
        failed = [d for d, r in results.items() if not r.get('success', False)]
        print(f"\n⚠️ Failed datasets: {', '.join(failed)}")
    
    return 0 if all_success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)