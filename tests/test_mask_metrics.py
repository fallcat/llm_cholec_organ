#!/usr/bin/env python3
"""Test script to evaluate GoNoGoNet with all mask metrics."""

import sys
import json
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, '/shared_data0/weiqiuy/llm_cholec_organ/src')

os.environ['EVAL_MODEL'] = 'gonogo'
os.environ['EVAL_NUM_SAMPLES'] = '2'
os.environ['EVAL_USE_CACHE'] = 'false'

from endopoint.eval.bbox_evaluator import BoundingBoxEvaluator
from endopoint.datasets.cholec_gonogo import CholecGoNoGoAdapter


def test_mask_metrics():
    """Test GoNoGoNet with all mask metrics."""
    print("=" * 80)
    print("TESTING MASK METRICS WITH GONOGONET")
    print("=" * 80)
    
    # Load dataset
    dataset = CholecGoNoGoAdapter()
    
    # Get image dimensions
    example = dataset.get_example('test', 0)
    img_width, img_height = example['image'].size
    
    # Test indices
    test_indices = [63, 558]
    
    # Output directory
    output_dir = Path("/shared_data0/weiqiuy/llm_cholec_organ/results/test_mask_metrics")
    
    # Initialize evaluator
    evaluator = BoundingBoxEvaluator(
        models=['gonogo'],
        dataset=None,
        dataset_adapter=dataset,
        canvas_width=img_width,
        canvas_height=img_height,
        output_dir=output_dir,
        use_cache=False,
        min_pixels=50
    )
    
    print(f"Testing {len(test_indices)} samples: {test_indices}")
    print()
    
    # Run evaluation
    results = evaluator.evaluate_model(
        model_name='gonogo',
        test_indices=test_indices,
        detection_mode='combined',
        use_fewshot=False,
        split='test'
    )
    
    # Display comprehensive results
    print("\n" + "=" * 80)
    print("COMPREHENSIVE IOU METRICS")
    print("=" * 80)
    
    metrics = results['metrics']
    
    # Overall metrics
    print("\nOVERALL METRICS:")
    print("-" * 40)
    print(f"Presence Accuracy:      {metrics['presence_accuracy']:.1%}")
    print()
    print("IoU Comparisons:")
    print(f"  Bbox-to-Bbox:         {metrics['mean_iou_bbox_to_bbox']:.3f}")
    print(f"  Bbox-to-Mask (GT):    {metrics['mean_iou_bbox_to_mask']:.3f}")
    print(f"  Mask-to-Mask:         {metrics['mean_iou_mask_to_mask']:.3f}  ← Best for segmentation")
    print(f"  Mask-to-Bbox (GT):    {metrics['mean_iou_mask_to_bbox']:.3f}")
    
    # IoU thresholds for mask-to-mask
    print("\nMask-to-Mask IoU at thresholds:")
    print(f"  IoU ≥ 0.30:           {metrics.get('iou_at_0.3_mask_to_mask', 0):.1%}")
    print(f"  IoU ≥ 0.50:           {metrics.get('iou_at_0.5_mask_to_mask', 0):.1%}")
    print(f"  IoU ≥ 0.75:           {metrics.get('iou_at_0.75_mask_to_mask', 0):.1%}")
    
    # Per-organ results
    print("\n" + "=" * 80)
    print("PER-ORGAN RESULTS")
    print("=" * 80)
    
    for organ, organ_metrics in metrics['per_organ'].items():
        print(f"\n{organ}:")
        print("-" * 40)
        print(f"  Presence Accuracy:    {organ_metrics['presence_accuracy']:.1%}")
        print(f"  Bbox-to-Bbox IoU:     {organ_metrics['mean_iou_bbox_to_bbox']:.3f}")
        print(f"  Bbox-to-Mask IoU:     {organ_metrics['mean_iou_bbox_to_mask']:.3f}")
        print(f"  Mask-to-Mask IoU:     {organ_metrics['mean_iou_mask_to_mask']:.3f}  ← Best")
        print(f"  Mask-to-Bbox IoU:     {organ_metrics['mean_iou_mask_to_bbox']:.3f}")
    
    # Check individual predictions
    print("\n" + "=" * 80)
    print("SAMPLE PREDICTIONS")
    print("=" * 80)
    
    for i, pred in enumerate(results['predictions'][:4]):  # Show first 4
        print(f"\nSample {pred['test_idx']}, {pred['organ_name']}:")
        print(f"  GT Present: {pred['ground_truth_present']}, Pred Present: {pred['predicted_present']}")
        if pred['ground_truth_present'] and pred['predicted_present']:
            print(f"  Has Mask: {pred.get('predicted_mask', False)}")
            print(f"  Bbox-to-Bbox IoU: {pred.get('iou_bbox_to_bbox', 0):.3f}")
            print(f"  Mask-to-Mask IoU: {pred.get('iou_mask_to_mask', 0):.3f}")
    
    # Save detailed results
    results_file = output_dir / "detailed_mask_metrics.json"
    with open(results_file, 'w') as f:
        json.dump({
            'test_indices': test_indices,
            'metrics': metrics,
            'model': 'gonogo'
        }, f, indent=2)
    
    print(f"\n✓ Detailed results saved to: {results_file}")
    
    return True


if __name__ == "__main__":
    success = test_mask_metrics()
    sys.exit(0 if success else 1)