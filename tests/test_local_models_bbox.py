#!/usr/bin/env python3
"""Test local models (GoNoGoNet, CholeNet) with bbox evaluator."""

import sys
import json
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, '/shared_data0/weiqiuy/llm_cholec_organ/src')

# Set environment variables
os.environ['EVAL_MODEL'] = 'gonogo'  # or 'cholenet'
os.environ['EVAL_NUM_SAMPLES'] = '2'
os.environ['EVAL_USE_CACHE'] = 'false'
os.environ['EVAL_DETECTION_MODE'] = 'combined'
os.environ['EVAL_USE_FEWSHOT'] = 'false'

from endopoint.eval.bbox_evaluator import BoundingBoxEvaluator
from endopoint.datasets.cholec_gonogo import CholecGoNoGoAdapter
from endopoint.datasets.cholecseg8k_local import CholecSeg8kLocalAdapter


def test_gonogo_with_evaluator():
    """Test GoNoGoNet with bbox evaluator."""
    print("=" * 80)
    print("TESTING GONOGONET WITH BBOX EVALUATOR")
    print("=" * 80)
    
    # Load dataset
    dataset = CholecGoNoGoAdapter()
    
    # Get image dimensions
    example = dataset.get_example('test', 0)
    img_width, img_height = example['image'].size
    
    # Test indices (just 2 for quick test)
    test_indices = [63, 558]
    
    # Output directory
    output_dir = Path("/shared_data0/weiqiuy/llm_cholec_organ/results/test_gonogo_local")
    
    # Initialize evaluator
    evaluator = BoundingBoxEvaluator(
        models=['gonogo'],  # Use GoNoGoNet adapter
        dataset=None,
        dataset_adapter=dataset,
        canvas_width=img_width,
        canvas_height=img_height,
        output_dir=output_dir,
        use_cache=False,
        min_pixels=50
    )
    
    print(f"Output directory: {evaluator.output_dir}")
    print(f"Testing {len(test_indices)} samples: {test_indices}")
    print()
    
    # Run evaluation
    try:
        results = evaluator.evaluate_model(
            model_name='gonogo',
            test_indices=test_indices,
            detection_mode='combined',
            use_fewshot=False,
            split='test'
        )
        
        # Display results
        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)
        
        metrics = results['metrics']
        print(f"Presence Accuracy: {metrics['presence_accuracy']:.1%}")
        print(f"Bbox-to-Bbox IoU: {metrics['mean_iou_bbox_to_bbox']:.3f}")
        print(f"Bbox-to-Mask IoU: {metrics['mean_iou_bbox_to_mask']:.3f}")
        
        # Per-organ results
        print("\nPer-organ results:")
        for organ, organ_metrics in metrics['per_organ'].items():
            print(f"  {organ}:")
            print(f"    - Presence Acc: {organ_metrics['presence_accuracy']:.1%}")
            print(f"    - Bbox IoU: {organ_metrics['mean_iou_bbox_to_bbox']:.3f}")
            print(f"    - Mask IoU: {organ_metrics['mean_iou_bbox_to_mask']:.3f}")
        
        print("\n✓ GoNoGoNet evaluation completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cholenet_with_evaluator():
    """Test CholeNet with bbox evaluator."""
    print("\n" + "=" * 80)
    print("TESTING CHOLENET WITH BBOX EVALUATOR")
    print("=" * 80)
    
    # Load dataset
    dataset = CholecSeg8kLocalAdapter()
    
    # Get image dimensions
    example = dataset.get_example('test', 0)
    img_width, img_height = example['image'].size
    
    # Test indices (just 2 for quick test)
    test_indices = [0, 10]
    
    # Output directory
    output_dir = Path("/shared_data0/weiqiuy/llm_cholec_organ/results/test_cholenet_local")
    
    # Initialize evaluator
    evaluator = BoundingBoxEvaluator(
        models=['cholenet'],  # Use CholeNet adapter
        dataset=None,
        dataset_adapter=dataset,
        canvas_width=img_width,
        canvas_height=img_height,
        output_dir=output_dir,
        use_cache=False,
        min_pixels=50
    )
    
    print(f"Output directory: {evaluator.output_dir}")
    print(f"Testing {len(test_indices)} samples: {test_indices}")
    print()
    
    # Run evaluation
    try:
        results = evaluator.evaluate_model(
            model_name='cholenet',
            test_indices=test_indices,
            detection_mode='combined',
            use_fewshot=False,
            split='test'
        )
        
        # Display results
        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)
        
        metrics = results['metrics']
        print(f"Presence Accuracy: {metrics['presence_accuracy']:.1%}")
        print(f"Bbox-to-Bbox IoU: {metrics['mean_iou_bbox_to_bbox']:.3f}")
        print(f"Bbox-to-Mask IoU: {metrics['mean_iou_bbox_to_mask']:.3f}")
        
        # Show top 5 organs by presence accuracy
        print("\nTop organs by presence accuracy:")
        organ_list = [(name, m['presence_accuracy']) for name, m in metrics['per_organ'].items()]
        organ_list.sort(key=lambda x: x[1], reverse=True)
        for organ, acc in organ_list[:5]:
            organ_metrics = metrics['per_organ'][organ]
            print(f"  {organ}: {acc:.1%} (IoU: {organ_metrics['mean_iou_bbox_to_bbox']:.3f})")
        
        print("\n✓ CholeNet evaluation completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Test both models
    success = True
    
    # Test GoNoGoNet
    if not test_gonogo_with_evaluator():
        success = False
    
    # Test CholeNet
    if not test_cholenet_with_evaluator():
        success = False
    
    if success:
        print("\n" + "=" * 80)
        print("ALL TESTS PASSED ✓")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("SOME TESTS FAILED ✗")
        print("=" * 80)
    
    sys.exit(0 if success else 1)