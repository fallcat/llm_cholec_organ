#!/usr/bin/env python3
"""Test script for CholeNet model."""

import sys
import json
from pathlib import Path
import numpy as np
from PIL import Image

# Add src to path
sys.path.insert(0, '/shared_data0/weiqiuy/llm_cholec_organ/src')

from endopoint.models.cholenet import CholeNet, load_cholenet_model
from endopoint.models.cholenet_adapter import CholeNetAdapter
from endopoint.datasets.cholecseg8k_local import CholecSeg8kLocalAdapter


def test_cholenet_model():
    """Test CholeNet model loading and inference."""
    
    print("=" * 80)
    print("TESTING CHOLENET MODEL")
    print("=" * 80)
    
    # Test 1: Load model
    print("\n1. Loading CholeNet model...")
    try:
        model = load_cholenet_model(device='cuda')
        print("✓ Model loaded successfully")
        print(f"  - Device: {next(model.parameters()).device}")
        print(f"  - Classes: {list(model.ID2LABEL.values())}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False
    
    # Test 2: Load dataset and get sample image
    print("\n2. Loading test image from CholecSeg8k dataset...")
    try:
        dataset = CholecSeg8kLocalAdapter()
        example = dataset.get_example('test', 0)
        image = example['image']
        ground_truth = example['labels']
        print(f"✓ Loaded test image: {image.size}")
        print(f"  - Ground truth label shape: {ground_truth.size}")
    except Exception as e:
        print(f"✗ Failed to load test image: {e}")
        return False
    
    # Test 3: Run inference
    print("\n3. Running model inference...")
    try:
        mask = model.get_segmentation_mask(image, target_size=(640, 384))
        print(f"✓ Generated segmentation mask: {mask.shape}")
        print(f"  - Unique classes in mask: {np.unique(mask).tolist()}")
        
        # Check mask dimensions
        assert mask.shape == (384, 640), f"Unexpected mask shape: {mask.shape}"
        print(f"  - Mask dimensions correct: {mask.shape}")
    except Exception as e:
        print(f"✗ Failed to run inference: {e}")
        return False
    
    # Test 4: Extract organ presence
    print("\n4. Extracting organ presence...")
    try:
        presence = model.get_organ_presence(mask, min_pixels=50)
        print(f"✓ Organ presence detected:")
        present_organs = [organ for organ, is_present in presence.items() if is_present]
        absent_organs = [organ for organ, is_present in presence.items() if not is_present]
        print(f"  - Present organs ({len(present_organs)}): {', '.join(present_organs[:5])}")
        if len(present_organs) > 5:
            print(f"    ... and {len(present_organs) - 5} more")
        print(f"  - Absent organs ({len(absent_organs)}): {', '.join(absent_organs[:5])}")
    except Exception as e:
        print(f"✗ Failed to extract organ presence: {e}")
        return False
    
    # Test 5: Extract bounding boxes
    print("\n5. Extracting bounding boxes...")
    try:
        bboxes = model.get_bounding_boxes(mask, min_pixels=50)
        print(f"✓ Bounding boxes extracted:")
        for organ, boxes in list(bboxes.items())[:3]:  # Show first 3 organs
            print(f"  - {organ}: {len(boxes)} regions")
            for i, (x1, y1, x2, y2) in enumerate(boxes[:2]):  # Show first 2 boxes
                print(f"    Box {i+1}: ({x1}, {y1}, {x2}, {y2})")
    except Exception as e:
        print(f"✗ Failed to extract bounding boxes: {e}")
        return False
    
    # Test 6: Test adapter JSON output
    print("\n6. Testing CholeNetAdapter JSON output...")
    try:
        adapter = CholeNetAdapter(verbose=False, use_cache=False, return_masks=False)
        
        # Create a mock prompt for CholecSeg8k organs
        prompt = """Detect the following organs in the image:
- Black Background
- Abdominal Wall
- Liver
- Gastrointestinal Tract
- Fat
- Grasper
- Connective Tissue
- Blood
- Cystic Duct
- L-hook Electrocautery
- Gallbladder
- Hepatic Vein
- Liver Ligament

Return JSON with format:
{
  "Organ Name": {"present": true/false, "bbox": [x1, y1, x2, y2] or null}
}"""
        
        # Process through adapter
        responses = adapter([(prompt, image)], system_prompt="")
        response_json = json.loads(responses[0])
        
        print("✓ Adapter JSON response generated:")
        # Show first few organs
        shown_organs = list(response_json.keys())[:3]
        for organ in shown_organs:
            organ_data = response_json[organ]
            print(f"  - {organ}: present={organ_data['present']}, bbox={organ_data['bbox']}")
        if len(response_json) > 3:
            print(f"  ... and {len(response_json) - 3} more organs")
        
        # Verify response structure
        expected_organs = ["Liver", "Gallbladder", "Fat", "Grasper"]
        for organ in expected_organs:
            if organ in response_json:
                assert "present" in response_json[organ], f"Missing 'present' field for {organ}"
                assert "bbox" in response_json[organ], f"Missing 'bbox' field for {organ}"
        
        print("\n  ✓ Response structure validated")
        
    except Exception as e:
        print(f"✗ Failed to test adapter: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 7: Compare with ground truth
    print("\n7. Comparing with ground truth...")
    try:
        # Convert ground truth PIL image to numpy
        gt_array = np.array(ground_truth)
        
        # Get dataset's ID to label mapping
        id2label = dataset.id2label
        
        # Calculate IoU for major organs
        test_organs = ["Liver", "Gallbladder", "Fat"]
        for organ_name in test_organs:
            if organ_name in model.LABEL2ID:
                class_id = model.LABEL2ID[organ_name]
                
                pred_mask = (mask == class_id)
                
                # Find corresponding ID in ground truth
                gt_class_id = None
                for gt_id, gt_name in id2label.items():
                    if gt_name.lower() == organ_name.lower():
                        gt_class_id = gt_id
                        break
                
                if gt_class_id is not None:
                    gt_mask = (gt_array == gt_class_id)
                    
                    intersection = np.logical_and(pred_mask, gt_mask).sum()
                    union = np.logical_or(pred_mask, gt_mask).sum()
                    
                    if union > 0:
                        iou = intersection / union
                        print(f"  - {organ_name} IoU: {iou:.3f}")
                    else:
                        print(f"  - {organ_name}: No pixels in prediction or ground truth")
    
    except Exception as e:
        print(f"✗ Failed to compare with ground truth: {e}")
        return False
    
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED ✓")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    success = test_cholenet_model()
    sys.exit(0 if success else 1)