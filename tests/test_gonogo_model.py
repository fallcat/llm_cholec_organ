#!/usr/bin/env python3
"""Test script for GoNoGoNet model."""

import sys
import json
from pathlib import Path
import numpy as np
from PIL import Image

# Add src to path
sys.path.insert(0, '/shared_data0/weiqiuy/llm_cholec_organ/src')

from endopoint.models.gonogonet import GoNoGoNet, load_gonogo_model
from endopoint.models.gonogonet_adapter import GoNoGoNetAdapter
from endopoint.datasets.cholec_gonogo import CholecGoNoGoAdapter


def test_gonogo_model():
    """Test GoNoGoNet model loading and inference."""
    
    print("=" * 80)
    print("TESTING GONOGONET MODEL")
    print("=" * 80)
    
    # Test 1: Load model
    print("\n1. Loading GoNoGoNet model...")
    try:
        model = load_gonogo_model(device='cuda')
        print("✓ Model loaded successfully")
        print(f"  - Device: {next(model.parameters()).device}")
        print(f"  - Classes: {model.ID2LABEL}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False
    
    # Test 2: Load dataset and get sample image
    print("\n2. Loading test image from dataset...")
    try:
        dataset = CholecGoNoGoAdapter()
        example = dataset.get_example('test', 0)
        image = example['image']
        ground_truth = example['gonogo_label']
        print(f"✓ Loaded test image: {image.size}")
        print(f"  - Ground truth label shape: {ground_truth.size}")
    except Exception as e:
        print(f"✗ Failed to load test image: {e}")
        return False
    
    # Test 3: Run inference
    print("\n3. Running model inference...")
    try:
        mask = model.get_segmentation_mask(image)
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
        for organ, is_present in presence.items():
            status = "Present" if is_present else "Absent"
            print(f"  - {organ}: {status}")
    except Exception as e:
        print(f"✗ Failed to extract organ presence: {e}")
        return False
    
    # Test 5: Extract bounding boxes
    print("\n5. Extracting bounding boxes...")
    try:
        bboxes = model.get_bounding_boxes(mask, min_pixels=50)
        print(f"✓ Bounding boxes extracted:")
        for organ, boxes in bboxes.items():
            print(f"  - {organ}: {len(boxes)} regions")
            for i, (x1, y1, x2, y2) in enumerate(boxes[:2]):  # Show first 2 boxes
                print(f"    Box {i+1}: ({x1}, {y1}, {x2}, {y2})")
    except Exception as e:
        print(f"✗ Failed to extract bounding boxes: {e}")
        return False
    
    # Test 6: Test adapter JSON output
    print("\n6. Testing GoNoGoAdapter JSON output...")
    try:
        adapter = GoNoGoAdapter(verbose=False, use_cache=False)
        
        # Create a mock prompt
        prompt = """Detect the following organs in the image:
- Go Zone
- NoGo Zone

Return JSON with format:
{
  "Go Zone": {"present": true/false, "bbox": [x1, y1, x2, y2] or null},
  "NoGo Zone": {"present": true/false, "bbox": [x1, y1, x2, y2] or null}
}"""
        
        # Process through adapter
        responses = adapter([(prompt, image)], system_prompt="")
        response_json = json.loads(responses[0])
        
        print("✓ Adapter JSON response generated:")
        print(json.dumps(response_json, indent=2)[:500] + "..." if len(json.dumps(response_json)) > 500 else json.dumps(response_json, indent=2))
        
        # Verify response structure
        assert "Go Zone" in response_json, "Missing 'Go Zone' in response"
        assert "NoGo Zone" in response_json, "Missing 'NoGo Zone' in response"
        
        for zone in ["Go Zone", "NoGo Zone"]:
            assert "present" in response_json[zone], f"Missing 'present' field for {zone}"
            assert "bbox" in response_json[zone], f"Missing 'bbox' field for {zone}"
            
            if response_json[zone]["present"]:
                bbox = response_json[zone]["bbox"]
                assert bbox is not None, f"Bbox is None for present {zone}"
                assert len(bbox) == 4, f"Invalid bbox format for {zone}: {bbox}"
        
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
        
        # Calculate IoU for each class
        for class_id, class_name in model.ID2LABEL.items():
            if class_id == 0:  # Skip background
                continue
            
            pred_mask = (mask == class_id)
            gt_mask = (gt_array == class_id)
            
            intersection = np.logical_and(pred_mask, gt_mask).sum()
            union = np.logical_or(pred_mask, gt_mask).sum()
            
            if union > 0:
                iou = intersection / union
                print(f"  - {class_name} IoU: {iou:.3f}")
            else:
                print(f"  - {class_name}: No pixels in prediction or ground truth")
    
    except Exception as e:
        print(f"✗ Failed to compare with ground truth: {e}")
        return False
    
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED ✓")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    success = test_gonogo_model()
    sys.exit(0 if success else 1)