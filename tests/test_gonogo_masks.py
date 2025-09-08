#!/usr/bin/env python3
"""Test GoNoGoNet mask generation and metrics."""

import sys
import json
import numpy as np
from PIL import Image

# Add src to path
sys.path.insert(0, '/shared_data0/weiqiuy/llm_cholec_organ/src')

from endopoint.models.gonogonet import load_gonogo_model
from endopoint.models.gonogonet_adapter import GoNoGoNetAdapter
from endopoint.datasets.cholec_gonogo import CholecGoNoGoAdapter
from endopoint.eval.bbox_evaluator import compute_mask_to_mask_iou
import torch


def test_gonogo_masks():
    """Test GoNoGoNet mask generation."""
    
    print("=" * 80)
    print("TESTING GONOGONET MASK GENERATION")
    print("=" * 80)
    
    # Load model directly
    print("\n1. Loading GoNoGoNet model...")
    model = load_gonogo_model(device='cuda')
    print("✓ Model loaded")
    
    # Load dataset
    print("\n2. Loading test image...")
    dataset = CholecGoNoGoAdapter()
    example = dataset.get_example('test', 0)
    image = example['image']
    gt_label = example['gonogo_label']
    print(f"✓ Image size: {image.size}")
    
    # Process image
    print("\n3. Processing image through model...")
    input_tensor = model.process_image(image, target_size=(640, 384))
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    # Get raw logits
    with torch.no_grad():
        logits = model(input_tensor)  # [1, n_classes, H, W]
    
    print(f"✓ Logits shape: {logits.shape}")
    print(f"  - Min logit: {logits.min().item():.3f}")
    print(f"  - Max logit: {logits.max().item():.3f}")
    
    # Convert to predictions
    pred_mask = torch.argmax(logits, dim=1)[0].cpu().numpy()
    print(f"✓ Prediction mask shape: {pred_mask.shape}")
    print(f"  - Unique classes: {np.unique(pred_mask).tolist()}")
    
    # Extract masks for each class
    print("\n4. Extracting class-specific masks...")
    for class_id, class_name in model.ID2LABEL.items():
        class_mask = (pred_mask == class_id)
        pixel_count = class_mask.sum()
        print(f"  - {class_name} (class {class_id}): {pixel_count} pixels")
        
        if class_id > 0 and pixel_count > 0:  # Non-background with pixels
            # Create binary mask for this class
            binary_mask = class_mask.astype(np.uint8)
            
            # Compare with ground truth
            gt_array = np.array(gt_label)
            gt_class_mask = (gt_array == class_id).astype(np.uint8)
            
            # Compute IoU
            iou = compute_mask_to_mask_iou(binary_mask, gt_class_mask)
            print(f"    IoU with GT: {iou:.3f}")
    
    # Test adapter
    print("\n5. Testing GoNoGoNetAdapter...")
    adapter = GoNoGoNetAdapter(verbose=True, use_cache=False, return_masks=True)
    
    prompt = """Detect the following organs in the image:
- Go Zone
- NoGo Zone"""
    
    responses = adapter([(prompt, image)], system_prompt="")
    response_json = json.loads(responses[0])
    
    print("\n✓ Adapter response:")
    for zone in ["Go Zone", "NoGo Zone"]:
        if zone in response_json:
            data = response_json[zone]
            print(f"  - {zone}:")
            print(f"    Present: {data['present']}")
            print(f"    Has bbox: {data['bbox'] is not None}")
            print(f"    Has mask: {'mask' in data}")
    
    # Check if full mask is included
    if "_full_mask" in response_json:
        full_mask_data = response_json["_full_mask"]
        print(f"\n  - Full mask included:")
        print(f"    Shape: {full_mask_data['shape']}")
        print(f"    Classes: {full_mask_data['classes']}")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    test_gonogo_masks()