#!/usr/bin/env python3
"""
Test script to verify RASO model works correctly with CholecSeg8k dataset.
"""

import sys
sys.path.insert(0, '/shared_data0/weiqiuy/llm_cholec_organ/src')

from endopoint.models.raso import load_raso_cholecseg8k
from pathlib import Path
from PIL import Image

def test_raso_cholecseg8k():
    """Test RASO model on CholecSeg8k images."""
    
    print("=" * 60)
    print("Testing RASO Model for CholecSeg8k")
    print("=" * 60)
    
    # Load the CholecSeg8k-specific RASO model
    print("\n1. Loading RASO model for CholecSeg8k...")
    model = load_raso_cholecseg8k()
    
    # Test image path (same as in notebook)
    test_image = "/shared_data0/weiqiuy/real_drs/data/abdomen_exlib/images/cholec80_video20_006.png"
    
    if not Path(test_image).exists():
        print(f"Error: Test image not found at {test_image}")
        # Try alternative paths
        alt_paths = [
            "/shared_data0/weiqiuy/datasets/cholecseg8k/train/images/video01_00040.png",
            "/shared_data0/weiqiuy/datasets/cholecseg8k/train/images/video01_00080.png"
        ]
        for alt_path in alt_paths:
            if Path(alt_path).exists():
                test_image = alt_path
                print(f"Using alternative image: {test_image}")
                break
    
    print(f"\n2. Testing on image: {Path(test_image).name}")
    print("-" * 40)
    
    # Run inference
    output = model.analyze_image(test_image)
    print(f"Raw output: {output}")
    
    # Parse output
    organs = model.parse_output(output)
    print(f"Parsed organs: {organs}")
    
    # Test with different thresholds
    print("\n3. Testing with multiple thresholds:")
    print("-" * 40)
    
    thresholds = [0.5, 0.65, 0.8]
    results = model.analyze_with_multiple_thresholds(test_image, thresholds)
    
    for threshold, detected_organs in results.items():
        print(f"Threshold {threshold}: {detected_organs}")
    
    # Test organ presence check
    print("\n4. Checking organ presence (example):")
    print("-" * 40)
    
    # Expected organs in CholecSeg8k
    cholecseg8k_organs = [
        "abdominal wall", "liver", "gastrointestinal tract", "fat",
        "grasper", "connective tissue", "blood", "cystic artery",
        "l-hook electrocautery", "gallbladder", "hepatocystic triangle",
        "liver ligament"
    ]
    
    # Check which organs are detected
    detected_set = set(organs)
    for organ in ["liver", "gallbladder", "grasper", "fat"]:
        is_present = organ in detected_set
        print(f"  {organ}: {'✓ Present' if is_present else '✗ Not detected'}")
    
    print("\n5. Important notes about RASO:")
    print("-" * 40)
    print("• RASO only provides organ presence detection")
    print("• It does NOT provide bounding boxes")
    print("• Output format: 'organ1 | organ2 | organ3'")
    print("• Organs are separated by '|' character")
    
    print("\n✅ Test completed successfully!")
    
    return model, organs


def test_batch_processing():
    """Test batch processing capability."""
    
    print("\n" + "=" * 60)
    print("Testing Batch Processing")
    print("=" * 60)
    
    model = load_raso_cholecseg8k()
    
    # Find available test images
    test_images = []
    base_path = Path("/shared_data0/weiqiuy/datasets/cholecseg8k/train/images")
    
    if base_path.exists():
        # Get first 3 images
        for img_path in sorted(base_path.glob("*.png"))[:3]:
            test_images.append(str(img_path))
    
    if not test_images:
        print("No test images found for batch processing test")
        return
    
    print(f"\nProcessing {len(test_images)} images in batch...")
    
    results = model.batch_analyze(test_images)
    
    for i, (img_path, organs) in enumerate(zip(test_images, results)):
        print(f"\nImage {i+1}: {Path(img_path).name}")
        print(f"  Detected organs: {organs}")
    
    print("\n✅ Batch processing test completed!")


if __name__ == "__main__":
    # Run main test
    model, organs = test_raso_cholecseg8k()
    
    # Run batch test if possible
    test_batch_processing()