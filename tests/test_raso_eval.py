#!/usr/bin/env python3
"""
Test RASO model integration with evaluation pipeline.
"""

import os
import sys
sys.path.insert(0, '/shared_data0/weiqiuy/llm_cholec_organ/src')

# Set environment variables for testing
os.environ['EVAL_DATASET'] = 'cholecseg8k'
os.environ['EVAL_MODEL'] = 'raso'
os.environ['EVAL_NUM_SAMPLES'] = '3'
os.environ['EVAL_USE_CACHE'] = 'false'  # Disable cache for testing
os.environ['EVAL_PERSISTENT_DIR'] = 'false'  # Use timestamped dir

from pathlib import Path
from endopoint.models import create_model
from endopoint.datasets.cholecseg8k_local import CholecSeg8kLocalAdapter
from PIL import Image

def test_raso_adapter():
    """Test RASO adapter directly."""
    
    print("=" * 60)
    print("Testing RASO Adapter")
    print("=" * 60)
    
    # Create RASO model adapter
    print("\n1. Creating RASO adapter...")
    model = create_model("raso", use_cache=False, verbose=True)
    print(f"   Model type: {type(model).__name__}")
    print(f"   Model ID: {model.model_id}")
    
    # Load a test image
    print("\n2. Loading test image...")
    data_dir = "/shared_data0/weiqiuy/datasets/cholecseg8k"
    adapter = CholecSeg8kLocalAdapter(data_dir=data_dir)
    
    # Get first test example
    example = adapter.get_example_by_global_index(0)
    # Example is a dict, not an object
    image = example['image']
    print(f"   Image size: {image.size}")
    
    # Create a test prompt (simulating bbox detection prompt)
    print("\n3. Testing organ detection...")
    test_prompt = """
    Detect the following organs in the surgical image and provide their bounding boxes:
    
    {
        "organs": ["liver", "gallbladder", "grasper", "fat", "blood"]
    }
    
    Return JSON with format:
    {
        "organ_name": {
            "present": true/false,
            "bbox": [x1, y1, x2, y2] or null
        }
    }
    """
    
    # Process through adapter
    system_prompt = "You are an expert medical image analyst."
    
    # Create batch with one query (text + image)
    batch = [(test_prompt, image)]
    
    # Run inference
    responses = model(batch, system_prompt=system_prompt)
    
    print("\n4. Response from RASO adapter:")
    print("-" * 40)
    print(responses[0])
    
    # Parse response
    import json
    try:
        result = json.loads(responses[0])
        print("\n5. Parsed results:")
        print("-" * 40)
        for organ, data in result.items():
            status = "✓ Present" if data.get('present') else "✗ Not detected"
            bbox = data.get('bbox', 'N/A')
            print(f"   {organ}: {status} (bbox: {bbox})")
    except Exception as e:
        print(f"Error parsing response: {e}")
    
    print("\n✅ RASO adapter test completed!")
    return True


def test_with_eval_pipeline():
    """Test RASO with the actual evaluation pipeline."""
    
    print("\n" + "=" * 60)
    print("Testing RASO with Evaluation Pipeline")
    print("=" * 60)
    
    from notebooks_py.eval_bbox_unified import main
    
    print("\nRunning evaluation with RASO model...")
    print("Configuration:")
    print(f"  Dataset: {os.environ['EVAL_DATASET']}")
    print(f"  Model: {os.environ['EVAL_MODEL']}")
    print(f"  Samples: {os.environ['EVAL_NUM_SAMPLES']}")
    
    try:
        # Run the main evaluation
        result = main()
        print("\n✅ Evaluation pipeline test completed!")
        return result == 0
    except Exception as e:
        print(f"\n❌ Error in evaluation pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Test 1: Direct adapter test
    success1 = test_raso_adapter()
    
    # Test 2: Full pipeline test
    print("\n" + "=" * 60)
    print("Do you want to run the full evaluation pipeline test? (y/n)")
    response = input().strip().lower()
    
    if response == 'y':
        success2 = test_with_eval_pipeline()
        
        if success1 and success2:
            print("\n🎉 All tests passed!")
        else:
            print("\n⚠️ Some tests failed")
    else:
        print("Skipping full pipeline test")
        
        if success1:
            print("\n🎉 Adapter test passed!")