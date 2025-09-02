#!/usr/bin/env python3
"""
Example script to test VLM models on CholecSeg8k dataset for organ detection.
This demonstrates how to use the open-source VLM models with the cholec evaluation pipeline.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import numpy as np
from PIL import Image
from llms import load_model
from cholec import CholecExample
from cholecseg8k_utils import load_cholecseg8k_sample


def test_vlm_on_cholec():
    """Test VLM models on a sample from CholecSeg8k dataset."""
    
    print("=" * 60)
    print("Testing VLM Models on CholecSeg8k Dataset")
    print("=" * 60)
    
    # Load a sample from the dataset
    print("\n1. Loading sample from CholecSeg8k dataset...")
    try:
        # Load the first sample
        sample = load_cholecseg8k_sample(0)
        image = sample['image']
        masks = sample['masks']
        class_labels = sample['class_labels']
        
        print(f"✓ Loaded sample with {len(class_labels)} organ annotations")
        print(f"  Image size: {image.size}")
        print(f"  Organs present: {', '.join(class_labels)}")
    except Exception as e:
        print(f"Note: Could not load from dataset ({e}), creating synthetic example...")
        # Create a synthetic example for testing
        image = Image.new('RGB', (854, 480), color='white')
        class_labels = ['Liver', 'Gallbladder']
    
    # Define VLM models to test
    vlm_models = [
        "llava-hf/llava-v1.6-mistral-7b-hf",
        # "Qwen/Qwen2.5-VL-7B-Instruct",  # Uncomment if model is available
        # "mistralai/Pixtral-12B-2409",   # Uncomment if model is available
        # "deepseek-ai/deepseek-vl2",     # Uncomment if model is available
    ]
    
    # Test prompt for organ detection
    organ_detection_prompt = """Look at this laparoscopic surgery image and identify which organs are visible.

For each of the following organs, indicate if it is present in the image:
- Liver
- Gallbladder
- Hepatocystic Triangle
- Fat
- Grasper
- Connective Tissue
- Blood
- Cystic Artery
- Cystic Vein
- Cystic Pedicle
- Gallbladder Plate
- Abdominal Wall

Respond in JSON format like this:
{
  "Liver": true/false,
  "Gallbladder": true/false,
  ...
}"""
    
    # Test pointing prompt
    pointing_prompt = """Look at this laparoscopic surgery image. If you can see a Gallbladder, point to its center location.

Respond in JSON format:
{
  "present": true/false,
  "x": <x-coordinate between 0 and image width>,
  "y": <y-coordinate between 0 and image height>
}

If the organ is not present, set x and y to null."""
    
    results = {}
    
    # Test each VLM model
    for model_name in vlm_models:
        print(f"\n2. Testing {model_name}...")
        
        try:
            # Load the model
            print(f"   Loading model...")
            model = load_model(
                model_name,
                use_cache=False,  # Disable cache for testing
                max_tokens=200,
                temperature=0.1,
                verbose=True
            )
            print(f"   ✓ Model loaded successfully")
            
            # Test organ detection
            print(f"   Testing organ detection...")
            detection_response = model((image, organ_detection_prompt))
            print(f"   Response: {detection_response[:200]}...")
            
            # Try to parse JSON response
            try:
                detection_result = json.loads(detection_response)
                print(f"   ✓ Successfully parsed detection response")
                print(f"   Detected organs: {[k for k, v in detection_result.items() if v]}")
            except json.JSONDecodeError:
                print(f"   ⚠ Could not parse JSON response")
            
            # Test pointing task
            print(f"   Testing pointing task...")
            pointing_response = model((image, pointing_prompt))
            print(f"   Response: {pointing_response[:200]}...")
            
            # Try to parse pointing response
            try:
                pointing_result = json.loads(pointing_response)
                print(f"   ✓ Successfully parsed pointing response")
                if pointing_result.get('present'):
                    print(f"   Pointing location: ({pointing_result.get('x')}, {pointing_result.get('y')})")
                else:
                    print(f"   Organ not detected for pointing")
            except json.JSONDecodeError:
                print(f"   ⚠ Could not parse JSON response")
            
            # Store results
            results[model_name] = {
                'detection': detection_response,
                'pointing': pointing_response,
                'success': True
            }
            
        except Exception as e:
            print(f"   ❌ Error testing model: {e}")
            results[model_name] = {
                'error': str(e),
                'success': False
            }
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for model_name, result in results.items():
        if result['success']:
            print(f"✅ {model_name}: Successfully tested")
        else:
            print(f"❌ {model_name}: Failed - {result.get('error', 'Unknown error')}")
    
    print("\n✨ Testing completed!")
    
    return results


def test_vlm_with_system_prompt():
    """Test VLM models with different system prompts for better accuracy."""
    
    print("\n" + "=" * 60)
    print("Testing VLM Models with System Prompts")
    print("=" * 60)
    
    # Create a simple test image
    image = Image.new('RGB', (854, 480), color='white')
    
    # System prompt for medical image analysis
    system_prompt = """You are an expert medical image analyst specializing in laparoscopic cholecystectomy procedures. 
You have extensive knowledge of abdominal anatomy and can accurately identify organs and structures in surgical images.
Always respond with precise, structured information."""
    
    # Test with LLaVA model
    model_name = "llava-hf/llava-v1.6-mistral-7b-hf"
    
    print(f"\nTesting {model_name} with system prompt...")
    
    try:
        model = load_model(
            model_name,
            use_cache=False,
            max_tokens=150,
            temperature=0.1
        )
        
        # Test with system prompt
        prompt = "Describe any medical structures you can identify in this image."
        response = model(
            (image, prompt),
            system_prompt=system_prompt
        )
        
        print(f"Response with system prompt: {response[:200]}...")
        print("✓ System prompt test successful")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    # Run the main test
    results = test_vlm_on_cholec()
    
    # Run system prompt test
    test_vlm_with_system_prompt()
    
    print("\n🎉 All tests completed!")