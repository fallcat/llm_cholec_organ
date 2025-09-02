#!/usr/bin/env python3
"""
Simple test script for VLM models on CholecSeg8k.
Focuses on testing open-source vision-language models.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from llms import load_model


def create_test_image_with_annotations():
    """Create a test image with clear visual markers."""
    # Create base image
    img = Image.new('RGB', (800, 600), color='#2c1810')  # Dark brown background
    draw = ImageDraw.Draw(img)
    
    # Draw organ-like shapes with labels
    # Liver (large reddish-brown shape)
    draw.ellipse([50, 100, 350, 300], fill='#8B4513', outline='#654321', width=3)
    draw.text((180, 190), "LIVER", fill='white')
    
    # Gallbladder (small green shape)
    draw.ellipse([400, 200, 500, 280], fill='#228B22', outline='#006400', width=2)
    draw.text((420, 235), "GB", fill='white')
    
    # Fat tissue (yellow patches)
    draw.ellipse([550, 150, 650, 220], fill='#FFD700', outline='#FFA500', width=2)
    draw.text((575, 180), "FAT", fill='black')
    
    # Instrument (gray metallic)
    draw.polygon([(100, 400), (120, 380), (200, 390), (180, 410)], 
                 fill='#C0C0C0', outline='#808080', width=2)
    draw.text((130, 390), "TOOL", fill='black')
    
    # Add coordinate grid for pointing tests
    for x in range(0, 800, 100):
        draw.line([(x, 0), (x, 600)], fill='#444444', width=1)
    for y in range(0, 600, 100):
        draw.line([(0, y), (800, y)], fill='#444444', width=1)
    
    return img


def test_vlm_model(model_name, test_complex=True):
    """Test a single VLM model with various prompts."""
    
    print(f"\n{'='*70}")
    print(f"Testing: {model_name}")
    print(f"{'='*70}")
    
    # Check CUDA
    cuda_available = torch.cuda.is_available()
    if not cuda_available and "llava" in model_name.lower():
        print("⚠️  Warning: CUDA not available, model may run slowly or fail")
    
    try:
        # Determine model loading parameters
        load_params = {
            "use_cache": False,
            "max_tokens": 150,
            "temperature": 0.1,
            "verbose": True
        }
        
        # Model-specific parameters
        if "llava" in model_name.lower():
            load_params["use_vllm"] = cuda_available  # Use vLLM only if CUDA available
        elif "qwen" in model_name.lower():
            load_params["use_vllm"] = cuda_available
        elif "pixtral" in model_name.lower():
            load_params["use_vllm"] = True  # Pixtral requires vLLM
        elif "deepseek" in model_name.lower():
            load_params["use_vllm"] = False  # DeepSeek doesn't use vLLM
        
        # Load model
        print(f"\n1. Loading model with params: {load_params}")
        model = load_model(model_name, **load_params)
        print("   ✅ Model loaded successfully")
        
        # Create test image
        test_image = create_test_image_with_annotations()
        print(f"\n2. Created test image: {test_image.size}")
        
        # Save test image for inspection
        test_image_path = Path(__file__).parent / f"test_image_{model_name.replace('/', '_')}.png"
        test_image.save(test_image_path)
        print(f"   Saved to: {test_image_path}")
        
        # Test 1: Simple description
        print("\n3. Test: Simple Description")
        print("   Prompt: 'What do you see in this image?'")
        
        response = model(
            (test_image, "What do you see in this image?"),
            system_prompt="You are analyzing a medical image. Be concise."
        )
        print(f"   Response: {response[:200]}...")
        
        # Test 2: Organ detection
        print("\n4. Test: Organ Detection")
        prompt = """Identify these organs if present: Liver, Gallbladder, Fat, Tool.
Respond with JSON only: {"Liver": true/false, "Gallbladder": true/false, "Fat": true/false, "Tool": true/false}"""
        
        print(f"   Prompt: {prompt[:80]}...")
        response = model((test_image, prompt))
        print(f"   Raw response: {response}")
        
        # Try to parse JSON
        try:
            if "{" in response and "}" in response:
                json_str = response[response.find("{"):response.rfind("}")+1]
                parsed = json.loads(json_str)
                print(f"   ✅ Parsed JSON: {parsed}")
                
                # Check accuracy (we know what's in our test image)
                expected = {"Liver": True, "Gallbladder": True, "Fat": True, "Tool": True}
                correct = sum(1 for k in expected if parsed.get(k) == expected[k])
                print(f"   Accuracy: {correct}/4 organs correct")
            else:
                print("   ⚠️  No JSON found in response")
        except json.JSONDecodeError as e:
            print(f"   ⚠️  JSON parse error: {e}")
        
        # Test 3: Pointing task (if complex tests enabled)
        if test_complex:
            print("\n5. Test: Pointing Task")
            prompt = """Point to the center of the Liver.
Image size is 800x600 pixels.
Respond with JSON only: {"x": <number>, "y": <number>}"""
            
            print(f"   Prompt: {prompt[:80]}...")
            response = model((test_image, prompt))
            print(f"   Raw response: {response}")
            
            try:
                if "{" in response and "}" in response:
                    json_str = response[response.find("{"):response.rfind("}")+1]
                    parsed = json.loads(json_str)
                    x, y = parsed.get("x"), parsed.get("y")
                    print(f"   ✅ Coordinates: ({x}, {y})")
                    
                    # Check if pointing is reasonable (Liver is at roughly 200, 200)
                    if x is not None and y is not None:
                        dist = np.sqrt((x - 200)**2 + (y - 200)**2)
                        if dist < 150:
                            print(f"   ✅ Good pointing! Distance from center: {dist:.1f} pixels")
                        else:
                            print(f"   ⚠️  Far from expected center. Distance: {dist:.1f} pixels")
            except Exception as e:
                print(f"   ⚠️  Pointing parse error: {e}")
        
        print(f"\n✅ Successfully tested {model_name}")
        
        # Clean up
        del model
        if cuda_available:
            torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error testing {model_name}:")
        print(f"   {type(e).__name__}: {e}")
        
        # More detailed error for import issues
        if "Import" in str(e) or "module" in str(e):
            print("\n   Possible fixes:")
            if "vllm" in str(e):
                print("   - Install vLLM: pip install vllm")
            if "transformers" in str(e):
                print("   - Install transformers: pip install transformers")
            if "deepseek" in str(e):
                print("   - Install deepseek_vl2: pip install deepseek-vl2")
        
        return False


def main():
    """Test all VLM models."""
    
    print("=" * 70)
    print("VLM MODEL TESTING SUITE")
    print("=" * 70)
    
    # Check environment
    print("\nEnvironment Check:")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # Define models to test
    vlm_models = [
        # Start with smaller/faster models
        "llava-hf/llava-v1.6-mistral-7b-hf",
        # "llava-hf/llava-v1.6-vicuna-13b-hf",  # Larger variant
        # "Qwen/Qwen2.5-VL-7B-Instruct",
        # "mistralai/Pixtral-12B-2409",
        # "deepseek-ai/deepseek-vl2",
    ]
    
    print(f"\nModels to test: {len(vlm_models)}")
    for model in vlm_models:
        print(f"  - {model}")
    
    results = {}
    
    # Test each model
    for model_name in vlm_models:
        success = test_vlm_model(model_name, test_complex=True)
        results[model_name] = success
        
        # Add delay between models to avoid memory issues
        import time
        time.sleep(2)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for model, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {model}")
    
    success_rate = sum(results.values()) / len(results) * 100
    print(f"\nSuccess rate: {success_rate:.0f}% ({sum(results.values())}/{len(results)})")
    
    print("\n✨ Testing completed!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test VLM models")
    parser.add_argument("--model", type=str, help="Test specific model only")
    parser.add_argument("--simple", action="store_true", help="Skip complex tests")
    
    args = parser.parse_args()
    
    if args.model:
        # Test single model
        test_vlm_model(args.model, test_complex=not args.simple)
    else:
        # Test all models
        main()