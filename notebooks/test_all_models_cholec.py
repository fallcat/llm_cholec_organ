#!/usr/bin/env python3
"""
Comprehensive test of all models (API + VLM) on CholecSeg8k dataset.
Tests organ detection and pointing tasks across all available models.
"""

import sys
import json
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from PIL import Image
from llms import load_model
from cholecseg8k_utils import load_cholecseg8k_sample, get_balanced_sample_indices
import torch


# Define all models to test
MODEL_CONFIGS = {
    # API-based models
    "api_models": [
        {"name": "gpt-4o-mini", "type": "OpenAI GPT-4o Mini"},
        {"name": "gpt-4o", "type": "OpenAI GPT-4o"},
        {"name": "claude-3-5-sonnet-20241022", "type": "Claude 3.5 Sonnet"},
        {"name": "claude-3-haiku-20240307", "type": "Claude 3 Haiku"},
        {"name": "gemini-2.0-flash-exp", "type": "Google Gemini 2.0 Flash"},
        {"name": "gemini-1.5-pro", "type": "Google Gemini 1.5 Pro"},
    ],
    # Open-source VLM models
    "vlm_models": [
        {"name": "llava-hf/llava-v1.6-mistral-7b-hf", "type": "LLaVA 1.6 (7B)"},
        {"name": "llava-hf/llava-v1.6-vicuna-13b-hf", "type": "LLaVA 1.6 (13B)"},
        {"name": "Qwen/Qwen2.5-VL-7B-Instruct", "type": "Qwen2.5-VL (7B)"},
        {"name": "mistralai/Pixtral-12B-2409", "type": "Pixtral (12B)"},
        {"name": "deepseek-ai/deepseek-vl2", "type": "DeepSeek-VL2"},
    ]
}

# CholecSeg8k organ classes
ORGAN_CLASSES = [
    "Liver", "Gallbladder", "Hepatocystic Triangle", "Fat",
    "Grasper", "Connective Tissue", "Blood", "Cystic Artery",
    "Cystic Vein", "Cystic Pedicle", "Gallbladder Plate", "Abdominal Wall"
]


def create_organ_detection_prompt():
    """Create a prompt for organ detection task."""
    return f"""You are analyzing a laparoscopic cholecystectomy surgery image. 
Identify which of the following organs/structures are visible in the image:

{', '.join(ORGAN_CLASSES)}

Respond ONLY with a JSON object where keys are organ names and values are boolean (true if present, false if not):
{{"Liver": true/false, "Gallbladder": true/false, ...}}"""


def create_pointing_prompt(organ_name, image_width, image_height):
    """Create a prompt for pointing task."""
    return f"""Look at this laparoscopic surgery image. 
If you can see a {organ_name}, point to its center location.

Image dimensions: {image_width}x{image_height} pixels

Respond ONLY with JSON:
{{"present": true/false, "x": <x-coordinate or null>, "y": <y-coordinate or null>}}

If the organ is not visible, set present to false and x,y to null."""


def test_model_on_sample(model, model_name, sample, verbose=True):
    """Test a single model on a CholecSeg8k sample."""
    results = {
        "model": model_name,
        "sample_id": sample.get("id", "unknown"),
        "detection": {},
        "pointing": {},
        "timing": {},
        "errors": []
    }
    
    image = sample['image']
    true_labels = sample.get('class_labels', [])
    masks = sample.get('masks', {})
    
    if verbose:
        print(f"\n  Testing on sample with organs: {', '.join(true_labels)}")
    
    # Test 1: Organ Detection
    try:
        start_time = time.time()
        detection_prompt = create_organ_detection_prompt()
        
        # Add system prompt for better accuracy
        system_prompt = """You are an expert medical image analyst specializing in laparoscopic surgery.
Analyze images carefully and provide accurate organ identification.
Always respond with valid JSON only."""
        
        response = model((image, detection_prompt), system_prompt=system_prompt)
        results["timing"]["detection"] = time.time() - start_time
        
        # Parse response
        try:
            # Try to extract JSON from response
            if "{" in response and "}" in response:
                json_str = response[response.find("{"):response.rfind("}")+1]
                detection_result = json.loads(json_str)
            else:
                detection_result = json.loads(response)
            
            results["detection"]["response"] = detection_result
            
            # Calculate accuracy
            correct = 0
            total = 0
            for organ in ORGAN_CLASSES:
                predicted = detection_result.get(organ, False)
                actual = organ in true_labels
                if predicted == actual:
                    correct += 1
                total += 1
            
            results["detection"]["accuracy"] = correct / total
            results["detection"]["success"] = True
            
            if verbose:
                print(f"    Detection accuracy: {correct}/{total} = {100*correct/total:.1f}%")
                detected = [k for k, v in detection_result.items() if v]
                print(f"    Detected: {', '.join(detected[:5])}...")
                
        except (json.JSONDecodeError, KeyError) as e:
            results["detection"]["success"] = False
            results["detection"]["error"] = str(e)
            if verbose:
                print(f"    Detection failed to parse JSON: {e}")
                print(f"    Raw response: {response[:200]}...")
    
    except Exception as e:
        results["errors"].append(f"Detection error: {str(e)}")
        if verbose:
            print(f"    Detection error: {e}")
    
    # Test 2: Pointing Task (test on first available organ)
    test_organ = true_labels[0] if true_labels else "Gallbladder"
    
    try:
        start_time = time.time()
        pointing_prompt = create_pointing_prompt(test_organ, image.width, image.height)
        
        response = model((image, pointing_prompt), system_prompt=system_prompt)
        results["timing"]["pointing"] = time.time() - start_time
        
        # Parse response
        try:
            if "{" in response and "}" in response:
                json_str = response[response.find("{"):response.rfind("}")+1]
                pointing_result = json.loads(json_str)
            else:
                pointing_result = json.loads(response)
            
            results["pointing"]["response"] = pointing_result
            results["pointing"]["target_organ"] = test_organ
            results["pointing"]["success"] = True
            
            # Check if pointing is correct (if we have mask data)
            if pointing_result.get("present") and test_organ in masks:
                x = pointing_result.get("x")
                y = pointing_result.get("y")
                
                if x is not None and y is not None:
                    # Check if point is within organ mask
                    mask = masks[test_organ]
                    if 0 <= int(y) < mask.shape[0] and 0 <= int(x) < mask.shape[1]:
                        hit = mask[int(y), int(x)] > 0
                        results["pointing"]["hit"] = hit
                        if verbose:
                            print(f"    Pointing at {test_organ}: ({x:.0f}, {y:.0f}) - {'HIT' if hit else 'MISS'}")
                    else:
                        results["pointing"]["hit"] = False
                        if verbose:
                            print(f"    Pointing out of bounds: ({x}, {y})")
            else:
                if verbose:
                    present = pointing_result.get("present", False)
                    print(f"    Pointing: organ {'detected' if present else 'not detected'}")
                    
        except (json.JSONDecodeError, KeyError) as e:
            results["pointing"]["success"] = False
            results["pointing"]["error"] = str(e)
            if verbose:
                print(f"    Pointing failed to parse JSON: {e}")
    
    except Exception as e:
        results["errors"].append(f"Pointing error: {str(e)}")
        if verbose:
            print(f"    Pointing error: {e}")
    
    return results


def test_all_models(num_samples=3, test_api=True, test_vlm=True):
    """Test all available models on CholecSeg8k samples."""
    
    print("=" * 80)
    print("COMPREHENSIVE MODEL TESTING ON CHOLECSEG8K")
    print("=" * 80)
    
    # Check CUDA availability for VLM models
    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA Available: {cuda_available}")
    if cuda_available:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load test samples
    print(f"\nLoading {num_samples} test samples from CholecSeg8k...")
    samples = []
    
    try:
        # Try to get balanced samples
        indices = get_balanced_sample_indices(num_samples_per_organ=1)[:num_samples]
        for idx in indices:
            sample = load_cholecseg8k_sample(idx)
            samples.append(sample)
        print(f"✓ Loaded {len(samples)} samples")
    except Exception as e:
        print(f"⚠ Could not load from dataset: {e}")
        print("  Creating synthetic test samples...")
        
        # Create synthetic samples for testing
        for i in range(num_samples):
            img = Image.new('RGB', (854, 480), color=(100, 50, 50))
            # Add some variation
            pixels = img.load()
            for x in range(100, 200):
                for y in range(100, 200):
                    pixels[x, y] = (200, 100, 100)
            
            samples.append({
                'id': f'synthetic_{i}',
                'image': img,
                'class_labels': ['Liver', 'Gallbladder'] if i % 2 == 0 else ['Fat', 'Grasper'],
                'masks': {}
            })
    
    all_results = []
    
    # Test API models
    if test_api:
        print("\n" + "=" * 40)
        print("TESTING API MODELS")
        print("=" * 40)
        
        for model_config in MODEL_CONFIGS["api_models"]:
            model_name = model_config["name"]
            model_type = model_config["type"]
            
            print(f"\n{'='*60}")
            print(f"Testing: {model_type}")
            print(f"Model: {model_name}")
            print(f"{'='*60}")
            
            try:
                # Load model
                print("Loading model...")
                model = load_model(
                    model_name,
                    use_cache=False,
                    max_tokens=300,
                    temperature=0.1,
                    verbose=False
                )
                print("✓ Model loaded")
                
                # Test on each sample
                for i, sample in enumerate(samples):
                    print(f"\nSample {i+1}/{len(samples)}:")
                    result = test_model_on_sample(model, model_name, sample, verbose=True)
                    result["model_type"] = "API"
                    all_results.append(result)
                
            except Exception as e:
                print(f"❌ Failed to test {model_name}: {e}")
                all_results.append({
                    "model": model_name,
                    "model_type": "API",
                    "error": str(e),
                    "success": False
                })
    
    # Test VLM models
    if test_vlm and cuda_available:
        print("\n" + "=" * 40)
        print("TESTING VLM MODELS")
        print("=" * 40)
        
        for model_config in MODEL_CONFIGS["vlm_models"]:
            model_name = model_config["name"]
            model_type = model_config["type"]
            
            print(f"\n{'='*60}")
            print(f"Testing: {model_type}")
            print(f"Model: {model_name}")
            print(f"{'='*60}")
            
            try:
                # Load model
                print("Loading model...")
                
                # Special handling for different models
                kwargs = {
                    "use_cache": False,
                    "max_tokens": 300,
                    "temperature": 0.1,
                    "verbose": True
                }
                
                # Use vLLM for supported models
                if "llava" in model_name.lower() or "qwen" in model_name.lower() or "pixtral" in model_name.lower():
                    kwargs["use_vllm"] = True
                
                model = load_model(model_name, **kwargs)
                print("✓ Model loaded")
                
                # Test on first sample only (VLMs are slower)
                sample = samples[0]
                print(f"\nTesting on sample:")
                result = test_model_on_sample(model, model_name, sample, verbose=True)
                result["model_type"] = "VLM"
                all_results.append(result)
                
                # Free memory
                del model
                if cuda_available:
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"❌ Failed to test {model_name}: {e}")
                all_results.append({
                    "model": model_name,
                    "model_type": "VLM",
                    "error": str(e),
                    "success": False
                })
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    # Group results by model
    model_summary = {}
    for result in all_results:
        model = result["model"]
        if model not in model_summary:
            model_summary[model] = {
                "type": result.get("model_type", "Unknown"),
                "detection_acc": [],
                "pointing_success": [],
                "errors": []
            }
        
        if "detection" in result and result["detection"].get("success"):
            if "accuracy" in result["detection"]:
                model_summary[model]["detection_acc"].append(result["detection"]["accuracy"])
        
        if "pointing" in result and result["pointing"].get("success"):
            model_summary[model]["pointing_success"].append(1)
        elif "pointing" in result:
            model_summary[model]["pointing_success"].append(0)
        
        if "error" in result or result.get("errors"):
            model_summary[model]["errors"].append(result.get("error", result.get("errors", [])))
    
    # Print summary table
    print("\n{:<30} {:<10} {:<20} {:<20}".format("Model", "Type", "Avg Detection Acc", "Pointing Success"))
    print("-" * 80)
    
    for model, stats in model_summary.items():
        model_short = model.split("/")[-1][:28]
        
        if stats["detection_acc"]:
            det_acc = f"{100*np.mean(stats['detection_acc']):.1f}%"
        else:
            det_acc = "N/A"
        
        if stats["pointing_success"]:
            point_success = f"{100*np.mean(stats['pointing_success']):.1f}%"
        else:
            point_success = "N/A"
        
        status = "❌" if stats["errors"] else "✓"
        
        print(f"{status} {model_short:<28} {stats['type']:<10} {det_acc:<20} {point_success:<20}")
    
    print("\n✨ Testing completed!")
    
    # Save results
    import pickle
    results_file = Path(__file__).parent / "model_test_results.pkl"
    with open(results_file, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nResults saved to: {results_file}")
    
    return all_results


def quick_test_single_model(model_name="llava-hf/llava-v1.6-mistral-7b-hf"):
    """Quick test of a single model for debugging."""
    
    print(f"Quick test of {model_name}")
    print("=" * 60)
    
    # Create a simple test image
    image = Image.new('RGB', (640, 480), color='darkred')
    pixels = image.load()
    # Add a bright region (simulating organ)
    for x in range(200, 400):
        for y in range(150, 300):
            pixels[x, y] = (255, 200, 200)
    
    try:
        # Load model
        print("Loading model...")
        model = load_model(
            model_name,
            use_cache=False,
            max_tokens=200,
            temperature=0.1,
            verbose=True
        )
        print("✓ Model loaded")
        
        # Simple test
        print("\nTest 1: Basic image description")
        response = model((image, "Describe what you see in this medical image."))
        print(f"Response: {response[:200]}...")
        
        # Organ detection test
        print("\nTest 2: Organ detection")
        detection_prompt = "List any organs you can identify in this image. Respond with: Liver, Gallbladder, or None."
        response = model((image, detection_prompt))
        print(f"Response: {response[:200]}...")
        
        # JSON response test
        print("\nTest 3: JSON response")
        json_prompt = 'Respond with JSON only: {"visible": true/false, "organs": ["list", "of", "organs"]}'
        response = model((image, json_prompt))
        print(f"Response: {response}")
        
        try:
            if "{" in response and "}" in response:
                json_str = response[response.find("{"):response.rfind("}")+1]
                parsed = json.loads(json_str)
                print(f"✓ Parsed JSON: {parsed}")
        except:
            print("⚠ Could not parse JSON")
        
        print("\n✓ Quick test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test models on CholecSeg8k")
    parser.add_argument("--quick", action="store_true", help="Run quick test only")
    parser.add_argument("--model", type=str, help="Specific model to test")
    parser.add_argument("--samples", type=int, default=2, help="Number of samples to test")
    parser.add_argument("--api", action="store_true", help="Test API models only")
    parser.add_argument("--vlm", action="store_true", help="Test VLM models only")
    
    args = parser.parse_args()
    
    if args.quick or args.model:
        # Quick test mode
        model = args.model or "llava-hf/llava-v1.6-mistral-7b-hf"
        quick_test_single_model(model)
    else:
        # Full test mode
        test_api = True
        test_vlm = True
        
        if args.api and not args.vlm:
            test_vlm = False
        elif args.vlm and not args.api:
            test_api = False
        
        test_all_models(
            num_samples=args.samples,
            test_api=test_api,
            test_vlm=test_vlm
        )