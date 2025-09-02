#!/usr/bin/env python3
"""
Simple test of endopoint models and datasets.
Minimal dependencies, uses only endopoint modules.
"""

# Set multiprocessing start method for vLLM CUDA compatibility
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import torch
from PIL import Image, ImageDraw

# Use only endopoint modules
from endopoint.datasets import build_dataset
from endopoint.models import LLaVAModel, OpenAIAdapter


def create_synthetic_medical_image():
    """Create a synthetic medical-looking image for testing."""
    img = Image.new('RGB', (640, 480), color='#3a1f1f')  # Dark reddish background
    draw = ImageDraw.Draw(img)
    
    # Draw organ-like shapes
    # Large liver-like shape
    draw.ellipse([50, 100, 250, 280], fill='#8B4513', outline='#654321', width=2)
    
    # Smaller gallbladder-like shape  
    draw.ellipse([300, 150, 380, 220], fill='#228B22', outline='#1a5a1a', width=2)
    
    # Some fatty tissue
    draw.ellipse([420, 200, 520, 270], fill='#FFD700', outline='#DAA520', width=2)
    
    # Tool/grasper shape
    draw.polygon([(100, 350), (120, 330), (180, 340), (160, 360)], 
                 fill='#C0C0C0', outline='#808080', width=2)
    
    return img


def test_dataset_loading():
    """Test loading CholecSeg8k dataset via endopoint."""
    print("\n" + "="*60)
    print("Testing Dataset Loading")
    print("="*60)
    
    try:
        # Build dataset
        dataset = build_dataset("cholecseg8k")
        print("✅ Dataset loaded successfully")
        
        # Check dataset properties
        print(f"  Dataset tag: {dataset.dataset_tag}")
        print(f"  Version: {dataset.version}")
        print(f"  Number of classes: {len(dataset.label_ids)}")
        print(f"  Organ classes: {list(dataset.id2label.values())[:5]}...")
        
        # Try loading a sample
        split = "train"  # CholecSeg8k only has 'train' split
        total = dataset.total(split)
        print(f"  Total {split} samples: {total}")
        
        if total > 0:
            # Get first example
            example = dataset.get_example(split, 0)
            print(f"  ✅ Successfully loaded example 0")
            
            # Convert to tensors
            img_t, lab_t = dataset.example_to_tensors(example)
            print(f"    Image shape: {img_t.shape}")
            print(f"    Label shape: {lab_t.shape}")
            
            # Get presence vector
            presence = dataset.labels_to_presence_vector(lab_t)
            present_organs = [
                dataset.id2label[i] 
                for i, p in enumerate(presence) 
                if p > 0 and i in dataset.label_ids
            ]
            print(f"    Organs present: {present_organs}")
        
        return dataset
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return None


def test_vlm_model():
    """Test VLM model loading and inference."""
    print("\n" + "="*60)
    print("Testing VLM Model (LLaVA)")
    print("="*60)
    
    # Check CUDA
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    try:
        # Load model
        print("Loading LLaVA model...")
        model = LLaVAModel(
            model_name="llava-hf/llava-v1.6-mistral-7b-hf",
            use_cache=False,
            max_tokens=100,
            temperature=0.1,
            verbose=True,
            use_vllm=cuda_available  # Use vLLM if CUDA available
        )
        print("✅ Model loaded successfully")
        
        # Create test image
        test_image = create_synthetic_medical_image()
        print("Created synthetic test image")
        
        # Test 1: Simple description
        print("\nTest 1: Image description")
        prompt = "What do you see in this medical image?"
        response = model((test_image, prompt))
        print(f"Response: {response[:150]}...")
        
        # Test 2: JSON response
        print("\nTest 2: JSON organ detection")
        prompt = """Identify organs in this image.
Respond with JSON only: {"Liver": true/false, "Gallbladder": true/false}"""
        
        response = model((test_image, prompt))
        print(f"Raw response: {response}")
        
        # Try parsing JSON
        try:
            if "{" in response and "}" in response:
                json_str = response[response.find("{"):response.rfind("}")+1]
                parsed = json.loads(json_str)
                print(f"✅ Parsed JSON: {parsed}")
        except Exception as e:
            print(f"⚠️  JSON parse failed: {e}")
        
        # Clean up
        del model
        if cuda_available:
            torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ VLM model test failed: {e}")
        return False


def test_api_model():
    """Test API model adapter."""
    print("\n" + "="*60)
    print("Testing API Model Adapter (OpenAI)")
    print("="*60)
    
    try:
        # Load model adapter
        print("Loading OpenAI adapter...")
        model = OpenAIAdapter(
            model_name="gpt-4o-mini",
            use_cache=False,
            max_tokens=100,
            verbose=True
        )
        print("✅ Adapter loaded successfully")
        
        # Create test image
        test_image = create_synthetic_medical_image()
        
        # Test inference
        print("\nTesting inference...")
        prompt = "Describe this medical image briefly."
        system_prompt = "You are a medical image analyst. Be concise."
        
        # API adapters expect list of prompts
        responses = model([(test_image, prompt)], system_prompt=system_prompt)
        
        if responses and responses[0]:
            print(f"Response: {responses[0][:150]}...")
            print("✅ API model test successful")
            return True
        else:
            print("⚠️  Empty response (check API key)")
            return False
            
    except Exception as e:
        print(f"❌ API model test failed: {e}")
        if "API" in str(e):
            print("   (This is expected if API key is not set)")
        return False


def test_full_pipeline():
    """Test complete pipeline with dataset and model."""
    print("\n" + "="*60)
    print("Testing Full Pipeline")
    print("="*60)
    
    try:
        # Load dataset
        dataset = build_dataset("cholecseg8k")
        print("✅ Dataset loaded")
        
        # Get a sample
        example = dataset.get_example("train", 0)  # Use 'train' split
        image = example['image']
        img_t, lab_t = dataset.example_to_tensors(example)
        
        # Get ground truth
        presence = dataset.labels_to_presence_vector(lab_t)
        true_organs = [
            dataset.id2label[i]
            for i, p in enumerate(presence)
            if p > 0 and i in dataset.label_ids
        ]
        print(f"Ground truth organs: {true_organs}")
        
        # Create detection prompt
        organ_classes = [dataset.id2label[i] for i in dataset.label_ids if i != 0]
        prompt = f"""Identify which organs are visible: {', '.join(organ_classes[:5])}...
Respond with JSON: {{"Liver": true/false, "Gallbladder": true/false, ...}}"""
        
        # Try with VLM if CUDA available
        if torch.cuda.is_available():
            print("\nTesting with VLM model...")
            model = LLaVAModel(
                model_name="llava-hf/llava-v1.6-mistral-7b-hf",
                use_cache=False,
                max_tokens=200,
                temperature=0.0,
                use_vllm=True
            )
            
            response = model((image, prompt))
            print(f"Model response: {response[:200]}...")
            
            # Clean up
            del model
            torch.cuda.empty_cache()
        else:
            print("⚠️  Skipping VLM test (no CUDA)")
        
        print("✅ Pipeline test completed")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("ENDOPOINT MODULE TESTING")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Dataset loading
    dataset = test_dataset_loading()
    results["dataset"] = dataset is not None
    
    # Test 2: VLM model
    if torch.cuda.is_available():
        results["vlm"] = test_vlm_model()
    else:
        print("\n⏭️  Skipping VLM test (no CUDA)")
        results["vlm"] = None
    
    # Test 3: API model
    results["api"] = test_api_model()
    
    # Test 4: Full pipeline
    if results["dataset"]:
        results["pipeline"] = test_full_pipeline()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for test, result in results.items():
        if result is None:
            status = "⏭️  SKIPPED"
        elif result:
            status = "✅ PASSED"
        else:
            status = "❌ FAILED"
        print(f"{test.ljust(15)}: {status}")
    
    print("\n✨ Testing completed!")


if __name__ == "__main__":
    main()