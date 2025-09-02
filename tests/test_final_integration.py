#!/usr/bin/env python3
"""Final integration test with CholecSeg8k dataset and VLM models."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import json
import torch
from PIL import Image

# Import from endopoint only
from endopoint.datasets import build_dataset
from endopoint.models import LLaVAModel

def test_cholec_with_llava():
    """Test LLaVA on actual CholecSeg8k samples."""
    print("="*70)
    print("FINAL INTEGRATION TEST: LLaVA + CholecSeg8k")
    print("="*70)
    
    print(f"\nEnvironment:")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    try:
        # 1. Load dataset
        print("\n1. Loading CholecSeg8k dataset...")
        dataset = build_dataset("cholecseg8k")
        print(f"✅ Dataset loaded: {dataset.dataset_tag}")
        
        # Get organ classes
        organ_classes = [
            dataset.id2label[i] 
            for i in dataset.label_ids 
            if i != 0
        ][:6]  # Just first 6 for testing
        print(f"   Organ classes: {organ_classes}")
        
        # 2. Load model
        print("\n2. Loading LLaVA model...")
        model = LLaVAModel(
            model_name="llava-hf/llava-v1.6-mistral-7b-hf",
            use_vllm=False,  # Use transformers for stability
            use_cache=False,
            max_tokens=100,
            temperature=0.0,  # Deterministic
            verbose=True
        )
        print("✅ Model loaded")
        
        # 3. Get a sample from dataset
        print("\n3. Loading sample from dataset...")
        example = dataset.get_example("train", 0)
        image = example['image']
        
        # Convert to tensors to get ground truth
        img_t, lab_t = dataset.example_to_tensors(example)
        presence = dataset.labels_to_presence_vector(lab_t)
        true_organs = [
            dataset.id2label[i]
            for i, p in enumerate(presence)
            if p > 0 and i in dataset.label_ids
        ]
        print(f"   Ground truth organs: {true_organs}")
        
        # 4. Test organ detection
        print("\n4. Testing organ detection...")
        detection_prompt = f"""Look at this laparoscopic surgery image.
Which of these organs are visible: {', '.join(organ_classes)}?

Respond with JSON only:
{{"Liver": true/false, "Gallbladder": true/false, ...}}"""
        
        system_prompt = "You are a medical image analyst. Respond with valid JSON only."
        
        response = model((image, detection_prompt), system_prompt=system_prompt)
        print(f"   Raw response: {response[:200]}...")
        
        # Try to parse JSON
        try:
            if "{" in response and "}" in response:
                json_str = response[response.find("{"):response.rfind("}")+1]
                predictions = json.loads(json_str)
                print(f"✅ Parsed predictions: {predictions}")
                
                # Calculate accuracy
                correct = 0
                for organ in organ_classes:
                    predicted = predictions.get(organ, False)
                    actual = organ in true_organs
                    if predicted == actual:
                        correct += 1
                
                accuracy = correct / len(organ_classes)
                print(f"   Accuracy: {correct}/{len(organ_classes)} = {accuracy:.1%}")
                
        except json.JSONDecodeError as e:
            print(f"⚠️  JSON parse error: {e}")
        
        # 5. Test pointing
        print("\n5. Testing pointing task...")
        if true_organs:
            target_organ = true_organs[0]
            pointing_prompt = f"""Point to the center of the {target_organ}.
Image size: {image.width}x{image.height} pixels.
Respond with JSON: {{"present": true/false, "x": number, "y": number}}"""
            
            response = model((image, pointing_prompt), system_prompt=system_prompt)
            print(f"   Raw response: {response}")
            
            try:
                if "{" in response and "}" in response:
                    json_str = response[response.find("{"):response.rfind("}")+1]
                    pointing = json.loads(json_str)
                    print(f"✅ Pointing result: {pointing}")
            except json.JSONDecodeError as e:
                print(f"⚠️  JSON parse error: {e}")
        
        print("\n✅ Integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up
        if 'model' in locals():
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("\nCleanup complete")


def test_multiple_samples(num_samples=3):
    """Test on multiple samples."""
    print("\n" + "="*70)
    print(f"TESTING ON {num_samples} SAMPLES")
    print("="*70)
    
    try:
        # Load dataset and model once
        dataset = build_dataset("cholecseg8k")
        model = LLaVAModel(
            model_name="llava-hf/llava-v1.6-mistral-7b-hf",
            use_vllm=False,
            use_cache=False,
            max_tokens=50,
            temperature=0.0,
            verbose=False
        )
        
        organ_classes = ["Liver", "Gallbladder", "Fat", "Grasper"]
        
        successes = 0
        for i in range(num_samples):
            print(f"\nSample {i+1}/{num_samples}:")
            
            try:
                # Get sample
                example = dataset.get_example("train", i * 100)  # Space out samples
                image = example['image']
                
                # Simple detection
                prompt = f"Are these organs visible: {', '.join(organ_classes)}? Answer with yes/no for each."
                response = model((image, prompt))
                
                if response:
                    print(f"  Response: {response[:100]}...")
                    successes += 1
                else:
                    print("  ⚠️ Empty response")
                    
            except Exception as e:
                print(f"  Error: {e}")
        
        print(f"\n✅ Success rate: {successes}/{num_samples}")
        return successes == num_samples
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    finally:
        if 'model' in locals():
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    # Run integration test
    success1 = test_cholec_with_llava()
    
    # Test multiple samples
    success2 = test_multiple_samples(3)
    
    if success1 and success2:
        print("\n✨ All integration tests passed!")
        print("\n🎉 VLM models are ready for full evaluation!")
    else:
        print("\n⚠️ Some tests failed")