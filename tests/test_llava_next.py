#!/usr/bin/env python3
"""Test LLaVA-NeXT model with proper configuration."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

def test_direct():
    """Test LLaVA-NeXT directly without wrapper."""
    print("Testing LLaVA-NeXT directly...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    model_name = "llava-hf/llava-v1.6-mistral-7b-hf"
    
    try:
        # Load model and processor
        print(f"\nLoading {model_name}...")
        processor = LlavaNextProcessor.from_pretrained(model_name)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        model.eval()
        print("✅ Model loaded")
        
        # Create test image
        image = Image.new('RGB', (224, 224), color='blue')
        
        # Test 1: Simple query
        print("\nTest 1: Simple color query")
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What color is this image?"},
                    {"type": "image"},
                ]
            }
        ]
        
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() if hasattr(v, 'cuda') else v for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False,
            )
        
        # Decode only the generated part
        generated = output[0][inputs['input_ids'].shape[1]:]
        response = processor.decode(generated, skip_special_tokens=True)
        print(f"Response: {response}")
        
        # Test 2: Medical context
        print("\nTest 2: Medical context")
        conversation = [
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": "Is this a medical image? Answer yes or no."},
                    {"type": "image"},
                ]
            }
        ]
        
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() if hasattr(v, 'cuda') else v for k, v in inputs.items()}
        
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        
        generated = output[0][inputs['input_ids'].shape[1]:]
        response = processor.decode(generated, skip_special_tokens=True)
        print(f"Response: {response}")
        
        print("\n✅ Direct test successful!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_wrapper():
    """Test through our wrapper."""
    print("\n" + "="*60)
    print("Testing through endopoint wrapper...")
    
    from endopoint.models import LLaVAModel
    
    try:
        model = LLaVAModel(
            model_name="llava-hf/llava-v1.6-mistral-7b-hf",
            use_vllm=False,  # Use transformers
            use_cache=False,
            max_tokens=30,
            verbose=True
        )
        print("✅ Wrapper model loaded")
        
        # Test
        image = Image.new('RGB', (224, 224), color='green')
        response = model((image, "What color is this image?"))
        print(f"Response: {response}")
        
        if response:
            print("✅ Wrapper test successful!")
            return True
        else:
            print("⚠️ Empty response")
            return False
            
    except Exception as e:
        print(f"❌ Wrapper error: {e}")
        return False


if __name__ == "__main__":
    # Test direct first
    success1 = test_direct()
    
    # Then test wrapper
    success2 = test_wrapper()
    
    if success1 and success2:
        print("\n✨ All tests passed!")
    else:
        print("\n⚠️ Some tests failed")