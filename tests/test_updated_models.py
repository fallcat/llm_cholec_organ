"""Test the updated CholeNet and GoNoGoNet models."""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
import json

# Add src to path
import sys
sys.path.append('src')

from endopoint.models.cholenet import CholeNet, load_cholenet_model
from endopoint.models.gonogo import GoNoGoNet, load_gonogo_model


def test_cholenet():
    """Test CholeNet with updated 4-class structure."""
    print("="*60)
    print("Testing CholeNet (4 organs)")
    print("="*60)
    
    checkpoint_path = "/shared_data0/weiqiuy/llm_cholec_organ/saved_models/organ_s0_tts56_tvs0_all_0.01_cosine_shuffle_last.pt"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"Device: {device}\n")
    
    # Load model
    try:
        model = CholeNet.load_from_checkpoint(checkpoint_path, device=device, n_classes=4)
        print("✓ Model loaded successfully")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Total parameters: {total_params:,}")
        
        # Test with random input
        print("\nTesting forward pass...")
        x = torch.randn(1, 3, 384, 640).to(device)
        
        with torch.no_grad():
            # Test ModelOutput format
            output = model(x, return_tuple=True)
            assert hasattr(output, 'logits'), "Output should have 'logits' field"
            print(f"✓ Output type: {type(output).__name__}")
            print(f"✓ Logits shape: {output.logits.shape}")
            print(f"✓ Logits range: [{output.logits.min():.3f}, {output.logits.max():.3f}]")
            
            # Test prediction
            pred = model.predict(x)
            print(f"✓ Prediction shape: {pred.shape}")
            
            # Check class distribution
            unique, counts = torch.unique(pred, return_counts=True)
            print(f"\nClass distribution in random prediction:")
            for cls, cnt in zip(unique.cpu().tolist(), counts.cpu().tolist()):
                pct = 100.0 * cnt / pred.numel()
                label = model.ID2LABEL.get(cls, f"Unknown_{cls}")
                print(f"  {label} (class {cls}): {cnt:,} pixels ({pct:.1f}%)")
        
        print("\n✓ CholeNet test passed!")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_gonogo():
    """Test GoNoGoNet with updated structure."""
    print("\n" + "="*60)
    print("Testing GoNoGoNet (3 zones)")
    print("="*60)
    
    checkpoint_path = "/shared_data0/weiqiuy/llm_cholec_organ/saved_models/gonogo_s0_tts56_tvs0_all_0.01_cosine_shuffle_best.pt"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"Device: {device}\n")
    
    # Load model
    try:
        model = GoNoGoNet.load_from_checkpoint(checkpoint_path, device=device, n_classes=3)
        print("✓ Model loaded successfully")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Total parameters: {total_params:,}")
        
        # Test with random input
        print("\nTesting forward pass...")
        x = torch.randn(1, 3, 384, 640).to(device)
        
        with torch.no_grad():
            # Test ModelOutput format
            output = model(x, return_tuple=True)
            assert hasattr(output, 'logits'), "Output should have 'logits' field"
            print(f"✓ Output type: {type(output).__name__}")
            print(f"✓ Logits shape: {output.logits.shape}")
            print(f"✓ Logits range: [{output.logits.min():.3f}, {output.logits.max():.3f}]")
            
            # Test prediction
            pred = model.predict(x)
            print(f"✓ Prediction shape: {pred.shape}")
            
            # Check class distribution
            unique, counts = torch.unique(pred, return_counts=True)
            print(f"\nClass distribution in random prediction:")
            for cls, cnt in zip(unique.cpu().tolist(), counts.cpu().tolist()):
                pct = 100.0 * cnt / pred.numel()
                label = model.ID2LABEL.get(cls, f"Unknown_{cls}")
                print(f"  {label} (class {cls}): {cnt:,} pixels ({pct:.1f}%)")
        
        print("\n✓ GoNoGoNet test passed!")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_with_real_image():
    """Test both models with a real image."""
    print("\n" + "="*60)
    print("Testing with real image processing")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create a dummy image
    dummy_image = Image.new('RGB', (640, 384), color='white')
    # Add some colored regions
    pixels = dummy_image.load()
    # Add a red region (potential liver)
    for i in range(100, 200):
        for j in range(100, 200):
            pixels[i, j] = (200, 50, 50)
    # Add a green region (potential gallbladder)
    for i in range(300, 350):
        for j in range(150, 250):
            pixels[i, j] = (50, 200, 50)
    
    print("\nTesting CholeNet with image...")
    try:
        model = load_cholenet_model(device=device)
        
        # Process image
        input_tensor = model.process_image(dummy_image)
        input_tensor = input_tensor.to(device)
        
        with torch.no_grad():
            output = model(input_tensor, return_tuple=True)
            pred_mask = output.logits.argmax(dim=1)[0].cpu().numpy()
        
        # Get organ presence
        presence = model.get_organ_presence(pred_mask, min_pixels=50)
        print("Detected organs:", {k: v for k, v in presence.items() if v})
        print("✓ CholeNet image processing passed")
        
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\nTesting GoNoGoNet with image...")
    try:
        model = load_gonogo_model(device=device)
        
        # Process image
        input_tensor = model.process_image(dummy_image)
        input_tensor = input_tensor.to(device)
        
        with torch.no_grad():
            output = model(input_tensor, return_tuple=True)
            pred_mask = output.logits.argmax(dim=1)[0].cpu().numpy()
        
        # Get zone presence
        presence = model.get_organ_presence(pred_mask, min_pixels=50)
        print("Detected zones:", {k: v for k, v in presence.items() if v})
        print("✓ GoNoGoNet image processing passed")
        
    except Exception as e:
        print(f"✗ Error: {e}")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("TESTING UPDATED MODELS")
    print("="*80)
    
    cholenet_ok = test_cholenet()
    gonogo_ok = test_gonogo()
    test_with_real_image()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"CholeNet: {'✓ PASSED' if cholenet_ok else '✗ FAILED'}")
    print(f"GoNoGoNet: {'✓ PASSED' if gonogo_ok else '✗ FAILED'}")
    
    if cholenet_ok and gonogo_ok:
        print("\n✓ All tests passed! Models are working correctly.")
    else:
        print("\n✗ Some tests failed. Please check the error messages above.")


if __name__ == '__main__':
    main()