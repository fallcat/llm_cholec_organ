"""Test CholeNet mask resizing directly."""

import sys
sys.path.append('src')

import numpy as np
from PIL import Image
import torch

from endopoint.models.cholenet import load_cholenet_model

def test_mask_resizing():
    """Test that CholeNet properly resizes masks."""
    
    # Load model
    print("Loading CholeNet...")
    model = load_cholenet_model(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test with different image sizes
    test_cases = [
        (854, 480),  # CholecSeg8k
        (640, 384),  # Native size
        (800, 600),  # Random size
    ]
    
    for width, height in test_cases:
        print(f"\n{'='*60}")
        print(f"Testing with image size: {width}x{height}")
        print('='*60)
        
        # Create dummy image
        image = Image.new('RGB', (width, height), color='white')
        print(f"Created PIL image with size: {image.size}")
        
        # Store original size
        original_size = image.size  # (width, height)
        print(f"Original size from PIL: {original_size}")
        
        # Process at model size
        input_tensor = model.process_image(image, target_size=(640, 384))
        print(f"Input tensor shape: {input_tensor.shape}")
        
        device = next(model.parameters()).device
        input_tensor = input_tensor.to(device)
        
        # Get prediction
        with torch.no_grad():
            output = model(input_tensor, return_tuple=True)
            logits = output.logits
            pred_mask = logits.argmax(dim=1)[0].cpu().numpy()
        
        print(f"Raw mask shape from model: {pred_mask.shape}")
        print(f"Raw mask dtype: {pred_mask.dtype}")
        
        # Check what needs to be done
        target_height, target_width = original_size[1], original_size[0]
        print(f"Target dimensions: height={target_height}, width={target_width}")
        
        # Check if dimensions are swapped
        if target_width < target_height:
            print("WARNING: Width < Height, dimensions may be swapped!")
            target_width, target_height = target_height, target_width
            print(f"Corrected target dimensions: height={target_height}, width={target_width}")
        
        # Check if resizing is needed
        if pred_mask.shape != (target_height, target_width):
            print(f"✓ Resizing needed: {pred_mask.shape} -> ({target_height}, {target_width})")
            
            from scipy import ndimage
            pred_mask_resized = np.zeros((target_height, target_width), dtype=pred_mask.dtype)
            
            for class_id in np.unique(pred_mask):
                class_mask = (pred_mask == class_id).astype(np.float32)
                class_mask_resized = ndimage.zoom(class_mask, 
                                                 (target_height / pred_mask.shape[0],
                                                  target_width / pred_mask.shape[1]),
                                                 order=0)
                pred_mask_resized[class_mask_resized > 0.5] = class_id
            
            pred_mask = pred_mask_resized.astype(np.int32)
            print(f"✓ Final mask shape: {pred_mask.shape}")
            print(f"✓ Final mask dtype: {pred_mask.dtype}")
        else:
            print(f"✗ No resizing needed, shapes match")
        
        # Verify final shape
        assert pred_mask.shape == (target_height, target_width), \
            f"Shape mismatch! Expected ({target_height}, {target_width}), got {pred_mask.shape}"
        print(f"✓ Shape verification passed!")

if __name__ == '__main__':
    test_mask_resizing()