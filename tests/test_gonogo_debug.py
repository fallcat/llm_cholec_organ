#!/usr/bin/env python3
"""Debug GoNoGoNet model output."""

import sys
import torch
import numpy as np
from PIL import Image

# Add src to path
sys.path.insert(0, '/shared_data0/weiqiuy/llm_cholec_organ/src')

from endopoint.models.gonogo import load_gonogo_model
from endopoint.datasets.cholec_gonogo import CholecGoNoGoAdapter


def debug_model():
    """Debug GoNoGoNet model outputs."""
    
    print("=" * 80)
    print("DEBUGGING GONOGONET MODEL")
    print("=" * 80)
    
    # Load model
    print("\n1. Loading model...")
    model = load_gonogo_model(device='cuda')
    model.eval()
    
    # Check model architecture
    print("\n2. Model architecture check:")
    print(f"  - Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - Output classes: {model.n_classes}")
    
    # Load dataset
    print("\n3. Loading test images...")
    dataset = CholecGoNoGoAdapter()
    
    # Test on multiple images
    for idx in [0, 10, 20]:
        print(f"\n--- Testing image {idx} ---")
        example = dataset.get_example('test', idx)
        image = example['image']
        gt_label = example['gonogo_label']
        
        # Process image
        input_tensor = model.process_image(image, target_size=(640, 384))
        device = next(model.parameters()).device
        input_tensor = input_tensor.to(device)
        
        print(f"Input tensor stats:")
        print(f"  - Shape: {input_tensor.shape}")
        print(f"  - Min: {input_tensor.min().item():.3f}")
        print(f"  - Max: {input_tensor.max().item():.3f}")
        print(f"  - Mean: {input_tensor.mean().item():.3f}")
        
        # Get raw outputs
        with torch.no_grad():
            logits = model(input_tensor)
        
        print(f"Logits stats:")
        print(f"  - Shape: {logits.shape}")
        print(f"  - Min: {logits.min().item():.3f}")
        print(f"  - Max: {logits.max().item():.3f}")
        print(f"  - Mean per channel:")
        for c in range(logits.shape[1]):
            print(f"    Channel {c}: mean={logits[0,c].mean().item():.3f}, std={logits[0,c].std().item():.3f}")
        
        # Apply softmax to see probabilities
        probs = torch.softmax(logits, dim=1)
        print(f"Probability stats:")
        for c, name in model.ID2LABEL.items():
            prob_c = probs[0, c]
            print(f"  - {name}: min={prob_c.min().item():.3f}, max={prob_c.max().item():.3f}, mean={prob_c.mean().item():.3f}")
        
        # Get predictions
        pred_mask = torch.argmax(logits, dim=1)[0].cpu().numpy()
        unique, counts = np.unique(pred_mask, return_counts=True)
        print(f"Predictions:")
        for u, c in zip(unique, counts):
            pct = 100 * c / pred_mask.size
            print(f"  - Class {u} ({model.ID2LABEL[u]}): {c} pixels ({pct:.1f}%)")
        
        # Check ground truth
        gt_array = np.array(gt_label)
        gt_unique, gt_counts = np.unique(gt_array, return_counts=True)
        print(f"Ground truth:")
        for u, c in zip(gt_unique, gt_counts):
            pct = 100 * c / gt_array.size
            if u in model.ID2LABEL:
                print(f"  - Class {u} ({model.ID2LABEL[u]}): {c} pixels ({pct:.1f}%)")
    
    # Check if model weights are frozen
    print("\n4. Model weight check:")
    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"  - {name[:50]}: requires_grad={param.requires_grad}, mean={param.mean().item():.6f}")
            break
    
    print("\n" + "=" * 80)
    print("DEBUG COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    debug_model()