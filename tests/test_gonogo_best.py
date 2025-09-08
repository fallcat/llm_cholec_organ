"""Test GoNoGoNet with best checkpoint."""

import torch
import numpy as np
from PIL import Image
from pathlib import Path

# Add src to path
import sys
sys.path.append('src')

from endopoint.models.gonogo import GoNoGoNet, load_gonogo_model


def test_best_checkpoint():
    """Test the best checkpoint instead of last."""
    
    checkpoint_path = "/shared_data0/weiqiuy/llm_cholec_organ/saved_models/gonogo_s0_tts56_tvs0_all_0.01_cosine_shuffle_best.pt"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Testing checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print("="*60)
    
    # Load checkpoint to inspect
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    print("\nCheckpoint contents:")
    for key in checkpoint.keys():
        if key in ['model_state_dict', 'state_dict']:
            state_dict = checkpoint[key]
            total_params = sum(p.numel() for p in state_dict.values())
            print(f"  {key}: {len(state_dict)} tensors, {total_params:,} parameters")
        elif key in ['epoch', 'miou_val_gonogo', 'mpa_val_gonogo', 'dice_val_gonogo']:
            print(f"  {key}: {checkpoint[key]}")
        else:
            print(f"  {key}: {type(checkpoint[key])}")
    
    print("\n" + "="*60)
    print("Loading model...")
    
    # Load model
    model = GoNoGoNet.load_from_checkpoint(checkpoint_path, device=device)
    print("Model loaded successfully!")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}")
    
    # Test with multiple random inputs
    print("\n" + "="*60)
    print("Testing predictions on random inputs...")
    
    for i in range(3):
        print(f"\nTest {i+1}:")
        x = torch.randn(1, 3, 384, 640).to(device)
        
        with torch.no_grad():
            output = model(x)
            pred = model.predict(x)
            proba = model.predict_proba(x)
        
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
        
        # Check prediction distribution
        unique, counts = torch.unique(pred, return_counts=True)
        print(f"  Predicted classes: {unique.cpu().tolist()}")
        for cls, cnt in zip(unique.cpu().tolist(), counts.cpu().tolist()):
            pct = 100.0 * cnt / pred.numel()
            if cls == 0:
                label = "Background"
            elif cls == 1:
                label = "Go Zone"
            else:
                label = "NoGo Zone"
            print(f"    {label} (class {cls}): {cnt:,} pixels ({pct:.1f}%)")
        
        # Check probability statistics
        print(f"  Probability sum check: {proba[0].sum(dim=0)[0,0]:.4f}")
        print(f"  Max probability per pixel: {proba.max(dim=1)[0].mean():.4f}")
    
    # Test with different preprocessing
    print("\n" + "="*60)
    print("Testing with normalized vs unnormalized input...")
    
    # Unnormalized (0-1 range)
    x_unnorm = torch.rand(1, 3, 384, 640).to(device)
    with torch.no_grad():
        out_unnorm = model(x_unnorm)
        pred_unnorm = model.predict(x_unnorm)
    
    unique_unnorm = torch.unique(pred_unnorm)
    print(f"\nUnnormalized input (0-1 range):")
    print(f"  Output range: [{out_unnorm.min():.3f}, {out_unnorm.max():.3f}]")
    print(f"  Predicted classes: {unique_unnorm.cpu().tolist()}")
    
    # Normalized with ImageNet stats
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    x_norm = (x_unnorm - mean) / std
    
    with torch.no_grad():
        out_norm = model(x_norm)
        pred_norm = model.predict(x_norm)
    
    unique_norm = torch.unique(pred_norm)
    print(f"\nNormalized input (ImageNet stats):")
    print(f"  Output range: [{out_norm.min():.3f}, {out_norm.max():.3f}]")
    print(f"  Predicted classes: {unique_norm.cpu().tolist()}")
    
    # Compare predictions
    diff = (pred_unnorm != pred_norm).sum().item()
    total = pred_unnorm.numel()
    print(f"\nPixels with different predictions: {diff:,} / {total:,} ({100*diff/total:.2f}%)")
    
    print("\n" + "="*60)
    print("Testing complete!")


if __name__ == '__main__':
    test_best_checkpoint()