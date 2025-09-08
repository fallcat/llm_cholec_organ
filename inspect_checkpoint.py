#!/usr/bin/env python3
"""Inspect checkpoint file structure."""

import torch
import sys

def inspect_checkpoint(checkpoint_path):
    """Inspect the structure of a checkpoint file."""
    
    print(f"Loading checkpoint: {checkpoint_path}")
    print("=" * 80)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Check type
    print(f"Checkpoint type: {type(checkpoint)}")
    
    if isinstance(checkpoint, dict):
        print(f"\nCheckpoint keys: {list(checkpoint.keys())}")
        
        # Check each key
        for key in checkpoint.keys():
            value = checkpoint[key]
            if key in ['model_state_dict', 'state_dict', 'model']:
                print(f"\n{key}:")
                if isinstance(value, dict):
                    # Show first few and last few keys
                    all_keys = list(value.keys())
                    print(f"  Total keys: {len(all_keys)}")
                    print(f"  First 5 keys:")
                    for k in all_keys[:5]:
                        print(f"    - {k}: shape={value[k].shape if hasattr(value[k], 'shape') else 'N/A'}")
                    print(f"  Last 5 keys:")
                    for k in all_keys[-5:]:
                        print(f"    - {k}: shape={value[k].shape if hasattr(value[k], 'shape') else 'N/A'}")
                    
                    # Check for specific layer patterns
                    has_unet = any('unet' in k for k in all_keys)
                    has_inc = any('inc' in k for k in all_keys)
                    has_outc = any('outc' in k for k in all_keys)
                    print(f"\n  Pattern check:")
                    print(f"    - Contains 'unet': {has_unet}")
                    print(f"    - Contains 'inc': {has_inc}")
                    print(f"    - Contains 'outc': {has_outc}")
                    
                    # Check weight statistics
                    print(f"\n  Weight statistics (first conv layer):")
                    for k in all_keys:
                        if 'inc' in k and 'weight' in k:
                            w = value[k]
                            print(f"    {k}:")
                            print(f"      Shape: {w.shape}")
                            print(f"      Min: {w.min().item():.6f}")
                            print(f"      Max: {w.max().item():.6f}")
                            print(f"      Mean: {w.mean().item():.6f}")
                            print(f"      Std: {w.std().item():.6f}")
                            break
            else:
                print(f"\n{key}: {value if not isinstance(value, torch.Tensor) else f'Tensor{value.shape}'}")
    else:
        # Direct state dict
        print("\nCheckpoint is a direct state dict")
        all_keys = list(checkpoint.keys())
        print(f"Total keys: {len(all_keys)}")
        print(f"First 5 keys:")
        for k in all_keys[:5]:
            print(f"  - {k}: shape={checkpoint[k].shape}")
        print(f"Last 5 keys:")
        for k in all_keys[-5:]:
            print(f"  - {k}: shape={checkpoint[k].shape}")


if __name__ == "__main__":
    # Default checkpoint
    checkpoint_path = "/shared_data0/weiqiuy/llm_cholec_organ/saved_models/gonogo_s0_tts56_tvs0_all_0.01_cosine_shuffle_last.pt"
    
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    
    inspect_checkpoint(checkpoint_path)