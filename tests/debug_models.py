"""Debug script for CholecNet and GoNoGoNet models."""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Add src to path
import sys
sys.path.append('src')

from endopoint.models.cholenet import CholeNet, load_cholenet_model
from endopoint.models.gonogonet import GoNoGoNet, load_gonogo_model


def analyze_checkpoint(checkpoint_path: str, model_type: str = 'cholenet'):
    """Analyze a checkpoint file to understand its structure and training state."""
    print(f"\n{'='*60}")
    print(f"Analyzing checkpoint: {checkpoint_path}")
    print(f"Model type: {model_type}")
    print('='*60)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Check if it's a dict or direct state_dict
    if isinstance(checkpoint, dict):
        print("\nCheckpoint structure:")
        for key in checkpoint.keys():
            if key == 'optimizer_state_dict':
                print(f"  - {key}: optimizer state")
                opt_state = checkpoint[key]
                if 'param_groups' in opt_state:
                    for i, group in enumerate(opt_state['param_groups']):
                        print(f"    Group {i}: lr={group.get('lr', 'N/A')}, "
                              f"momentum={group.get('momentum', 'N/A')}, "
                              f"weight_decay={group.get('weight_decay', 'N/A')}")
            elif key in ['model_state_dict', 'state_dict']:
                state_dict = checkpoint[key]
                print(f"  - {key}: {len(state_dict)} parameters")
                # Analyze parameter shapes
                total_params = 0
                for name, param in state_dict.items():
                    total_params += param.numel()
                print(f"    Total parameters: {total_params:,}")
            elif key == 'epoch':
                print(f"  - {key}: {checkpoint[key]}")
            elif key == 'loss':
                print(f"  - {key}: {checkpoint[key]}")
            else:
                print(f"  - {key}: {type(checkpoint[key])}")
    else:
        print(f"Checkpoint is direct state_dict with {len(checkpoint)} parameters")
    
    # Analyze gradients if present
    if isinstance(checkpoint, dict) and 'optimizer_state_dict' in checkpoint:
        opt_state = checkpoint['optimizer_state_dict']
        if 'state' in opt_state:
            print(f"\nOptimizer state analysis:")
            print(f"  Number of tracked parameters: {len(opt_state['state'])}")
            
            # Sample a few parameters
            grad_magnitudes = []
            momentum_magnitudes = []
            
            for param_id, param_state in list(opt_state['state'].items())[:10]:
                if 'momentum_buffer' in param_state:
                    momentum = param_state['momentum_buffer']
                    if isinstance(momentum, torch.Tensor):
                        mag = torch.abs(momentum).mean().item()
                        momentum_magnitudes.append(mag)
            
            if momentum_magnitudes:
                print(f"  Average momentum magnitude (first 10): {np.mean(momentum_magnitudes):.6f}")
                print(f"  Min momentum: {np.min(momentum_magnitudes):.6f}")
                print(f"  Max momentum: {np.max(momentum_magnitudes):.6f}")


def test_model_forward(model_type: str = 'cholenet', device: str = 'cuda'):
    """Test forward pass of the model with random input."""
    print(f"\n{'='*60}")
    print(f"Testing {model_type} forward pass")
    print('='*60)
    
    # Create model
    if model_type == 'cholenet':
        model = CholeNet(n_classes=14).to(device)
    else:
        model = GoNoGoNet(n_classes=3).to(device)
    
    # Test with different input sizes
    test_sizes = [(384, 640), (256, 512), (512, 512)]
    
    for h, w in test_sizes:
        print(f"\nTesting input size: {h}x{w}")
        x = torch.randn(2, 3, h, w).to(device)
        
        with torch.no_grad():
            output = model(x)
            
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
        
        # Test predictions
        pred = model.predict(x)
        print(f"  Prediction shape: {pred.shape}")
        print(f"  Unique predicted classes: {torch.unique(pred).cpu().tolist()}")
        
        # Test probabilities
        proba = model.predict_proba(x)
        print(f"  Probability shape: {proba.shape}")
        print(f"  Probability sum check (should be ~1.0): {proba[0].sum(dim=0)[0,0]:.4f}")


def test_loaded_model(checkpoint_path: str, model_type: str = 'cholenet', device: str = 'cuda'):
    """Test a loaded model from checkpoint."""
    print(f"\n{'='*60}")
    print(f"Testing loaded {model_type} model")
    print('='*60)
    
    try:
        # Load model
        if model_type == 'cholenet':
            model = CholeNet.load_from_checkpoint(checkpoint_path, device=device)
        else:
            model = GoNoGoNet.load_from_checkpoint(checkpoint_path, device=device)
        
        print("Model loaded successfully!")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Test with random input
        x = torch.randn(1, 3, 384, 640).to(device)
        
        with torch.no_grad():
            output = model(x)
            
        print(f"\nTest forward pass:")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
        
        # Check for NaN or Inf
        if torch.isnan(output).any():
            print("  WARNING: Output contains NaN values!")
        if torch.isinf(output).any():
            print("  WARNING: Output contains Inf values!")
            
        # Test prediction
        pred = model.predict(x)
        unique_preds = torch.unique(pred)
        print(f"  Unique predictions: {unique_preds.cpu().tolist()}")
        
        # Check class distribution
        if model_type == 'cholenet':
            n_classes = 14
        else:
            n_classes = 3
            
        for c in range(n_classes):
            count = (pred == c).sum().item()
            pct = 100.0 * count / pred.numel()
            print(f"    Class {c}: {count:,} pixels ({pct:.1f}%)")
            
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()


def check_gradient_flow(model_type: str = 'cholenet', device: str = 'cuda'):
    """Check gradient flow through the model."""
    print(f"\n{'='*60}")
    print(f"Checking gradient flow for {model_type}")
    print('='*60)
    
    # Create model
    if model_type == 'cholenet':
        model = CholeNet(n_classes=14).to(device)
        n_classes = 14
    else:
        model = GoNoGoNet(n_classes=3).to(device)
        n_classes = 3
    
    # Create dummy input and target
    x = torch.randn(2, 3, 256, 256).to(device)
    target = torch.randint(0, n_classes, (2, 256, 256)).to(device)
    
    # Forward pass
    output = model(x)
    
    # Compute loss
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, target)
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    print("\nGradient analysis:")
    grad_norms = []
    zero_grad_params = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            grad_norms.append(grad_norm)
            
            if grad_norm == 0:
                zero_grad_params.append(name)
        else:
            zero_grad_params.append(name)
    
    if grad_norms:
        print(f"  Number of parameters with gradients: {len(grad_norms)}")
        print(f"  Average gradient norm: {np.mean(grad_norms):.6f}")
        print(f"  Min gradient norm: {np.min(grad_norms):.6f}")
        print(f"  Max gradient norm: {np.max(grad_norms):.6f}")
        
        if zero_grad_params:
            print(f"\n  Parameters with zero/no gradients: {len(zero_grad_params)}")
            for param_name in zero_grad_params[:5]:
                print(f"    - {param_name}")
    else:
        print("  WARNING: No gradients computed!")
    
    # Plot gradient distribution
    if grad_norms:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.hist(grad_norms, bins=50)
        plt.xlabel('Gradient Norm')
        plt.ylabel('Count')
        plt.title(f'{model_type} Gradient Norms')
        plt.yscale('log')
        
        plt.subplot(1, 2, 2)
        plt.hist(np.log10(np.array(grad_norms) + 1e-10), bins=50)
        plt.xlabel('Log10(Gradient Norm)')
        plt.ylabel('Count')
        plt.title(f'{model_type} Log Gradient Norms')
        
        plt.tight_layout()
        plt.savefig(f'gradient_analysis_{model_type}.png')
        print(f"\n  Gradient plot saved to gradient_analysis_{model_type}.png")


def main():
    """Run debugging tests."""
    
    # Default checkpoint paths
    cholenet_checkpoint = "/shared_data0/weiqiuy/llm_cholec_organ/saved_models/organ_s0_tts56_tvs0_all_0.01_cosine_shuffle_last.pt"
    gonogo_checkpoint = "/shared_data0/weiqiuy/llm_cholec_organ/saved_models/gonogo_s0_tts56_tvs0_all_0.01_cosine_shuffle_last.pt"
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Test CholecNet
    print("\n" + "="*80)
    print("TESTING CHOLECNET")
    print("="*80)
    
    if Path(cholenet_checkpoint).exists():
        analyze_checkpoint(cholenet_checkpoint, 'cholenet')
        test_loaded_model(cholenet_checkpoint, 'cholenet', device)
    else:
        print(f"Checkpoint not found: {cholenet_checkpoint}")
    
    test_model_forward('cholenet', device)
    check_gradient_flow('cholenet', device)
    
    # Test GoNoGoNet
    print("\n" + "="*80)
    print("TESTING GONOGONET")
    print("="*80)
    
    if Path(gonogo_checkpoint).exists():
        analyze_checkpoint(gonogo_checkpoint, 'gonogo')
        test_loaded_model(gonogo_checkpoint, 'gonogo', device)
    else:
        print(f"Checkpoint not found: {gonogo_checkpoint}")
    
    test_model_forward('gonogo', device)
    check_gradient_flow('gonogo', device)
    
    print("\n" + "="*80)
    print("DEBUGGING COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()