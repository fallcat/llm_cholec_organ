"""Demo script to run CholeNet and GoNoGoNet on sample images."""

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import random
import glob
import os

# Add src to path
import sys
sys.path.append('src')

from endopoint.models.cholenet import CholeNet, load_cholenet_model
from endopoint.models.gonogo import GoNoGoNet, load_gonogo_model


def visualize_cholenet_results(image, mask, model, save_path=None):
    """Visualize CholeNet segmentation results."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    
    # Full segmentation mask
    axes[0, 1].imshow(mask, cmap='tab10', vmin=0, vmax=3)
    axes[0, 1].set_title("Full Segmentation")
    axes[0, 1].axis('off')
    
    # Overlay
    axes[0, 2].imshow(image)
    axes[0, 2].imshow(mask, alpha=0.5, cmap='tab10', vmin=0, vmax=3)
    axes[0, 2].set_title("Overlay")
    axes[0, 2].axis('off')
    
    # Individual organ masks
    organ_names = ["Liver", "Gallbladder", "Hepatocystic Triangle"]
    for i, (class_id, organ_name) in enumerate(zip([1, 2, 3], organ_names)):
        organ_mask = (mask == class_id).astype(np.float32)
        axes[1, i].imshow(image)
        if organ_mask.any():
            axes[1, i].contour(organ_mask, levels=[0.5], colors='red', linewidths=2)
            # Highlight the organ region
            masked = np.ma.masked_where(organ_mask == 0, organ_mask)
            axes[1, i].imshow(masked, alpha=0.3, cmap='Reds', vmin=0, vmax=1)
        axes[1, i].set_title(f"{organ_name} (Class {class_id})")
        axes[1, i].axis('off')
    
    plt.suptitle("CholeNet Organ Segmentation Results")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"  Saved visualization to {save_path}")
    plt.show()


def visualize_gonogo_results(image, mask, model, save_path=None):
    """Visualize GoNoGoNet segmentation results."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Go Zone (Safe)
    go_mask = (mask == 1).astype(np.float32)
    axes[1].imshow(image)
    if go_mask.any():
        axes[1].contour(go_mask, levels=[0.5], colors='green', linewidths=2)
        masked = np.ma.masked_where(go_mask == 0, go_mask)
        axes[1].imshow(masked, alpha=0.3, cmap='Greens', vmin=0, vmax=1)
    axes[1].set_title("Go Zone (Safe to Cut)")
    axes[1].axis('off')
    
    # NoGo Zone (Unsafe)
    nogo_mask = (mask == 2).astype(np.float32)
    axes[2].imshow(image)
    if nogo_mask.any():
        axes[2].contour(nogo_mask, levels=[0.5], colors='red', linewidths=2)
        masked = np.ma.masked_where(nogo_mask == 0, nogo_mask)
        axes[2].imshow(masked, alpha=0.3, cmap='Reds', vmin=0, vmax=1)
    axes[2].set_title("NoGo Zone (Unsafe)")
    axes[2].axis('off')
    
    # Combined overlay
    axes[3].imshow(image)
    # Create RGB overlay
    overlay = np.zeros((*mask.shape, 3))
    overlay[mask == 1] = [0, 1, 0]  # Green for Go
    overlay[mask == 2] = [1, 0, 0]  # Red for NoGo
    axes[3].imshow(overlay, alpha=0.4)
    axes[3].set_title("Combined Zones")
    axes[3].axis('off')
    
    plt.suptitle("GoNoGoNet Safety Zone Segmentation")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"  Saved visualization to {save_path}")
    plt.show()


def find_sample_images(data_dir="/shared_data0/weiqiuy/real_drs/data/abdomen_exlib", 
                       n_samples=5):
    """Find sample images from the dataset."""
    images_dir = Path(data_dir) / "images"
    
    if not images_dir.exists():
        print(f"Warning: {images_dir} does not exist. Using dummy images.")
        return None
    
    # Find all image files
    image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
    
    if not image_files:
        print(f"Warning: No images found in {images_dir}")
        return None
    
    # Sample randomly
    n_samples = min(n_samples, len(image_files))
    sampled_files = random.sample(image_files, n_samples)
    
    return sampled_files


def create_dummy_images(n_samples=3):
    """Create dummy test images with different patterns."""
    images = []
    
    for i in range(n_samples):
        # Create base image with gradient background
        img = Image.new('RGB', (640, 384))
        pixels = img.load()
        
        # Add gradient background
        for x in range(640):
            for y in range(384):
                r = int(100 + (x / 640) * 50)
                g = int(100 + (y / 384) * 50)
                b = 150
                pixels[x, y] = (r, g, b)
        
        # Add different colored regions for each image
        if i == 0:
            # Add a large red region (simulating liver)
            for x in range(200, 400):
                for y in range(100, 250):
                    pixels[x, y] = (180, 60, 60)
            # Add a small green region (simulating gallbladder)
            for x in range(350, 400):
                for y in range(200, 250):
                    pixels[x, y] = (60, 180, 60)
        
        elif i == 1:
            # Add circular regions
            center_x, center_y = 320, 192
            for x in range(640):
                for y in range(384):
                    dist = ((x - center_x)**2 + (y - center_y)**2)**0.5
                    if dist < 100:
                        pixels[x, y] = (200, 100, 100)
                    elif dist < 150:
                        pixels[x, y] = (100, 200, 100)
        
        else:
            # Add diagonal stripe pattern
            for x in range(640):
                for y in range(384):
                    if (x + y) % 100 < 50:
                        pixels[x, y] = (150, 100, 100)
        
        images.append(img)
    
    return images


def run_demo(use_real_images=True, n_samples=5, save_results=True):
    """Run demo on sample images."""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path("demo_results")
    if save_results:
        output_dir.mkdir(exist_ok=True)
        print(f"Results will be saved to {output_dir}/")
    
    # Load models
    print("\nLoading models...")
    cholenet = load_cholenet_model(device=device)
    gonogo = load_gonogo_model(device=device)
    print("✓ Models loaded successfully")
    
    # Get sample images
    print(f"\nPreparing {n_samples} sample images...")
    if use_real_images:
        image_files = find_sample_images(n_samples=n_samples)
        if image_files:
            images = [(Image.open(f).convert('RGB'), f.name) for f in image_files]
        else:
            print("Using dummy images instead...")
            dummy_images = create_dummy_images(n_samples)
            images = [(img, f"dummy_{i}.png") for i, img in enumerate(dummy_images)]
    else:
        dummy_images = create_dummy_images(n_samples)
        images = [(img, f"dummy_{i}.png") for i, img in enumerate(dummy_images)]
    
    print(f"✓ Loaded {len(images)} images")
    
    # Process each image
    for idx, (image, image_name) in enumerate(images):
        print(f"\n{'='*60}")
        print(f"Processing image {idx+1}/{len(images)}: {image_name}")
        print('='*60)
        
        # Resize if needed
        if image.size != (640, 384):
            image = image.resize((640, 384), Image.BILINEAR)
        
        # Convert to numpy for display
        image_np = np.array(image)
        
        # Run CholeNet
        print("\nRunning CholeNet (Organ Detection)...")
        input_tensor = cholenet.process_image(image)
        input_tensor = input_tensor.to(device)
        
        with torch.no_grad():
            output = cholenet(input_tensor, return_tuple=True)
            cholenet_mask = output.logits.argmax(dim=1)[0].cpu().numpy()
        
        # Get organ statistics
        presence = cholenet.get_organ_presence(cholenet_mask, min_pixels=50)
        detected_organs = [k for k, v in presence.items() if v]
        print(f"  Detected organs: {detected_organs if detected_organs else 'None'}")
        
        # Get bounding boxes
        bboxes = cholenet.get_bounding_boxes(cholenet_mask, min_pixels=50)
        for organ, boxes in bboxes.items():
            if boxes:
                print(f"  {organ}: {len(boxes)} region(s)")
                for box_idx, (x1, y1, x2, y2) in enumerate(boxes[:2]):  # Show max 2 boxes
                    area = (x2 - x1) * (y2 - y1)
                    print(f"    Region {box_idx+1}: bbox=({x1},{y1},{x2},{y2}), area={area} pixels")
        
        # Visualize CholeNet results
        save_path = output_dir / f"cholenet_{idx}_{Path(image_name).stem}.png" if save_results else None
        visualize_cholenet_results(image_np, cholenet_mask, cholenet, save_path)
        
        # Run GoNoGoNet
        print("\nRunning GoNoGoNet (Safety Zones)...")
        input_tensor = gonogo.process_image(image)
        input_tensor = input_tensor.to(device)
        
        with torch.no_grad():
            output = gonogo(input_tensor, return_tuple=True)
            gonogo_mask = output.logits.argmax(dim=1)[0].cpu().numpy()
        
        # Get zone statistics
        zones = gonogo.get_organ_presence(gonogo_mask, min_pixels=50)
        detected_zones = [k for k, v in zones.items() if v]
        print(f"  Detected zones: {detected_zones if detected_zones else 'None'}")
        
        # Calculate zone percentages
        total_pixels = gonogo_mask.size
        background_pct = 100 * np.sum(gonogo_mask == 0) / total_pixels
        go_pct = 100 * np.sum(gonogo_mask == 1) / total_pixels
        nogo_pct = 100 * np.sum(gonogo_mask == 2) / total_pixels
        
        print(f"  Zone distribution:")
        print(f"    Background: {background_pct:.1f}%")
        print(f"    Go Zone (Safe): {go_pct:.1f}%")
        print(f"    NoGo Zone (Unsafe): {nogo_pct:.1f}%")
        
        # Visualize GoNoGoNet results
        save_path = output_dir / f"gonogo_{idx}_{Path(image_name).stem}.png" if save_results else None
        visualize_gonogo_results(image_np, gonogo_mask, gonogo, save_path)
    
    print(f"\n{'='*60}")
    print("Demo completed!")
    if save_results:
        print(f"Results saved to {output_dir}/")
    print('='*60)


def main():
    """Main demo function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run CholeNet and GoNoGoNet demo")
    parser.add_argument('--n-samples', type=int, default=3, 
                       help='Number of samples to process (default: 3)')
    parser.add_argument('--use-dummy', action='store_true',
                       help='Use dummy images instead of real dataset')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save results to disk')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("CholeNet and GoNoGoNet Demo")
    print("="*80)
    
    run_demo(
        use_real_images=not args.use_dummy,
        n_samples=args.n_samples,
        save_results=not args.no_save
    )


if __name__ == '__main__':
    main()