#!/usr/bin/env python3
"""
RASO (Recognize Anything in Surgery) model adapter.

This module provides an interface to use the RASO model for surgical image analysis
within the endopoint framework.
"""

import sys
import torch
from PIL import Image
from pathlib import Path
from typing import List, Dict, Union, Tuple, Optional

# Add RASO to path
sys.path.insert(0, '/shared_data0/weiqiuy/github/raso')

from raso.models import raso
from raso import inference_ram, get_transform


class RASORModel:
    """
    RASO model wrapper for surgical image analysis.
    """
    
    def __init__(self, 
                 model_path: str = '/shared_data0/weiqiuy/github/hf_repos/raso/raso_zeroshot.pth',
                 image_size: int = 384,
                 vit: str = 'swin_l',
                 device: Optional[str] = None):
        """
        Initialize RASO model.
        
        Args:
            model_path: Path to the pretrained RASO model
            image_size: Input image size
            vit: Vision transformer architecture
            device: Device to load model on (auto-detected if None)
        """
        self.model_path = model_path
        self.image_size = image_size
        self.vit = vit
        
        # Setup device
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Load model and transform
        self.model = None
        self.transform = None
        self._load_model()
    
    def _load_model(self):
        """Load the RASO model and preprocessing transform."""
        print(f"Loading RASO model from {self.model_path}")
        
        # Load model
        self.model = raso(
            pretrained=self.model_path,
            image_size=self.image_size,
            vit=self.vit
        )
        self.model.eval()
        self.model = self.model.to(self.device)
        
        # Setup transform
        self.transform = get_transform(image_size=self.image_size)
        
        print(f"✅ RASO model loaded on {self.device}")
        print(f"   Image size: {self.image_size}x{self.image_size}")
        print(f"   Architecture: {self.vit}")
    
    def analyze_image(self, 
                      image: Union[str, Image.Image],
                      threshold: float = 0.65) -> List[str]:
        """
        Analyze a single image using RASO.
        
        Args:
            image: Path to image file or PIL Image object
            threshold: Confidence threshold for results
        
        Returns:
            List of detected tags/labels
        """
        # Load image if path provided
        if isinstance(image, str):
            if not Path(image).exists():
                raise FileNotFoundError(f"Image not found: {image}")
            image_pil = Image.open(image).convert('RGB')
        elif isinstance(image, Image.Image):
            image_pil = image.convert('RGB')
        else:
            raise ValueError("Image must be a file path or PIL Image")
        
        # Preprocess image
        image_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            result = inference_ram(image_tensor, self.model)
        
        return result[0] if result else []
    
    def batch_analyze(self,
                      images: List[Union[str, Image.Image]], 
                      threshold: float = 0.65) -> List[List[str]]:
        """
        Analyze multiple images in batch.
        
        Args:
            images: List of image paths or PIL Images
            threshold: Confidence threshold for results
        
        Returns:
            List of results, one per input image
        """
        results = []
        
        for i, image in enumerate(images):
            try:
                tags = self.analyze_image(image, threshold=threshold)
                results.append(tags)
            except Exception as e:
                print(f"Error processing image {i}: {e}")
                results.append([])
        
        return results
    
    def analyze_with_multiple_thresholds(self,
                                       image: Union[str, Image.Image],
                                       thresholds: List[float] = [0.5, 0.65, 0.8]) -> Dict[float, List[str]]:
        """
        Analyze image with multiple confidence thresholds.
        
        Args:
            image: Path to image file or PIL Image object
            thresholds: List of thresholds to test
        
        Returns:
            Dictionary mapping threshold to detected tags
        """
        results = {}
        
        for threshold in thresholds:
            try:
                tags = self.analyze_image(image, threshold=threshold)
                results[threshold] = tags
            except Exception as e:
                print(f"Error with threshold {threshold}: {e}")
                results[threshold] = []
        
        return results


def load_raso_model(model_path: str = '/shared_data0/weiqiuy/github/hf_repos/raso/raso_zeroshot.pth',
                    image_size: int = 384,
                    vit: str = 'swin_l',
                    device: Optional[str] = None) -> RASORModel:
    """
    Convenience function to load RASO model.
    
    Args:
        model_path: Path to the pretrained RASO model
        image_size: Input image size
        vit: Vision transformer architecture
        device: Device to load model on
    
    Returns:
        RASORModel instance
    """
    return RASORModel(model_path=model_path, 
                      image_size=image_size, 
                      vit=vit, 
                      device=device)


def demo_raso():
    """Demonstrate RASO usage on surgical images."""
    
    print("🏥 RASO (Recognize Anything in Surgery) Demo")
    print("=" * 50)
    
    # Load model
    raso_model = load_raso_model()
    
    # Example image paths - adjust these to your available images
    example_images = [
        "/shared_data0/weiqiuy/real_drs/data/abdomen_exlib/images/cholec80_video20_006.png",
        "/shared_data0/weiqiuy/datasets/cholecseg8k/train/images/video01_00040.png",
        "/shared_data0/weiqiuy/datasets/cholecseg8k/train/images/video01_00080.png"
    ]
    
    for image_path in example_images:
        if not Path(image_path).exists():
            print(f"⚠️  Skipping missing image: {image_path}")
            continue
            
        print(f"\n📋 Results for: {Path(image_path).name}")
        print("-" * 40)
        
        # Test with multiple thresholds
        results = raso_model.analyze_with_multiple_thresholds(image_path)
        
        for threshold, tags in results.items():
            print(f"Threshold {threshold}: {tags}")
    
    print("\n✨ Demo completed!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RASO inference on surgical images")
    parser.add_argument('--image', type=str, help='Path to a single image')
    parser.add_argument('--threshold', type=float, default=0.65, help='Confidence threshold')
    parser.add_argument('--model-path', type=str, 
                       default='/shared_data0/weiqiuy/github/hf_repos/raso/raso_zeroshot.pth',
                       help='Path to RASO model')
    
    args = parser.parse_args()
    
    if args.image:
        # Single image analysis
        print(f"🔍 Analyzing single image: {args.image}")
        raso_model = load_raso_model(model_path=args.model_path)
        results = raso_model.analyze_image(args.image, threshold=args.threshold)
        print(f"Results with threshold {args.threshold}: {results}")
    else:
        # Run demo
        demo_raso()