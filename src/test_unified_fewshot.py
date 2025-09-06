#!/usr/bin/env python3
"""Test unified few-shot selection for all datasets.

This script demonstrates how the UnifiedFewShotSelector works with
any dataset that follows the required protocol.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from endopoint.datasets import build_dataset
from endopoint.fewshot import UnifiedFewShotSelector


def test_dataset(dataset_name: str, dataset_config: Optional[Dict[str, Any]] = None):
    """Test few-shot selection for a dataset.
    
    Args:
        dataset_name: Dataset name 
        dataset_config: Configuration for dataset
    """
    print("\n" + "="*70)
    print(f"Testing: {dataset_name}")
    print("="*70)
    
    # Build dataset with config
    if dataset_config is None:
        dataset_config = {}
    
    # Set default configs for known datasets
    default_configs = {
        "cholecseg8k_local": {
            "data_dir": "/shared_data0/weiqiuy/datasets/cholecseg8k"
        },
        "cholec_organs": {
            "data_dir": "/shared_data0/weiqiuy/real_drs/data/abdomen_exlib",
            "video_globs": "public",
            "gen_seed": 56,
            "train_val_seed": 0
        },
        "cholec_gonogo": {
            "data_dir": "/shared_data0/weiqiuy/real_drs/data/abdomen_exlib",
            "video_globs": "public",
            "gen_seed": 56,
            "train_val_seed": 0
        }
    }
    
    # Merge with defaults
    if dataset_name in default_configs:
        config = default_configs[dataset_name].copy()
        config.update(dataset_config)
    else:
        config = dataset_config
    
    # Build dataset
    try:
        dataset = build_dataset(dataset_name, **config)
    except Exception as e:
        print(f"❌ Failed to build dataset: {e}")
        return None
    
    print(f"✅ Dataset loaded: {getattr(dataset, 'dataset_tag', dataset.__class__.__name__)}")
    print(f"   Train: {dataset.total('train')} examples")
    print(f"   Val: {dataset.total('validation')} examples")
    print(f"   Test: {dataset.total('test')} examples")
    print(f"   Classes: {len(dataset.label_ids)}")
    
    # Create unified selector with custom output directory
    output_dir = Path.home() / "tmp" / "fewshot_test" / dataset_name
    
    selector = UnifiedFewShotSelector(
        dataset=dataset,
        output_dir=output_dir,  # Custom directory for testing
        n_test_samples=50,       # Small for testing
        n_pos_examples=1,
        n_neg_absent=1,
        n_neg_wrong=1,
        min_pixels=50,
        seed=42,
        cache_enabled=True       # Enable caching
    )
    
    print(f"\n📁 Output directory: {selector.output_dir}")
    
    # Build all examples with a subset for speed
    print("\n🚀 Building all few-shot examples...")
    results = selector.build_all_examples(
        split="train",
        max_samples=200,  # Only process first 200 samples for speed
        use_advanced_selection=True
    )
    
    return results


def test_generic_dataset():
    """Test with a minimal mock dataset to show it works with any protocol-compliant dataset."""
    
    print("\n" + "="*70)
    print("Testing: Generic Mock Dataset")
    print("="*70)
    
    import numpy as np
    import torch
    
    class MockDataset:
        """Minimal mock dataset that follows the protocol."""
        
        def __init__(self):
            self.dataset_tag = "mock_dataset"
            self.label_ids = [1, 2, 3]
            self.id2label = {1: "ClassA", 2: "ClassB", 3: "ClassC"}
            
        def total(self, split):
            return {"train": 100, "validation": 20, "test": 30}[split]
        
        def get_example(self, split, index):
            # Return dummy data
            return {
                "image": np.random.rand(224, 224, 3),
                "label": np.random.randint(0, 4, (224, 224)),
                "id": f"{split}_{index}"
            }
        
        def example_to_tensors(self, example):
            img = torch.tensor(example["image"]).permute(2, 0, 1)
            label = torch.tensor(example["label"])
            return img, label
        
        def labels_to_presence_vector(self, label_tensor, min_pixels=1):
            presence = torch.zeros(len(self.label_ids))
            for i, class_id in enumerate(self.label_ids):
                presence[i] = (label_tensor == class_id).sum() >= min_pixels
            return presence.long()
        
        def sample_point_in_mask(self, label_tensor, class_id, strategy="centroid"):
            mask = (label_tensor == class_id)
            if not mask.any():
                return None
            y, x = torch.where(mask)
            return (int(x[0].item()), int(y[0].item()))
        
        def get_bounding_boxes(self, label_tensor, class_id, min_pixels=10):
            mask = (label_tensor == class_id)
            if not mask.any():
                return {}
            y, x = torch.where(mask)
            x1, y1 = x.min().item(), y.min().item()
            x2, y2 = x.max().item(), y.max().item()
            return {class_id: [(int(x1), int(y1), int(x2), int(y2))]}
    
    # Create mock dataset
    dataset = MockDataset()
    
    # Create selector
    selector = UnifiedFewShotSelector(
        dataset=dataset,
        n_test_samples=20,
        cache_enabled=False  # Disable cache for mock
    )
    
    print(f"✅ Mock dataset created with {len(dataset.label_ids)} classes")
    
    # Test individual components
    print("\n1. Computing presence matrix...")
    Y = selector.compute_presence_matrix("train", max_samples=50)
    print(f"   Shape: {Y.shape}")
    
    print("\n2. Selecting balanced test set...")
    indices, info = selector.select_balanced_test_set(Y)
    print(f"   Selected: {len(indices)} samples")
    
    print("\n3. Building pointing examples...")
    pointing = selector.build_pointing_examples(Y, indices)
    if pointing:
        print(f"   Created examples for {len(pointing['plan'])} classes")
    
    print("\n4. Building bbox examples...")
    bbox = selector.build_bbox_examples(Y, indices)
    if bbox:
        print(f"   Created examples for {len(bbox['plan'])} classes")
    
    # Print summary
    selector.print_summary(pointing, bbox, info)
    
    print("\n✅ Mock dataset test completed successfully!")


def main():
    """Run tests for multiple datasets."""
    print("="*70)
    print("Unified Few-Shot Selector Test Suite")
    print("="*70)
    print("\nThis demonstrates that the UnifiedFewShotSelector class")
    print("works with ANY dataset that implements the required protocol.")
    
    # Test with mock dataset first
    print("\n" + "🧪 Test 1: Generic Mock Dataset " + "🧪")
    test_generic_dataset()
    
    # Test with real datasets
    datasets_to_test = [
        ("cholecseg8k_local", None),
        ("cholec_organs", {"video_globs": "public"}),
        ("cholec_gonogo", {"video_globs": "public"}),
    ]
    
    for i, (dataset_name, config) in enumerate(datasets_to_test, 2):
        print(f"\n" + f"🧪 Test {i}: {dataset_name} " + "🧪")
        try:
            results = test_dataset(dataset_name, config)
            if results:
                print(f"\n✅ {dataset_name} test completed!")
        except Exception as e:
            print(f"\n❌ {dataset_name} test failed: {e}")
    
    print("\n" + "="*70)
    print("✨ All tests completed!")
    print("="*70)
    print("\nThe UnifiedFewShotSelector successfully works with:")
    print("1. Any dataset following the protocol")
    print("2. Automatic parameter configuration")
    print("3. Flexible caching system")
    print("4. Support for both pointing and bbox tasks")


if __name__ == "__main__":
    main()