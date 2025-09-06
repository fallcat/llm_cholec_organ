#!/usr/bin/env python3
"""Create few-shot examples for combined bounding box detection."""

import json
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, '/shared_data0/weiqiuy/llm_cholec_organ/src')

from endopoint.datasets.cholecseg8k_local import CholecSeg8kLocalAdapter


def create_combined_fewshot_examples(num_examples: int = 5):
    """Create few-shot examples showing complete organ detection.
    
    Args:
        num_examples: Number of example images to create
        
    Returns:
        List of example dictionaries
    """
    # Load dataset
    data_dir = "/shared_data0/weiqiuy/datasets/cholecseg8k"
    adapter = CholecSeg8kLocalAdapter(data_dir=data_dir)
    
    print(f"Creating {num_examples} few-shot examples for combined detection...")
    
    # Select diverse examples from training set
    # Pick examples with varying numbers of organs present
    train_size = adapter.total('train')
    
    # Sample indices evenly across the training set
    selected_indices = np.linspace(100, train_size - 100, num_examples, dtype=int)
    
    examples = []
    
    for idx in selected_indices:
        example = adapter.get_example('train', idx)
        
        # Get ground truth masks
        _, lab_tensor = adapter.example_to_tensors(example)
        mask = lab_tensor.numpy()
        
        # Extract bounding boxes for all present organs
        detections = {}
        
        for organ_id in adapter.label_ids:
            organ_name = adapter.id2label[organ_id]
            organ_mask = (mask == organ_id).astype(np.uint8)
            
            if organ_mask.sum() >= 50:  # min_pixels threshold
                # Get bounding box
                rows, cols = np.where(organ_mask)
                if len(rows) > 0:
                    bbox = [
                        int(cols.min()),
                        int(rows.min()),
                        int(cols.max()),
                        int(rows.max())
                    ]
                    detections[organ_name] = {
                        "present": True,
                        "bbox": bbox
                    }
                else:
                    detections[organ_name] = {"present": False}
            else:
                detections[organ_name] = {"present": False}
        
        # Count present organs
        num_present = sum(1 for d in detections.values() if d.get("present", False))
        
        examples.append({
            "train_idx": int(idx),
            "num_organs_present": num_present,
            "detections": detections
        })
        
        print(f"  Example {len(examples)}: idx={idx}, {num_present} organs present")
    
    # Save examples
    output_dir = Path("/shared_data0/weiqiuy/llm_cholec_organ/data_info/cholecseg8k_local_balanced_200")
    output_file = output_dir / "fewshot_examples_combined.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            "num_examples": num_examples,
            "examples": examples,
            "description": "Few-shot examples for combined organ detection (all organs in one query)"
        }, f, indent=2)
    
    print(f"\n✅ Saved {num_examples} few-shot examples to {output_file}")
    
    # Print summary
    print("\nExample diversity:")
    for i, ex in enumerate(examples, 1):
        organs_present = [k for k, v in ex['detections'].items() if v.get('present', False)]
        print(f"  Example {i}: {ex['num_organs_present']} organs - {', '.join(organs_present[:3])}...")
    
    return examples


if __name__ == "__main__":
    create_combined_fewshot_examples(num_examples=5)