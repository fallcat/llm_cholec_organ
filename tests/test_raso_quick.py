#!/usr/bin/env python3
"""
Quick test of RASO model on all three datasets (without full evaluation pipeline).
"""

import sys
sys.path.insert(0, '/shared_data0/weiqiuy/llm_cholec_organ/src')

from pathlib import Path
from endopoint.models import create_model
from endopoint.datasets.cholecseg8k_local import CholecSeg8kLocalAdapter
from endopoint.datasets.cholec_organs import CholecOrgansAdapter
from endopoint.datasets.cholec_gonogo import CholecGoNoGoAdapter


def test_raso_on_dataset(dataset_name, adapter, num_samples=3):
    """Quick test of RASO on a dataset.
    
    Args:
        dataset_name: Name of the dataset
        adapter: Dataset adapter instance
        num_samples: Number of samples to test
        
    Returns:
        Dict with test results
    """
    print(f"\n{'='*60}")
    print(f"Testing RASO on {dataset_name}")
    print(f"{'='*60}")
    
    # Get dataset info
    if hasattr(adapter, 'get_class_names'):
        organ_classes = adapter.get_class_names()
    elif hasattr(adapter, 'class_names'):
        organ_classes = adapter.class_names
    else:
        # Default organ classes for CholecSeg8k (exact names from label file)
        organ_classes = [
            "black background", "abdominal wall", "liver", "gastrointestinal tract", 
            "fat", "grasper", "connective tissue", "blood", "cystic duct",
            "l-hook electrocautery", "gallbladder", "hepatic vein", "liver ligament"
        ]
    print(f"Organ classes ({len(organ_classes)}): {', '.join(organ_classes[:5])}...")
    
    # Create RASO model
    print("\nInitializing RASO model...")
    model = create_model("raso", use_cache=False, verbose=False)
    
    # Test on a few samples
    results = []
    
    for i in range(min(num_samples, adapter.total("test"))):
        print(f"\nSample {i+1}/{num_samples}:")
        
        # Get example
        example = adapter.get_example_by_global_index(i)
        image = example['image']
        gt_organs = example.get('present_organs', [])
        
        print(f"  Image size: {image.size}")
        print(f"  Ground truth organs: {gt_organs}")
        
        # Create detection prompt
        prompt = f"""
        Detect the following organs in the surgical image:
        
        {{
            "organs": {organ_classes}
        }}
        
        Return JSON with format:
        {{
            "organ_name": {{
                "present": true/false,
                "bbox": null
            }}
        }}
        """
        
        # Run RASO
        batch = [(prompt, image)]
        system_prompt = "You are an expert medical image analyst."
        
        try:
            responses = model(batch, system_prompt=system_prompt)
            
            # Parse response
            import json
            result = json.loads(responses[0])
            
            # Extract detected organs
            detected = [organ for organ, data in result.items() if data.get('present', False)]
            print(f"  RASO detected: {detected}")
            
            # Calculate accuracy for this sample with case-insensitive comparison
            # Normalize ground truth organs (convert to lowercase and replace spaces/underscores)
            def normalize_organ(name):
                return name.lower().replace('_', ' ').replace('-', ' ').strip()
            
            gt_normalized = {normalize_organ(o) for o in gt_organs} if gt_organs else set()
            detected_normalized = {normalize_organ(o) for o in detected}
            
            correct = len(gt_normalized & detected_normalized)
            total = len(gt_normalized | detected_normalized)
            accuracy = correct / total if total > 0 else 0
            
            print(f"  Sample accuracy: {accuracy:.2%}")
            
            results.append({
                'ground_truth': gt_organs,
                'detected': detected,
                'accuracy': accuracy
            })
            
        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                'ground_truth': gt_organs,
                'detected': [],
                'accuracy': 0,
                'error': str(e)
            })
    
    # Calculate overall metrics
    if results:
        avg_accuracy = sum(r['accuracy'] for r in results) / len(results)
        print(f"\n📊 Overall accuracy: {avg_accuracy:.2%}")
    else:
        avg_accuracy = 0
    
    return {
        'dataset': dataset_name,
        'num_samples': len(results),
        'avg_accuracy': avg_accuracy,
        'results': results
    }


def main():
    """Test RASO on all three datasets."""
    
    print("=" * 80)
    print("QUICK RASO TEST ON ALL DATASETS")
    print("=" * 80)
    
    # Load all three datasets
    datasets = {
        'CholecSeg8k': CholecSeg8kLocalAdapter(
            data_dir="/shared_data0/weiqiuy/datasets/cholecseg8k"
        ),
        'CholecOrgans': CholecOrgansAdapter(),
        'CholecGoNoGo': CholecGoNoGoAdapter()
    }
    
    all_results = {}
    
    for name, adapter in datasets.items():
        result = test_raso_on_dataset(name, adapter, num_samples=3)
        all_results[name] = result
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    for name, result in all_results.items():
        print(f"\n{name}:")
        print(f"  Samples tested: {result['num_samples']}")
        print(f"  Average accuracy: {result['avg_accuracy']:.2%}")
        
        # Show detected organs
        if result['results']:
            first_result = result['results'][0]
            if 'detected' in first_result:
                print(f"  Example detection: {first_result['detected'][:5]}...")
    
    print("\n✅ Quick test completed!")
    
    return all_results


if __name__ == "__main__":
    results = main()