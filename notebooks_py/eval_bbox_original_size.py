#!/usr/bin/env python
"""
Comprehensive Bounding Box Evaluation Pipeline - Using Original Image Size
This version uses the original image dimensions without rescaling.

USAGE:
    # Quick test with 5 samples
    EVAL_QUICK_TEST=true python3 eval_bbox_original_size.py
    
    # Full evaluation with all test samples
    python3 eval_bbox_original_size.py
    
    # Evaluate specific number of samples
    EVAL_NUM_SAMPLES=20 python3 eval_bbox_original_size.py
    
    # Use specific models
    EVAL_MODELS='gpt-4.1,claude-sonnet-4-20250514' python3 eval_bbox_original_size.py
    
    # Test with LLaVA only
    EVAL_MODELS='llava-hf/llava-v1.6-mistral-7b-hf' python3 eval_bbox_original_size.py
    
    # Run only few-shot evaluation (skip zero-shot)
    EVAL_SKIP_ZERO_SHOT=true python3 eval_bbox_original_size.py
    
    # Run only zero-shot evaluation (skip few-shot)
    EVAL_SKIP_FEW_SHOT=true python3 eval_bbox_original_size.py
    
    # Disable cache for fresh API calls
    EVAL_USE_CACHE=false python3 eval_bbox_original_size.py
    
    # Combine options
    EVAL_NUM_SAMPLES=10 EVAL_USE_CACHE=false EVAL_MODELS='gpt-4.1' python3 eval_bbox_original_size.py

ENVIRONMENT VARIABLES:
    EVAL_NUM_SAMPLES    - Number of samples to evaluate (default: all 200)
    EVAL_MODELS         - Comma-separated list of models (default: all 8 models - see DEFAULT_MODELS in code)
    EVAL_USE_CACHE      - Whether to use cached responses (default: true)
    EVAL_QUICK_TEST     - Quick test mode with 5 samples (default: false)
    EVAL_SKIP_ZERO_SHOT - Skip zero-shot evaluation (default: false)
    EVAL_SKIP_FEW_SHOT  - Skip few-shot evaluation (default: false)
    EVAL_DATASET        - Dataset to use: cholecseg8k, cholecorgans, cholecgonogo (default: cholecseg8k)

OUTPUT:
    Results are saved to: results/bbox_original_YYYYMMDD_HHMMSS/
    - zero_shot/MODEL/DATASET_bbox/*.json          # Per-sample results
    - fewshot_bbox/MODEL/DATASET_bbox/*.json       # Few-shot results
    - raw_results.pkl                              # Complete results pickle
    - summary.csv                                  # Summary statistics
    - metrics_comparison.txt                       # Comprehensive comparison

METRICS:
    The evaluator provides:
    - IoU@0.3, IoU@0.5, IoU@0.75: Detection at various thresholds
    - Mean IoU: Average IoU across all predictions
    - Presence Accuracy: How well organs are detected
    - Gated IoU: Combined detection + localization performance
    - Per-organ and macro/micro averages

PREREQUISITES:
    1. Install dependencies: pip install numpy torch pandas tqdm datasets pillow
    2. Create API_KEYS2.json with OpenAI, Anthropic, and Google API keys
    3. Generate balanced test set: python3 notebooks/test_unified_fewshot_simplified.ipynb

ANALYZING RESULTS:
    # View latest results
    python3 eval_bbox_analyze.py --latest
    
    # View specific results
    python3 eval_bbox_analyze.py results/bbox_original_20250901_041511

EXAMPLE OUTPUT:
    Model: gpt-4.1 | Prompt: zero_shot | Split: test | Examples used: 200
    Organ                 PresAcc  IoU@0.3  IoU@0.5  IoU@0.75  MeanIoU
    Liver                 100.0%   85.0%    70.0%    40.0%     0.623
    Gallbladder           75.0%    60.0%    45.0%    20.0%     0.412
    ...
    Macro Average         82.5%    62.3%    48.7%    28.1%     0.485
"""

import os
import sys
import json
import pickle
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add src to path
ROOT_DIR = ".."
sys.path.append(os.path.join(ROOT_DIR, 'src'))

# Load API keys
with open(f"{ROOT_DIR}/API_KEYS2.json", "r") as file:
    api_keys = json.load(file)

os.environ['OPENAI_API_KEY'] = api_keys['OPENAI_API_KEY']
os.environ['ANTHROPIC_API_KEY'] = api_keys['ANTHROPIC_API_KEY']
os.environ['GOOGLE_API_KEY'] = api_keys['GOOGLE_API_KEY']

# Import from endopoint package
from endopoint.datasets.cholecseg8k import CholecSeg8kAdapter
from endopoint.eval.bbox_evaluator import BoundingBoxEvaluator
from endopoint.fewshot.unified import UnifiedFewShotSelector

# Import dataset utilities
from datasets import load_dataset

print("✓ Environment setup complete")


def load_balanced_test_indices(dataset_name="cholecseg8k", n_samples=200):
    """Load the balanced test indices from the cached file.
    
    Args:
        dataset_name: Name of the dataset
        n_samples: Number of test samples (default: 200)
        
    Returns:
        List of test indices
    """
    data_dir = Path(ROOT_DIR) / "data_info" / f"{dataset_name}_balanced_{n_samples}"
    
    # Try different possible filenames
    possible_files = [
        data_dir / "balanced_test_indices_advanced_200.json",
        data_dir / "balanced_test_indices.json"
    ]
    
    for indices_file in possible_files:
        if indices_file.exists():
            with open(indices_file, 'r') as f:
                data = json.load(f)
            
            # Handle different formats
            if 'indices' in data:
                return data['indices']
            elif 'test_indices' in data:
                return data['test_indices']
            else:
                raise ValueError(f"Unknown format in {indices_file}")
    
    raise FileNotFoundError(
        f"Balanced test indices not found in {data_dir}\n"
        "Please run notebooks/test_unified_fewshot_simplified.ipynb first"
    )


def load_fewshot_plan(dataset_name="cholecseg8k", task="bbox", n_samples=200):
    """Load the few-shot plan for bounding box task.
    
    Args:
        dataset_name: Name of the dataset
        task: Task type ("bbox" or "pointing")
        n_samples: Number of test samples
        
    Returns:
        Dictionary with few-shot plan
    """
    data_dir = Path(ROOT_DIR) / "data_info" / f"{dataset_name}_balanced_{n_samples}"
    
    # Try different possible filenames
    possible_files = [
        data_dir / f"fewshot_plan_{task}_{n_samples}.json",
        data_dir / f"fewshot_plan_{task}_n{n_samples}.json"
    ]
    
    for plan_file in possible_files:
        if plan_file.exists():
            with open(plan_file, 'r') as f:
                return json.load(f)
    
    raise FileNotFoundError(
        f"Few-shot plan not found in {data_dir}\n"
        "Tried: {', '.join(str(f) for f in possible_files)}\n"
        "Please run notebooks/test_unified_fewshot_simplified.ipynb first"
    )


def get_original_image_size(dataset, split="train", sample_idx=0):
    """Get the original image size from the dataset.
    
    Args:
        dataset: HuggingFace dataset object
        split: Dataset split to check
        sample_idx: Sample index to check
        
    Returns:
        Tuple of (width, height)
    """
    example = dataset[split][sample_idx]
    img = example['image']  # PIL Image
    return img.size  # Returns (width, height)


def main(num_samples=None, models=None, use_cache=True, skip_zero_shot=False, skip_few_shot=False, dataset_name="cholecseg8k"):
    """Main evaluation function.
    
    Args:
        num_samples: Optional number of samples to evaluate. If None, uses all 200 test samples.
        models: Optional list of model names to evaluate. If None, uses default models.
        use_cache: Whether to use cache for model responses (default: True).
        skip_zero_shot: Whether to skip zero-shot evaluation (default: False).
        skip_few_shot: Whether to skip few-shot evaluation (default: False).
        dataset_name: Dataset to use (default: "cholecseg8k")
    """
    
    # Configuration
    DEFAULT_MODELS = [
        # Commercial APIs
        "gpt-4.1",
        "claude-sonnet-4-20250514",
        "gemini-2.0-flash",
        # Open Source VLMs
        "llava-hf/llava-v1.6-mistral-7b-hf",
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "mistralai/Pixtral-12B-2409",
        "deepseek-ai/deepseek-vl2",
        "nvidia/Llama-3.2-11B-Vision-Surgical-CholecT50"
    ]
    
    MODELS = models if models is not None else DEFAULT_MODELS
    
    print("\n" + "="*60)
    print("Starting Bounding Box Evaluation (Original Image Size)")
    print("="*60)
    print(f"Dataset: {dataset_name}")
    print(f"Models to evaluate: {', '.join(MODELS)}")
    
    # Load dataset
    print(f"\n📊 Loading {dataset_name} dataset...")
    if dataset_name == "cholecseg8k":
        dataset = load_dataset("minwoosun/CholecSeg8k")
        dataset_adapter = CholecSeg8kAdapter()
    elif dataset_name == "cholecorgans":
        # Add support for other datasets here
        raise NotImplementedError(f"Dataset {dataset_name} not yet supported")
    elif dataset_name == "cholecgonogo":
        # Add support for other datasets here
        raise NotImplementedError(f"Dataset {dataset_name} not yet supported")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    print(f"✓ Dataset loaded")
    
    # Get original image dimensions
    original_width, original_height = get_original_image_size(dataset, split="train", sample_idx=0)
    print(f"✓ Using original image dimensions: {original_width}x{original_height}")
    
    # Load test indices
    test_indices = load_balanced_test_indices(dataset_name=dataset_name, n_samples=200)
    print(f"✓ Loaded {len(test_indices)} balanced test samples")
    
    # Select subset if requested
    if num_samples is not None and num_samples < len(test_indices):
        # Use linspace to get evenly spaced indices
        selected_idx = np.linspace(0, len(test_indices) - 1, num_samples, dtype=int)
        test_indices = [test_indices[i] for i in selected_idx]
        print(f"📌 Selected {len(test_indices)} evenly spaced samples for evaluation")
    
    # Load few-shot plan if needed
    fewshot_plan = None
    if not skip_few_shot:
        try:
            fewshot_plan = load_fewshot_plan(dataset_name=dataset_name, task="bbox", n_samples=200)
            print(f"✓ Loaded few-shot plan for bounding box task")
        except FileNotFoundError as e:
            print(f"⚠️ Few-shot plan not found: {e}")
            print("Continuing with zero-shot only...")
            skip_few_shot = True
    
    # Create output directory
    use_persistent_dir = os.environ.get('EVAL_PERSISTENT_DIR', '').lower() == 'true'
    if use_persistent_dir:
        # Use a persistent directory without timestamp
        output_dir = Path(ROOT_DIR) / "results" / f"bbox_original_{dataset_name}"
        print(f"✓ Using persistent directory: {output_dir}")
    else:
        # Use timestamped directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(ROOT_DIR) / "results" / f"bbox_original_{dataset_name}_{timestamp}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize evaluator with original image dimensions
    evaluator = BoundingBoxEvaluator(
        models=MODELS,
        dataset=dataset,
        dataset_adapter=dataset_adapter,
        canvas_width=original_width,    # Use original width
        canvas_height=original_height,  # Use original height
        output_dir=output_dir,
        use_cache=use_cache,
    )
    
    # Run evaluation
    print("\n" + "="*60)
    print("Running Evaluation")
    print("="*60)
    
    results = {}
    
    # Zero-shot evaluation
    if not skip_zero_shot:
        print("\n📝 Zero-shot evaluation...")
        for model_name in tqdm(MODELS, desc="Models"):
            print(f"\n  Evaluating {model_name}...")
            model_results = evaluator.evaluate_model(
                model_name=model_name,
                test_indices=test_indices,
                prompt_type="standard",
                use_fewshot=False,
                split="train"  # Using train split where our test indices come from
            )
            
            if model_name not in results:
                results[model_name] = {}
            results[model_name]["zero_shot"] = model_results
            
            # Print summary
            print(f"    Presence Acc: {model_results['metrics']['presence_accuracy']:.1%}")
            print(f"    IoU@0.5: {model_results['metrics']['iou_at_0.5']:.1%}")
            print(f"    Mean IoU: {model_results['metrics']['mean_iou']:.3f}")
    
    # Few-shot evaluation
    if not skip_few_shot and fewshot_plan is not None:
        print("\n📝 Few-shot evaluation...")
        for model_name in tqdm(MODELS, desc="Models"):
            print(f"\n  Evaluating {model_name}...")
            model_results = evaluator.evaluate_model(
                model_name=model_name,
                test_indices=test_indices,
                prompt_type="standard",
                use_fewshot=True,
                fewshot_plan=fewshot_plan,
                split="train"
            )
            
            if model_name not in results:
                results[model_name] = {}
            results[model_name]["few_shot"] = model_results
            
            # Print summary
            print(f"    Presence Acc: {model_results['metrics']['presence_accuracy']:.1%}")
            print(f"    IoU@0.5: {model_results['metrics']['iou_at_0.5']:.1%}")
            print(f"    Mean IoU: {model_results['metrics']['mean_iou']:.3f}")
    
    # Save complete results
    results_file = output_dir / "raw_results.pkl"
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n✓ Saved raw results to {results_file}")
    
    # Generate summary CSV
    summary_data = []
    for model_name in MODELS:
        if model_name in results:
            row = {"Model": model_name}
            
            if "zero_shot" in results[model_name]:
                metrics = results[model_name]["zero_shot"]["metrics"]
                row.update({
                    "ZS_PresAcc": metrics["presence_accuracy"],
                    "ZS_IoU@0.3": metrics.get("iou_at_0.3", 0),
                    "ZS_IoU@0.5": metrics["iou_at_0.5"],
                    "ZS_IoU@0.75": metrics.get("iou_at_0.75", 0),
                    "ZS_MeanIoU": metrics["mean_iou"]
                })
            
            if "few_shot" in results[model_name]:
                metrics = results[model_name]["few_shot"]["metrics"]
                row.update({
                    "FS_PresAcc": metrics["presence_accuracy"],
                    "FS_IoU@0.3": metrics.get("iou_at_0.3", 0),
                    "FS_IoU@0.5": metrics["iou_at_0.5"],
                    "FS_IoU@0.75": metrics.get("iou_at_0.75", 0),
                    "FS_MeanIoU": metrics["mean_iou"]
                })
            
            summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = output_dir / "summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"✓ Saved summary to {summary_file}")
    
    print("\n✨ Evaluation complete!")
    
    # Print final summary
    print("\n" + "="*60)
    print("Final Summary")
    print("="*60)
    
    for model_name in MODELS:
        if model_name in results:
            print(f"\n{model_name}:")
            model_results = results[model_name]
            
            if "zero_shot" in model_results:
                metrics = model_results["zero_shot"]["metrics"]
                print(f"  Zero-shot: PresAcc={metrics['presence_accuracy']:.1%}, "
                      f"IoU@0.5={metrics['iou_at_0.5']:.1%}, "
                      f"MeanIoU={metrics['mean_iou']:.3f}")
            
            if "few_shot" in model_results:
                metrics = model_results["few_shot"]["metrics"]
                print(f"  Few-shot:  PresAcc={metrics['presence_accuracy']:.1%}, "
                      f"IoU@0.5={metrics['iou_at_0.5']:.1%}, "
                      f"MeanIoU={metrics['mean_iou']:.3f}")
    
    print(f"\n📁 Results saved to: {output_dir}")
    return results


if __name__ == "__main__":
    # Simple environment variable based configuration
    import os
    
    # Read configuration from environment variables or use defaults
    NUM_SAMPLES = os.environ.get('EVAL_NUM_SAMPLES', None)
    if NUM_SAMPLES:
        NUM_SAMPLES = int(NUM_SAMPLES)
    
    MODELS = os.environ.get('EVAL_MODELS', None)
    if MODELS:
        MODELS = MODELS.split(',')
    
    DATASET = os.environ.get('EVAL_DATASET', 'cholecseg8k')
    QUICK_TEST = os.environ.get('EVAL_QUICK_TEST', '').lower() == 'true'
    USE_CACHE = os.environ.get('EVAL_USE_CACHE', 'true').lower() != 'false'
    SKIP_ZERO_SHOT = os.environ.get('EVAL_SKIP_ZERO_SHOT', '').lower() == 'true'
    SKIP_FEW_SHOT = os.environ.get('EVAL_SKIP_FEW_SHOT', '').lower() == 'true'
    
    if QUICK_TEST:
        # Use MODELS if set, otherwise default to gpt-4.1 for quick test
        quick_models = MODELS if MODELS else ["gpt-4.1"]
        print(f"🚀 Running in quick test mode (5 samples, models: {', '.join(quick_models)})")
        main(num_samples=5, models=quick_models, use_cache=USE_CACHE,
             skip_zero_shot=SKIP_ZERO_SHOT, skip_few_shot=SKIP_FEW_SHOT,
             dataset_name=DATASET)
    else:
        # Show configuration
        print("\n" + "="*60)
        print("Bounding Box Evaluation Configuration (Original Image Size)")
        print("="*60)
        print(f"Dataset: {DATASET}")
        print(f"Samples: {NUM_SAMPLES if NUM_SAMPLES else 'all (200)'}") 
        print(f"Models: {', '.join(MODELS) if MODELS else 'all default models'}")
        print(f"Cache: {'enabled' if USE_CACHE else 'disabled'}")
        print(f"Skip zero-shot: {SKIP_ZERO_SHOT}")
        print(f"Skip few-shot: {SKIP_FEW_SHOT}")
        print("\nTo customize, either:")
        print("1. Set environment variables before running:")
        print("   export EVAL_NUM_SAMPLES=10")
        print("   export EVAL_DATASET=cholecorgans")
        print("   export EVAL_MODELS='gpt-4.1,claude-sonnet-4-20250514'")
        print("   export EVAL_USE_CACHE=false  # to disable cache")
        print("   export EVAL_SKIP_ZERO_SHOT=true  # to skip zero-shot")
        print("   export EVAL_SKIP_FEW_SHOT=true  # to skip few-shot")
        print("   export EVAL_QUICK_TEST=true")
        print("2. Or run inline: EVAL_QUICK_TEST=true python3 eval_bbox_original_size.py")
        print("="*60 + "\n")
        
        # Run evaluation
        main(num_samples=NUM_SAMPLES, models=MODELS, use_cache=USE_CACHE,
             skip_zero_shot=SKIP_ZERO_SHOT, skip_few_shot=SKIP_FEW_SHOT,
             dataset_name=DATASET)