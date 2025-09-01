#!/usr/bin/env python
"""
Comprehensive Pointing Evaluation Pipeline - Using Original Image Size
This version uses the original image dimensions without rescaling.

USAGE:
    # Quick test with 5 samples
    EVAL_QUICK_TEST=true python3 eval_pointing_original_size.py
    
    # Full evaluation with all test samples
    python3 eval_pointing_original_size.py
    
    # Evaluate specific number of samples
    EVAL_NUM_SAMPLES=20 python3 eval_pointing_original_size.py
    
    # Use specific models
    EVAL_MODELS='gpt-4o-mini,claude-3-5-sonnet-20241022' python3 eval_pointing_original_size.py
    
    # Disable cache for fresh API calls
    EVAL_USE_CACHE=false python3 eval_pointing_original_size.py
    
    # Disable enhanced metrics (not recommended)
    EVAL_USE_ENHANCED=false python3 eval_pointing_original_size.py
    
    # Combine options
    EVAL_NUM_SAMPLES=10 EVAL_USE_CACHE=false EVAL_MODELS='gpt-4o-mini' python3 eval_pointing_original_size.py

ENVIRONMENT VARIABLES:
    EVAL_NUM_SAMPLES    - Number of samples to evaluate (default: all)
    EVAL_MODELS         - Comma-separated list of models (default: gpt-4o-mini, claude-3-5-sonnet-20241022, gemini-2.0-flash-exp)
    EVAL_USE_CACHE      - Whether to use cached responses (default: true)
    EVAL_USE_ENHANCED   - Whether to use enhanced metrics (default: true)
    EVAL_QUICK_TEST     - Quick test mode with 5 samples (default: false)

OUTPUT:
    Results are saved to: results/pointing_original_YYYYMMDD_HHMMSS/
    - zero_shot/MODEL/cholecseg8k_pointing/*.json      # Per-sample results
    - fewshot_standard/MODEL/cholecseg8k_pointing/*.json
    - fewshot_hard_negatives/MODEL/cholecseg8k_pointing/*.json
    - raw_results.pkl                                   # Complete results pickle
    - summary.csv                                       # Summary statistics
    - metrics_comparison.txt                            # Comprehensive comparison

METRICS:
    The enhanced evaluator provides:
    - Confusion matrix (TP, FN, TN, FP) per organ
    - Presence Accuracy: How well organs are detected
    - Hit@Point|Present: Pointing accuracy when organ is detected
    - Gated Accuracy: Combined detection + pointing performance
    - Macro/Micro averages across all organs

PREREQUISITES:
    1. Install dependencies: pip install numpy torch pandas tqdm datasets pillow
    2. Create API_KEYS2.json with OpenAI, Anthropic, and Google API keys
    3. Prepare few-shot examples: python3 prepare_fewshot_examples.py

ANALYZING RESULTS:
    # View latest results
    python3 eval_pointing_analyze.py --latest
    
    # View specific results
    python3 eval_pointing_analyze.py results/pointing_original_20250901_041511

CACHE MANAGEMENT:
    # Clear all caches before evaluation
    python3 clear_cache.py
    
    # Debug cache issues
    python3 debug_cache.py

EXAMPLE OUTPUT:
    Model: gpt-4o-mini | Prompt: zero_shot | Split: train | Examples used: 10
    ID  Label                     TP   FN   TN   FP   PresenceAcc   Hit@Pt|Pres   GatedAcc
     2  Liver                      10    0    0    0   100.00%       40.00%        40.00%
    ...
    Macro PresenceAcc= 57.50%   Macro Hit@Point|Present= 14.94%   Macro GatedAcc= 55.00%
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

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
from endopoint.eval.enhanced_evaluator import EnhancedPointingEvaluator

# Import dataset and few-shot utilities
from datasets import load_dataset
from few_shot_selection import (
    load_balanced_indices,
    load_fewshot_plan,
)

print("✓ Environment setup complete")


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


def main(num_samples=None, models=None, use_cache=True, use_enhanced=True):
    """Main evaluation function.
    
    Args:
        num_samples: Optional number of samples to evaluate (uses linspace to select subset).
                    If None, uses all test samples.
        models: Optional list of model names to evaluate. If None, uses default models.
        use_cache: Whether to use cache for model responses (default: True).
                  Set to False to bypass cache (useful for testing changes).
        use_enhanced: Whether to use enhanced metrics (default: True).
                     Enhanced metrics match the notebook's comprehensive output.
    """
    
    # Configuration
    DEFAULT_MODELS = [
        "gpt-4o-mini",
        "claude-3-5-sonnet-20241022",
        "gemini-2.0-flash-exp"
    ]
    
    MODELS = models if models is not None else DEFAULT_MODELS
    
    # Data directories
    data_dir = Path(ROOT_DIR) / "data_info" / "cholecseg8k"
    
    # Test indices file
    test_indices_file = str(data_dir / "balanced_indices_train_100.json")
    
    # Few-shot plan files
    fewshot_plan_files = {
        "standard": str(data_dir / "fewshot_plan_train_pos1_neg1_seed43_excl100.json"),
        "hard_negatives": str(data_dir / "fewshot_plan_train_pos1_neg1_nearmiss1_seed45_excl100.json"),
    }
    
    # Check if files exist
    if not Path(test_indices_file).exists():
        print(f"❌ Test indices file not found: {test_indices_file}")
        print("Please run prepare_fewshot_examples.py first")
        return
    
    for plan_name, plan_file in fewshot_plan_files.items():
        if not Path(plan_file).exists():
            print(f"❌ Few-shot plan file not found: {plan_file}")
            print("Please run prepare_fewshot_examples.py first")
            return
    
    print("\n" + "="*60)
    print("Starting Pointing Evaluation (Original Image Size)")
    print("="*60)
    print(f"Models to evaluate: {', '.join(MODELS)}")
    
    # Load dataset
    print("\n📊 Loading CholecSeg8k dataset...")
    dataset = load_dataset("minwoosun/CholecSeg8k")
    print(f"✓ Dataset loaded")
    
    # Get original image dimensions
    original_width, original_height = get_original_image_size(dataset, split="train", sample_idx=0)
    print(f"✓ Using original image dimensions: {original_width}x{original_height}")
    
    # Load test indices
    test_indices = load_balanced_indices(test_indices_file)
    print(f"✓ Loaded {len(test_indices)} test samples")
    
    # Select subset using linspace if requested
    if num_samples is not None and num_samples < len(test_indices):
        import numpy as np
        # Use linspace to get evenly spaced indices
        selected_idx = np.linspace(0, len(test_indices) - 1, num_samples, dtype=int)
        test_indices = [test_indices[i] for i in selected_idx]
        print(f"📌 Selected {len(test_indices)} evenly spaced samples for evaluation")
    
    # Load few-shot plans
    fewshot_plans = {}
    for plan_name, plan_file in fewshot_plan_files.items():
        fewshot_plans[plan_name] = load_fewshot_plan(plan_file)
        print(f"✓ Loaded few-shot plan: {plan_name}")
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(ROOT_DIR) / "results" / f"pointing_original_{timestamp}"
    
    # Initialize evaluator with original image dimensions
    evaluator = EnhancedPointingEvaluator(
        models=MODELS,
        dataset=dataset,
        dataset_adapter=CholecSeg8kAdapter(),
        canvas_width=original_width,    # Use original width
        canvas_height=original_height,  # Use original height
        output_dir=output_dir,
        use_cache=use_cache,
    )
    
    # Run evaluation with enhanced metrics
    if use_enhanced:
        results = evaluator.run_full_evaluation_enhanced(
            test_indices=test_indices,
            fewshot_plans=fewshot_plans,
            split="train",  # Using train split for test indices
        )
    else:
        results = evaluator.run_full_evaluation(
            test_indices=test_indices,
            fewshot_plans=fewshot_plans,
        )
    
    print("\n✨ Evaluation complete!")
    
    # Print final summary
    print("\n" + "="*60)
    print("Final Summary")
    print("="*60)
    
    if use_enhanced:
        # Enhanced summary with comprehensive metrics
        for model_name in MODELS:
            print(f"\n{model_name}:")
            model_results = results[model_name]
            for eval_type in ["zero_shot", "few_shot_standard", "few_shot_hard_negatives"]:
                if eval_type in model_results and "metrics_rows" in model_results[eval_type]:
                    rows = model_results[eval_type]["metrics_rows"]
                    totals = model_results[eval_type]["metrics_totals"]
                    
                    # Calculate macro metrics
                    pres_accs = [r["PresenceAcc"] for r in rows if r["PresenceAcc"] is not None]
                    gated_accs = [r["GatedAcc"] for r in rows if r["GatedAcc"] is not None]
                    hit_rates = [r["Hit@Point|Present"] for r in rows if r["Hit@Point|Present"] is not None]
                    f1_scores = [r["F1"] for r in rows if r["F1"] is not None]
                    
                    import numpy as np
                    macro_presence = np.mean(pres_accs) if pres_accs else 0.0
                    macro_gated = np.mean(gated_accs) if gated_accs else 0.0
                    macro_hit = np.mean(hit_rates) if hit_rates else 0.0
                    macro_f1 = np.mean(f1_scores) if f1_scores else 0.0
                    
                    print(f"  {eval_type:25} PresAcc: {macro_presence:.3f}, GatedAcc: {macro_gated:.3f}, Hit@Pt: {macro_hit:.3f}, F1: {macro_f1:.3f}")
    else:
        # Standard summary
        for model_name in MODELS:
            print(f"\n{model_name}:")
            model_results = results[model_name]
            for eval_type in ["zero_shot", "few_shot_standard", "few_shot_hard_negatives"]:
                if eval_type in model_results:
                    metrics = model_results[eval_type]["metrics"]
                    print(f"  {eval_type:25} Acc: {metrics['overall_accuracy']:.3f}, F1: {metrics['avg_f1']:.3f}")


if __name__ == "__main__":
    # Simple environment variable based configuration
    # Works the same in notebooks and command line
    import os
    
    # Read configuration from environment variables or use defaults
    NUM_SAMPLES = os.environ.get('EVAL_NUM_SAMPLES', None)
    if NUM_SAMPLES:
        NUM_SAMPLES = int(NUM_SAMPLES)
    
    MODELS = os.environ.get('EVAL_MODELS', None)
    if MODELS:
        MODELS = MODELS.split(',')
    
    QUICK_TEST = os.environ.get('EVAL_QUICK_TEST', '').lower() == 'true'
    USE_CACHE = os.environ.get('EVAL_USE_CACHE', 'true').lower() != 'false'
    USE_ENHANCED = os.environ.get('EVAL_USE_ENHANCED', 'true').lower() != 'false'
    
    if QUICK_TEST:
        print("🚀 Running in quick test mode (5 samples, gpt-4o-mini only)")
        main(num_samples=5, models=["gpt-4o-mini"], use_cache=USE_CACHE, use_enhanced=USE_ENHANCED)
    else:
        # Show configuration
        print("\n" + "="*60)
        print("Pointing Evaluation Configuration (Original Image Size)")
        print("="*60)
        print(f"Samples: {NUM_SAMPLES if NUM_SAMPLES else 'all'}")
        print(f"Models: {', '.join(MODELS) if MODELS else 'all default models'}")
        print(f"Cache: {'enabled' if USE_CACHE else 'disabled'}")
        print(f"Enhanced metrics: {'enabled' if USE_ENHANCED else 'disabled'}")
        print("\nTo customize, either:")
        print("1. In notebooks: Call main() directly with parameters")
        print("2. Set environment variables before running:")
        print("   export EVAL_NUM_SAMPLES=10")
        print("   export EVAL_MODELS='gpt-4o-mini,claude-3-5-sonnet-20241022'")
        print("   export EVAL_USE_CACHE=false  # to disable cache")
        print("   export EVAL_USE_ENHANCED=false  # to disable enhanced metrics")
        print("   export EVAL_QUICK_TEST=true")
        print("3. Or run inline: EVAL_USE_CACHE=false python3 eval_pointing_original_size.py")
        print("="*60 + "\n")
        
        # Run evaluation
        main(num_samples=NUM_SAMPLES, models=MODELS, use_cache=USE_CACHE, use_enhanced=USE_ENHANCED)