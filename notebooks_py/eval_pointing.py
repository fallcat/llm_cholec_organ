#!/usr/bin/env python
"""
Comprehensive Pointing Evaluation Pipeline
Evaluates zero-shot and few-shot (with/without hard negatives) pointing performance
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
from endopoint.eval import PointingEvaluator

# Import dataset and few-shot utilities
from datasets import load_dataset
from few_shot_selection import (
    load_balanced_indices,
    load_fewshot_plan,
)

print("✓ Environment setup complete")


def main(num_samples=None, models=None, use_cache=True):
    """Main evaluation function.
    
    Args:
        num_samples: Optional number of samples to evaluate (uses linspace to select subset).
                    If None, uses all test samples.
        models: Optional list of model names to evaluate. If None, uses default models.
        use_cache: Whether to use cache for model responses (default: True).
                  Set to False to bypass cache (useful for testing changes).
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
    print("Starting Pointing Evaluation")
    print("="*60)
    print(f"Models to evaluate: {', '.join(MODELS)}")
    
    # Load dataset
    print("\n📊 Loading CholecSeg8k dataset...")
    dataset = load_dataset("minwoosun/CholecSeg8k")
    print(f"✓ Dataset loaded")
    
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
    output_dir = Path(ROOT_DIR) / "results" / f"pointing_{timestamp}"
    
    # Initialize evaluator
    evaluator = PointingEvaluator(
        models=MODELS,
        dataset=dataset,
        dataset_adapter=CholecSeg8kAdapter(),
        canvas_width=768,
        canvas_height=768,
        output_dir=output_dir,
        use_cache=use_cache,
    )
    
    # Run evaluation
    results = evaluator.run_full_evaluation(
        test_indices=test_indices,
        fewshot_plans=fewshot_plans,
    )
    
    print("\n✨ Evaluation complete!")
    
    # Print final summary
    print("\n" + "="*60)
    print("Final Summary")
    print("="*60)
    
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
    
    if QUICK_TEST:
        print("🚀 Running in quick test mode (5 samples, gpt-4o-mini only)")
        main(num_samples=5, models=["gpt-4o-mini"], use_cache=USE_CACHE)
    else:
        # Show configuration
        print("\n" + "="*60)
        print("Pointing Evaluation Configuration")
        print("="*60)
        print(f"Samples: {NUM_SAMPLES if NUM_SAMPLES else 'all'}")
        print(f"Models: {', '.join(MODELS) if MODELS else 'all default models'}")
        print(f"Cache: {'enabled' if USE_CACHE else 'disabled'}")
        print("\nTo customize, either:")
        print("1. In notebooks: Call main() directly with parameters")
        print("2. Set environment variables before running:")
        print("   export EVAL_NUM_SAMPLES=10")
        print("   export EVAL_MODELS='gpt-4o-mini,claude-3-5-sonnet-20241022'")
        print("   export EVAL_USE_CACHE=false  # to disable cache")
        print("   export EVAL_QUICK_TEST=true")
        print("3. Or run inline: EVAL_USE_CACHE=false python3 eval_pointing.py")
        print("="*60 + "\n")
        
        # Run evaluation
        main(num_samples=NUM_SAMPLES, models=MODELS, use_cache=USE_CACHE)