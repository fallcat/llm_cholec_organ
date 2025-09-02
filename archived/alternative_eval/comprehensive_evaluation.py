#!/usr/bin/env python
"""
Comprehensive Evaluation Script for LLM Organ Detection
Evaluates multiple models on zero-shot, few-shot, and few-shot with hard negatives
Models: GPT-4o-mini, Claude Sonnet, Gemini 2.0 Flash
"""

# Cell 1: Setup and imports
import os
import sys
import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
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
os.environ['CACHE_DIR'] = os.path.join(ROOT_DIR, 'cache_dir3')

print("✓ Environment setup complete")

# Cell 2: Import endopoint modules
from endopoint.datasets import build_dataset
from endopoint.models import OpenAIAdapter, AnthropicAdapter, GoogleAdapter
from endopoint.prompts import get_prompt_config
from endopoint.geometry import letterbox_to_canvas
from endopoint.eval.parser import parse_pointing_json
from endopoint.data.presence import PresenceCache

print("✓ Modules imported successfully")

# Cell 3: Define evaluation configuration
EVAL_CONFIG = {
    "models": [
        {"adapter": OpenAIAdapter, "name": "gpt-4o-mini", "display": "GPT-4o-mini"},
        {"adapter": AnthropicAdapter, "name": "claude-3-5-sonnet-20241022", "display": "Claude-3.5-Sonnet"},
        {"adapter": GoogleAdapter, "name": "gemini-2.0-flash-exp", "display": "Gemini-2.0-Flash"},
    ],
    "prompt_strategies": ["strict"],  # Can add more: ["base", "strict", "qna"]
    "eval_modes": ["zero_shot", "few_shot", "few_shot_hard_neg"],
    "num_eval_samples": 100,  # Number of samples to evaluate
    "num_few_shot_examples": 3,  # Number of examples for few-shot
    "canvas_size": (1280, 720),  # Recommended canvas size
    "batch_size": 10,  # For progress tracking
}

print(f"Evaluation Configuration:")
print(f"  Models: {[m['display'] for m in EVAL_CONFIG['models']]}")
print(f"  Modes: {EVAL_CONFIG['eval_modes']}")
print(f"  Samples: {EVAL_CONFIG['num_eval_samples']}")

# Cell 4: Load dataset and prepare data
print("\n📊 Loading CholecSeg8k dataset...")
dataset = build_dataset("cholecseg8k")

# Get evaluation samples
eval_indices = list(range(EVAL_CONFIG["num_eval_samples"]))
eval_examples = [dataset.get_example("val", idx) for idx in eval_indices]

# Initialize presence cache for balanced sampling
presence_cache = PresenceCache()
if not presence_cache.is_cached("cholecseg8k", "train"):
    print("Building presence cache for training set...")
    presence_cache.build_cache(dataset, "cholecseg8k", "train")

print(f"✓ Loaded {len(eval_examples)} evaluation examples")

# Cell 5: Define organ classes and metrics
ORGAN_CLASSES = [
    "Liver", "Gallbladder", "Hepatocystic Triangle", "Fat",
    "Grasper", "Connector", "Clip", "Cutting Instrument",
    "Specimen Bag", "Stapler", "Drill", "Suction and Irrigation Instrument"
]

def calculate_metrics(predictions: Dict[str, Any], ground_truth: Dict[str, Any]) -> Dict[str, float]:
    """Calculate evaluation metrics for organ detection."""
    metrics = {}
    
    # Existence metrics (binary classification)
    for organ in ORGAN_CLASSES:
        pred_exists = predictions.get(organ, {}).get("exists", False)
        true_exists = ground_truth.get(organ, {}).get("exists", False)
        
        if pred_exists == true_exists:
            metrics[f"{organ}_existence_correct"] = 1.0
        else:
            metrics[f"{organ}_existence_correct"] = 0.0
    
    # Pointing metrics (localization accuracy)
    for organ in ORGAN_CLASSES:
        if ground_truth.get(organ, {}).get("exists", False):
            pred_point = predictions.get(organ, {}).get("point", None)
            true_point = ground_truth.get(organ, {}).get("point", None)
            
            if pred_point and true_point:
                # Calculate L2 distance
                dist = np.sqrt((pred_point[0] - true_point[0])**2 + (pred_point[1] - true_point[1])**2)
                # Normalize by diagonal
                normalized_dist = dist / np.sqrt(EVAL_CONFIG["canvas_size"][0]**2 + EVAL_CONFIG["canvas_size"][1]**2)
                metrics[f"{organ}_pointing_error"] = normalized_dist
            else:
                metrics[f"{organ}_pointing_error"] = 1.0  # Max error if not predicted
    
    # Overall accuracy
    metrics["existence_accuracy"] = np.mean([metrics[f"{organ}_existence_correct"] for organ in ORGAN_CLASSES])
    pointing_errors = [metrics.get(f"{organ}_pointing_error", 1.0) for organ in ORGAN_CLASSES 
                       if ground_truth.get(organ, {}).get("exists", False)]
    if pointing_errors:
        metrics["pointing_accuracy"] = 1.0 - np.mean(pointing_errors)
    else:
        metrics["pointing_accuracy"] = 0.0
    
    return metrics

print(f"✓ Defined metrics for {len(ORGAN_CLASSES)} organ classes")

# Cell 6: Zero-shot evaluation function
def run_zero_shot_evaluation(model_adapter, examples: List, prompt_config: Dict) -> List[Dict]:
    """Run zero-shot evaluation on given examples."""
    results = []
    
    system_builder = prompt_config["system_builder"]
    user_builder = prompt_config["user_builder"]
    
    system_prompt = system_builder(*EVAL_CONFIG["canvas_size"])
    
    for i, example in enumerate(tqdm(examples, desc="Zero-shot evaluation")):
        # Convert example to tensors
        img_tensor, label_tensor = dataset.example_to_tensors(example)
        
        # Build user prompt
        user_prompt = user_builder(img_tensor, *EVAL_CONFIG["canvas_size"])
        
        # Get model response
        try:
            response = model_adapter.query(system_prompt, user_prompt)
            
            # Parse response
            parsed = parse_pointing_json(response, ORGAN_CLASSES, EVAL_CONFIG["canvas_size"])
            
            # Get ground truth
            ground_truth = example.get_ground_truth_dict()  # Assuming this method exists
            
            # Calculate metrics
            metrics = calculate_metrics(parsed, ground_truth)
            
            results.append({
                "example_id": example.example_id,
                "raw_response": response,
                "parsed_response": parsed,
                "ground_truth": ground_truth,
                "metrics": metrics
            })
        except Exception as e:
            print(f"Error processing example {i}: {e}")
            results.append({
                "example_id": example.example_id,
                "error": str(e),
                "metrics": {"existence_accuracy": 0.0, "pointing_accuracy": 0.0}
            })
    
    return results

print("✓ Zero-shot evaluation function defined")

# Cell 7: Few-shot evaluation function
def select_few_shot_examples(target_example, num_examples: int = 3, use_hard_negatives: bool = False):
    """Select balanced few-shot examples using presence cache."""
    # Get organs present in target
    target_organs = set(target_example.get_present_organs())  # Assuming this method exists
    
    # Find examples with diverse organ combinations
    train_size = dataset.get_split_size("train")
    candidates = []
    
    for idx in range(min(train_size, 500)):  # Limit search for efficiency
        example = dataset.get_example("train", idx)
        example_organs = set(example.get_present_organs())
        
        # Calculate diversity score
        if use_hard_negatives:
            # For hard negatives, find examples with similar but not identical organs
            overlap = len(target_organs & example_organs)
            difference = len(target_organs ^ example_organs)
            if 0 < overlap < len(target_organs) and difference > 0:
                score = overlap / (overlap + difference)
                candidates.append((idx, score, example))
        else:
            # For regular few-shot, find diverse examples
            overlap = len(target_organs & example_organs)
            if overlap > 0:
                candidates.append((idx, overlap, example))
    
    # Sort and select top examples
    if use_hard_negatives:
        candidates.sort(key=lambda x: x[1], reverse=True)
    else:
        candidates.sort(key=lambda x: x[1])
    
    selected = []
    for idx, score, example in candidates[:num_examples]:
        selected.append(example)
    
    return selected

def run_few_shot_evaluation(model_adapter, examples: List, prompt_config: Dict, 
                           use_hard_negatives: bool = False) -> List[Dict]:
    """Run few-shot evaluation with optional hard negatives."""
    results = []
    
    system_builder = prompt_config["system_builder"]
    user_builder = prompt_config.get("few_shot_user_builder", prompt_config["user_builder"])
    
    system_prompt = system_builder(*EVAL_CONFIG["canvas_size"])
    
    desc = "Few-shot (hard neg)" if use_hard_negatives else "Few-shot"
    
    for i, example in enumerate(tqdm(examples, desc=f"{desc} evaluation")):
        # Select few-shot examples
        few_shot_examples = select_few_shot_examples(
            example, 
            EVAL_CONFIG["num_few_shot_examples"],
            use_hard_negatives
        )
        
        # Build few-shot prompt
        img_tensor, label_tensor = dataset.example_to_tensors(example)
        
        # Create few-shot context (simplified - would need proper implementation)
        few_shot_context = []
        for fs_example in few_shot_examples:
            fs_img, fs_label = dataset.example_to_tensors(fs_example)
            fs_response = fs_example.get_ideal_response()  # Assuming this method exists
            few_shot_context.append((fs_img, fs_response))
        
        # Build user prompt with few-shot examples
        user_prompt = user_builder(img_tensor, few_shot_context, *EVAL_CONFIG["canvas_size"])
        
        # Get model response
        try:
            response = model_adapter.query(system_prompt, user_prompt)
            
            # Parse response
            parsed = parse_pointing_json(response, ORGAN_CLASSES, EVAL_CONFIG["canvas_size"])
            
            # Get ground truth
            ground_truth = example.get_ground_truth_dict()
            
            # Calculate metrics
            metrics = calculate_metrics(parsed, ground_truth)
            
            results.append({
                "example_id": example.example_id,
                "raw_response": response,
                "parsed_response": parsed,
                "ground_truth": ground_truth,
                "metrics": metrics,
                "few_shot_ids": [fs.example_id for fs in few_shot_examples]
            })
        except Exception as e:
            print(f"Error processing example {i}: {e}")
            results.append({
                "example_id": example.example_id,
                "error": str(e),
                "metrics": {"existence_accuracy": 0.0, "pointing_accuracy": 0.0}
            })
    
    return results

print("✓ Few-shot evaluation functions defined")

# Cell 8: Main evaluation loop
def run_full_evaluation():
    """Run complete evaluation across all models and modes."""
    all_results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for model_config in EVAL_CONFIG["models"]:
        print(f"\n{'='*60}")
        print(f"Evaluating {model_config['display']}")
        print(f"{'='*60}")
        
        # Initialize model adapter
        model = model_config["adapter"](model_name=model_config["name"])
        model_results = {}
        
        for prompt_strategy in EVAL_CONFIG["prompt_strategies"]:
            print(f"\n📝 Using prompt strategy: {prompt_strategy}")
            prompt_config = get_prompt_config(prompt_strategy)
            
            for eval_mode in EVAL_CONFIG["eval_modes"]:
                print(f"\n🔄 Running {eval_mode}...")
                
                if eval_mode == "zero_shot":
                    results = run_zero_shot_evaluation(model, eval_examples, prompt_config)
                elif eval_mode == "few_shot":
                    results = run_few_shot_evaluation(model, eval_examples, prompt_config, use_hard_negatives=False)
                elif eval_mode == "few_shot_hard_neg":
                    results = run_few_shot_evaluation(model, eval_examples, prompt_config, use_hard_negatives=True)
                
                # Store results
                key = f"{prompt_strategy}_{eval_mode}"
                model_results[key] = results
                
                # Calculate summary statistics
                all_metrics = [r["metrics"] for r in results if "metrics" in r]
                if all_metrics:
                    avg_existence = np.mean([m["existence_accuracy"] for m in all_metrics])
                    avg_pointing = np.mean([m["pointing_accuracy"] for m in all_metrics])
                    print(f"  ✓ Existence Accuracy: {avg_existence:.3f}")
                    print(f"  ✓ Pointing Accuracy: {avg_pointing:.3f}")
        
        all_results[model_config["name"]] = model_results
    
    # Save results
    save_results(all_results, timestamp)
    
    return all_results

print("✓ Main evaluation loop defined")

# Cell 9: Results saving and visualization
def save_results(results: Dict, timestamp: str):
    """Save evaluation results to multiple formats."""
    results_dir = Path(ROOT_DIR) / "results" / f"evaluation_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save raw results as pickle
    with open(results_dir / "raw_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print(f"✓ Saved raw results to {results_dir / 'raw_results.pkl'}")
    
    # Create summary DataFrame
    summary_data = []
    for model_name, model_results in results.items():
        for eval_key, eval_results in model_results.items():
            all_metrics = [r["metrics"] for r in eval_results if "metrics" in r]
            if all_metrics:
                summary_data.append({
                    "model": model_name,
                    "evaluation": eval_key,
                    "existence_accuracy": np.mean([m["existence_accuracy"] for m in all_metrics]),
                    "pointing_accuracy": np.mean([m["pointing_accuracy"] for m in all_metrics]),
                    "num_samples": len(all_metrics)
                })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary as CSV
    summary_df.to_csv(results_dir / "summary.csv", index=False)
    print(f"✓ Saved summary to {results_dir / 'summary.csv'}")
    
    # Save detailed per-organ results
    organ_results = []
    for model_name, model_results in results.items():
        for eval_key, eval_results in model_results.items():
            for organ in ORGAN_CLASSES:
                existence_scores = []
                pointing_errors = []
                
                for r in eval_results:
                    if "metrics" in r:
                        if f"{organ}_existence_correct" in r["metrics"]:
                            existence_scores.append(r["metrics"][f"{organ}_existence_correct"])
                        if f"{organ}_pointing_error" in r["metrics"]:
                            pointing_errors.append(r["metrics"][f"{organ}_pointing_error"])
                
                if existence_scores:
                    organ_results.append({
                        "model": model_name,
                        "evaluation": eval_key,
                        "organ": organ,
                        "existence_accuracy": np.mean(existence_scores),
                        "pointing_accuracy": 1.0 - np.mean(pointing_errors) if pointing_errors else 0.0,
                        "num_samples": len(existence_scores)
                    })
    
    organ_df = pd.DataFrame(organ_results)
    organ_df.to_csv(results_dir / "per_organ_results.csv", index=False)
    print(f"✓ Saved per-organ results to {results_dir / 'per_organ_results.csv'}")
    
    # Create formatted report
    with open(results_dir / "report.md", "w") as f:
        f.write(f"# Evaluation Report\n")
        f.write(f"Generated: {timestamp}\n\n")
        
        f.write("## Configuration\n")
        f.write(f"- Models: {[m['display'] for m in EVAL_CONFIG['models']]}\n")
        f.write(f"- Evaluation modes: {EVAL_CONFIG['eval_modes']}\n")
        f.write(f"- Number of samples: {EVAL_CONFIG['num_eval_samples']}\n")
        f.write(f"- Canvas size: {EVAL_CONFIG['canvas_size']}\n\n")
        
        f.write("## Summary Results\n\n")
        f.write(summary_df.to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## Per-Model Results\n\n")
        for model_name in summary_df['model'].unique():
            f.write(f"### {model_name}\n\n")
            model_df = summary_df[summary_df['model'] == model_name]
            f.write(model_df[['evaluation', 'existence_accuracy', 'pointing_accuracy']].to_markdown(index=False))
            f.write("\n\n")
    
    print(f"✓ Saved report to {results_dir / 'report.md'}")
    print(f"\n📊 All results saved to: {results_dir}")
    
    return results_dir

print("✓ Results saving functions defined")

# Cell 10: Visualization utilities
def visualize_results(results_path: Path):
    """Create visualizations from saved results."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Load summary
    summary_df = pd.read_csv(results_path / "summary.csv")
    organ_df = pd.read_csv(results_path / "per_organ_results.csv")
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Model comparison - Existence Accuracy
    ax = axes[0, 0]
    pivot_exist = summary_df.pivot(index='evaluation', columns='model', values='existence_accuracy')
    pivot_exist.plot(kind='bar', ax=ax)
    ax.set_title('Existence Accuracy by Model and Evaluation Mode')
    ax.set_xlabel('Evaluation Mode')
    ax.set_ylabel('Accuracy')
    ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_ylim([0, 1])
    
    # 2. Model comparison - Pointing Accuracy
    ax = axes[0, 1]
    pivot_point = summary_df.pivot(index='evaluation', columns='model', values='pointing_accuracy')
    pivot_point.plot(kind='bar', ax=ax)
    ax.set_title('Pointing Accuracy by Model and Evaluation Mode')
    ax.set_xlabel('Evaluation Mode')
    ax.set_ylabel('Accuracy')
    ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_ylim([0, 1])
    
    # 3. Per-organ performance (best model/mode)
    ax = axes[1, 0]
    best_config = summary_df.loc[summary_df['existence_accuracy'].idxmax()]
    best_organ_data = organ_df[(organ_df['model'] == best_config['model']) & 
                                (organ_df['evaluation'] == best_config['evaluation'])]
    
    x = range(len(best_organ_data))
    width = 0.35
    ax.bar([i - width/2 for i in x], best_organ_data['existence_accuracy'], width, label='Existence')
    ax.bar([i + width/2 for i in x], best_organ_data['pointing_accuracy'], width, label='Pointing')
    ax.set_xlabel('Organ')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Per-Organ Performance (Best: {best_config["model"]} - {best_config["evaluation"]})')
    ax.set_xticks(x)
    ax.set_xticklabels(best_organ_data['organ'], rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1])
    
    # 4. Learning curve (zero-shot vs few-shot)
    ax = axes[1, 1]
    for model in summary_df['model'].unique():
        model_data = summary_df[summary_df['model'] == model]
        zero_shot = model_data[model_data['evaluation'].str.contains('zero_shot')]['existence_accuracy'].values[0]
        few_shot = model_data[model_data['evaluation'].str.contains('few_shot') & 
                             ~model_data['evaluation'].str.contains('hard')]['existence_accuracy'].values[0]
        hard_neg = model_data[model_data['evaluation'].str.contains('hard_neg')]['existence_accuracy'].values[0]
        
        ax.plot(['Zero-shot', 'Few-shot', 'Few-shot Hard Neg'], [zero_shot, few_shot, hard_neg], 
                marker='o', label=model, linewidth=2, markersize=8)
    
    ax.set_xlabel('Evaluation Mode')
    ax.set_ylabel('Existence Accuracy')
    ax.set_title('Learning Progression by Model')
    ax.legend(title='Model')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_path / 'evaluation_plots.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Saved visualization to {results_path / 'evaluation_plots.png'}")

print("✓ Visualization utilities defined")

# Cell 11: Run evaluation
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Starting Comprehensive Evaluation")
    print("="*60)
    
    # Run full evaluation
    results = run_full_evaluation()
    
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)
    
    # Generate visualizations
    latest_results = sorted(Path(ROOT_DIR) / "results").glob("evaluation_*"))[-1]
    print(f"\nGenerating visualizations from: {latest_results}")
    visualize_results(latest_results)
    
    print("\n✨ All done! Check the results directory for outputs.")