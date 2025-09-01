#!/usr/bin/env python
"""
Simple Evaluation Script using existing LLM and dataset utilities
Evaluates GPT-4o-mini, Claude Sonnet, and Gemini on organ detection tasks
"""

# Cell 1: Setup and imports
import os
import sys
import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import pandas as pd

# Add src to path
ROOT_DIR = ".."
sys.path.append(os.path.join(ROOT_DIR, 'src'))

# Load API keys
with open(f"{ROOT_DIR}/API_KEYS2.json", "r") as file:
    api_keys = json.load(file)

os.environ['OPENAI_API_KEY'] = api_keys['OPENAI_API_KEY']
os.environ['ANTHROPIC_API_KEY'] = api_keys['ANTHROPIC_API_KEY']
os.environ['GOOGLE_API_KEY'] = api_keys['GOOGLE_API_KEY']

# Import existing modules
from datasets import load_dataset
from llms import load_model
from cholecseg8k_utils import example_to_tensors, presence_qas_from_example, labels_to_presence_vector
from prompts.explanations import load_cholec_prompt

print("✓ Environment setup complete")

# Cell 2: Configuration
MODELS = [
    "gpt-4o-mini",
    "claude-3-5-sonnet-20241022", 
    "gemini-2.0-flash-exp"
]

ORGAN_CLASSES = [
    "Liver", "Gallbladder", "Hepatocystic Triangle", "Fat",
    "Grasper", "Connector", "Clip", "Cutting Instrument",
    "Specimen Bag", "Stapler", "Drill", "Suction and Irrigation Instrument"
]

NUM_EVAL_SAMPLES = 50  # Adjust based on budget
CANVAS_SIZE = (1280, 720)

print(f"Models to evaluate: {MODELS}")
print(f"Number of samples: {NUM_EVAL_SAMPLES}")

# Cell 3: Load dataset
print("\n📊 Loading CholecSeg8k dataset...")
dataset = load_dataset("minwoosun/CholecSeg8k")

# Get validation samples (simple random selection for now)
# You could implement balanced sampling here if needed
import random
random.seed(42)
val_size = len(dataset['val'])
val_indices = random.sample(range(val_size), min(NUM_EVAL_SAMPLES, val_size))

eval_samples = [dataset['val'][i] for i in val_indices]
print(f"✓ Loaded {len(eval_samples)} evaluation samples")

# Cell 4: Zero-shot evaluation
def run_zero_shot(model_name, samples, prompt_type="strict"):
    """Run zero-shot evaluation using existing LLM utilities."""
    print(f"\n🔄 Running zero-shot with {model_name}...")
    
    # Load model
    model = load_model(model_name)
    
    # Load prompt template - the function returns the prompts based on baseline type
    prompts = load_cholec_prompt(baseline=prompt_type)
    
    results = []
    for sample in tqdm(samples, desc=f"Zero-shot {model_name}"):
        # Convert example to tensors
        img_t, lab_t = example_to_tensors(sample)
        
        # Get presence vector from labels for ground truth
        y_true = labels_to_presence_vector(lab_t)  # [12] tensor
        
        # For each organ, ask the model
        predictions = {}
        for organ_idx, organ_name in enumerate(ORGAN_CLASSES):
            # Get the prompt for this organ
            if prompt_type in prompts and organ_name in prompts[prompt_type]:
                system_prompt, user_prompt = prompts[prompt_type][organ_name]
            else:
                # Fallback to a simple prompt
                system_prompt = "You are an expert medical image analyst."
                user_prompt = f"Is there a {organ_name} visible in this surgical image? Answer only 'yes' or 'no'."
            
            # Query model (the model handles image encoding internally)
            response = model(system_prompt, user_prompt, sample['image'])
            
            # Parse yes/no response
            is_present = 1 if 'yes' in response.lower() else 0
            predictions[organ_name] = {"present": is_present, "raw": response}
        
        results.append({
            "sample_id": sample.get('id', val_indices[len(results)]),
            "predictions": predictions,
            "ground_truth": {ORGAN_CLASSES[i]: int(y_true[i].item()) for i in range(len(ORGAN_CLASSES))}
        })
    
    return results

# Cell 5: Few-shot evaluation
def get_few_shot_examples(target_sample, all_samples, n=3):
    """Select diverse few-shot examples."""
    # Simple selection: take first n samples different from target
    examples = []
    for s in all_samples:
        if s.get('id') != target_sample.get('id'):
            examples.append(s)
            if len(examples) >= n:
                break
    return examples

def run_few_shot(model_name, samples, prompt_type="strict", n_examples=3):
    """Run few-shot evaluation."""
    print(f"\n🔄 Running {n_examples}-shot with {model_name}...")
    
    model = load_model(model_name)
    system_prompt, user_prompt_template = load_cholec_prompt(
        prompt_type=prompt_type,
        canvas_size=CANVAS_SIZE,
        few_shot=True
    )
    
    # Get training samples for few-shot examples
    train_samples = dataset['train'][:100]  # Use first 100 train samples
    
    results = []
    for sample in tqdm(samples, desc=f"Few-shot {model_name}"):
        # Get few-shot examples
        examples = get_few_shot_examples(sample, train_samples, n=n_examples)
        
        # Build few-shot prompt
        few_shot_context = "\n\nExamples:\n"
        for i, ex in enumerate(examples, 1):
            few_shot_context += f"\nExample {i}:\n"
            few_shot_context += f"Image: [Image {i}]\n"
            # Add expected output format
            few_shot_context += json.dumps({
                organ: {"present": 1 if organ in ex.get('annotations', {}) else 0}
                for organ in ORGAN_CLASSES[:3]  # Show format with first 3 organs
            }, indent=2)
        
        # Combine with main prompt
        full_prompt = user_prompt_template.format(image=sample['image']) + few_shot_context
        
        response = model.query(system_prompt, full_prompt)
        
        try:
            parsed = json.loads(response)
        except:
            parsed = {"error": "Failed to parse JSON", "raw": response}
        
        results.append({
            "sample_id": sample.get('id', 'unknown'),
            "response": parsed,
            "ground_truth": sample.get('annotations', {}),
            "few_shot_ids": [ex.get('id') for ex in examples]
        })
    
    return results

# Cell 6: Few-shot with hard negatives
def get_hard_negative_examples(target_sample, all_samples, n=3):
    """Select hard negative examples (similar but different organs)."""
    target_organs = set(target_sample.get('annotations', {}).keys())
    
    candidates = []
    for s in all_samples:
        sample_organs = set(s.get('annotations', {}).keys())
        # Hard negative: some overlap but not identical
        overlap = len(target_organs & sample_organs)
        if 0 < overlap < len(target_organs):
            similarity = overlap / len(target_organs | sample_organs)
            candidates.append((similarity, s))
    
    # Sort by similarity and take top n
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in candidates[:n]]

def run_few_shot_hard_neg(model_name, samples, prompt_type="strict", n_examples=3):
    """Run few-shot with hard negatives."""
    print(f"\n🔄 Running few-shot (hard negatives) with {model_name}...")
    
    model = load_model(model_name)
    system_prompt, user_prompt_template = load_cholec_prompt(
        prompt_type=prompt_type,
        canvas_size=CANVAS_SIZE,
        few_shot=True
    )
    
    train_samples = dataset['train'][:200]  # Use more samples for finding hard negatives
    
    results = []
    for sample in tqdm(samples, desc=f"Few-shot HN {model_name}"):
        # Get hard negative examples
        examples = get_hard_negative_examples(sample, train_samples, n=n_examples)
        
        # Build few-shot prompt with emphasis on differences
        few_shot_context = "\n\nCarefully note these examples:\n"
        for i, ex in enumerate(examples, 1):
            few_shot_context += f"\nExample {i} (pay attention to differences):\n"
            few_shot_context += f"Image: [Image {i}]\n"
            few_shot_context += json.dumps({
                organ: {"present": 1 if organ in ex.get('annotations', {}) else 0}
                for organ in ORGAN_CLASSES[:3]
            }, indent=2)
        
        full_prompt = user_prompt_template.format(image=sample['image']) + few_shot_context
        
        response = model.query(system_prompt, full_prompt)
        
        try:
            parsed = json.loads(response)
        except:
            parsed = {"error": "Failed to parse JSON", "raw": response}
        
        results.append({
            "sample_id": sample.get('id', 'unknown'),
            "response": parsed,
            "ground_truth": sample.get('annotations', {}),
            "hard_negative_ids": [ex.get('id') for ex in examples]
        })
    
    return results

# Cell 7: Calculate metrics
def calculate_metrics(results):
    """Calculate accuracy metrics from results."""
    metrics = {
        "total_samples": len(results),
        "parse_success": 0,
        "organ_metrics": {}
    }
    
    for organ in ORGAN_CLASSES:
        metrics["organ_metrics"][organ] = {
            "correct": 0,
            "total": 0,
            "accuracy": 0.0
        }
    
    for result in results:
        response = result["response"]
        ground_truth = result["ground_truth"]
        
        # Check if parsing succeeded
        if "error" not in response:
            metrics["parse_success"] += 1
            
            # Check each organ
            for organ in ORGAN_CLASSES:
                gt_present = organ in ground_truth and ground_truth[organ].get("present", False)
                
                if organ in response:
                    pred_present = response[organ].get("present", 0) == 1
                    
                    metrics["organ_metrics"][organ]["total"] += 1
                    if pred_present == gt_present:
                        metrics["organ_metrics"][organ]["correct"] += 1
    
    # Calculate accuracies
    metrics["parse_rate"] = metrics["parse_success"] / metrics["total_samples"] if metrics["total_samples"] > 0 else 0
    
    for organ in ORGAN_CLASSES:
        organ_stats = metrics["organ_metrics"][organ]
        if organ_stats["total"] > 0:
            organ_stats["accuracy"] = organ_stats["correct"] / organ_stats["total"]
    
    # Overall accuracy
    total_correct = sum(m["correct"] for m in metrics["organ_metrics"].values())
    total_predictions = sum(m["total"] for m in metrics["organ_metrics"].values())
    metrics["overall_accuracy"] = total_correct / total_predictions if total_predictions > 0 else 0
    
    return metrics

# Cell 8: Run full evaluation
def run_full_evaluation():
    """Run complete evaluation pipeline."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(ROOT_DIR) / "results" / f"eval_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    summary_data = []
    
    for model_name in MODELS:
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")
        
        model_results = {}
        
        # Zero-shot
        zero_shot_results = run_zero_shot(model_name, eval_samples)
        zero_shot_metrics = calculate_metrics(zero_shot_results)
        model_results["zero_shot"] = {
            "results": zero_shot_results,
            "metrics": zero_shot_metrics
        }
        print(f"Zero-shot accuracy: {zero_shot_metrics['overall_accuracy']:.3f}")
        
        # Few-shot
        few_shot_results = run_few_shot(model_name, eval_samples, n_examples=3)
        few_shot_metrics = calculate_metrics(few_shot_results)
        model_results["few_shot"] = {
            "results": few_shot_results,
            "metrics": few_shot_metrics
        }
        print(f"Few-shot accuracy: {few_shot_metrics['overall_accuracy']:.3f}")
        
        # Few-shot with hard negatives
        hard_neg_results = run_few_shot_hard_neg(model_name, eval_samples, n_examples=3)
        hard_neg_metrics = calculate_metrics(hard_neg_results)
        model_results["few_shot_hard_neg"] = {
            "results": hard_neg_results,
            "metrics": hard_neg_metrics
        }
        print(f"Few-shot (hard neg) accuracy: {hard_neg_metrics['overall_accuracy']:.3f}")
        
        all_results[model_name] = model_results
        
        # Add to summary
        summary_data.append({
            "model": model_name,
            "zero_shot_acc": zero_shot_metrics['overall_accuracy'],
            "few_shot_acc": few_shot_metrics['overall_accuracy'],
            "few_shot_hn_acc": hard_neg_metrics['overall_accuracy'],
            "parse_rate": zero_shot_metrics['parse_rate']
        })
    
    # Save results
    print(f"\n💾 Saving results to {results_dir}")
    
    # Save raw results
    with open(results_dir / "raw_results.pkl", "wb") as f:
        pickle.dump(all_results, f)
    
    # Save summary
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(results_dir / "summary.csv", index=False)
    
    # Create markdown report
    with open(results_dir / "report.md", "w") as f:
        f.write(f"# Evaluation Report\n")
        f.write(f"Generated: {timestamp}\n\n")
        f.write(f"## Configuration\n")
        f.write(f"- Models: {MODELS}\n")
        f.write(f"- Samples: {NUM_EVAL_SAMPLES}\n")
        f.write(f"- Canvas: {CANVAS_SIZE}\n\n")
        f.write(f"## Summary Results\n\n")
        f.write(summary_df.to_markdown(index=False))
        f.write("\n\n")
        
        # Add per-model details
        for model_name, model_results in all_results.items():
            f.write(f"## {model_name}\n\n")
            for eval_type, eval_data in model_results.items():
                metrics = eval_data["metrics"]
                f.write(f"### {eval_type.replace('_', ' ').title()}\n")
                f.write(f"- Overall Accuracy: {metrics['overall_accuracy']:.3f}\n")
                f.write(f"- Parse Success Rate: {metrics['parse_rate']:.3f}\n")
                
                # Top 5 best/worst organs
                organ_accs = [(o, m["accuracy"]) for o, m in metrics["organ_metrics"].items()]
                organ_accs.sort(key=lambda x: x[1], reverse=True)
                
                f.write(f"- Best organs: {', '.join([f'{o} ({a:.2f})' for o, a in organ_accs[:3]])}\n")
                f.write(f"- Worst organs: {', '.join([f'{o} ({a:.2f})' for o, a in organ_accs[-3:]])}\n\n")
    
    print(f"✅ Evaluation complete! Results saved to {results_dir}")
    
    # Print summary
    print("\n📊 Summary:")
    print(summary_df.to_string(index=False))
    
    return all_results, results_dir

# Cell 9: Visualization
def create_comparison_plot(results_dir):
    """Create a simple comparison plot."""
    import matplotlib.pyplot as plt
    
    # Load summary
    summary_df = pd.read_csv(results_dir / "summary.csv")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(summary_df))
    width = 0.25
    
    ax.bar(x - width, summary_df['zero_shot_acc'], width, label='Zero-shot')
    ax.bar(x, summary_df['few_shot_acc'], width, label='Few-shot')
    ax.bar(x + width, summary_df['few_shot_hn_acc'], width, label='Few-shot (Hard Neg)')
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df['model'])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'comparison.png', dpi=150)
    print(f"✓ Saved plot to {results_dir / 'comparison.png'}")
    
    return fig

# Cell 10: Main execution
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Starting Comprehensive Evaluation")
    print("="*60)
    
    # Run evaluation
    results, results_dir = run_full_evaluation()
    
    # Create visualization
    try:
        create_comparison_plot(results_dir)
    except ImportError:
        print("⚠️ Matplotlib not available, skipping visualization")
    
    print("\n✨ All done!")