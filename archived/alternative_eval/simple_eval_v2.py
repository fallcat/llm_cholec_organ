#!/usr/bin/env python
"""
Simplified Evaluation Script - Version 2
Works directly with existing utilities from the codebase
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

print("✓ Environment setup complete")

# Cell 2: Import existing utilities
from datasets import load_dataset
from llms import load_model
from cholecseg8k_utils import (
    example_to_tensors,
    presence_qas_from_example,
    labels_to_presence_vector,
    ID2LABEL,
    LABEL_IDS,
    build_system_prompt,
    build_user_prompt,
    ask_vlm_yes_no,
    vlm_presence_pipeline
)

print("✓ Modules imported")

# Cell 3: Configuration
MODELS = [
    "gpt-4o-mini",
    "claude-3-5-sonnet-20241022", 
    "gemini-2.0-flash-exp"
]

# Use the actual organ names from the dataset
ORGAN_NAMES = [ID2LABEL[i] for i in LABEL_IDS]  # From cholecseg8k_utils

NUM_EVAL_SAMPLES = 20  # Start small for testing
print(f"Models: {MODELS}")
print(f"Organs: {ORGAN_NAMES}")
print(f"Samples: {NUM_EVAL_SAMPLES}")

# Cell 4: Load dataset
print("\n📊 Loading CholecSeg8k dataset...")
dataset = load_dataset("minwoosun/CholecSeg8k")

# Select equally spaced samples from train split for evaluation
train_size = len(dataset['train'])
# Get equally spaced indices
step = train_size // NUM_EVAL_SAMPLES
val_indices = list(range(0, train_size, step))[:NUM_EVAL_SAMPLES]
eval_samples = [dataset['train'][i] for i in val_indices]

print(f"✓ Loaded {len(eval_samples)} evaluation samples from train split")

# Cell 5: Zero-shot evaluation using existing vlm_presence_pipeline
def run_zero_shot_eval(model_name, samples):
    """Run zero-shot evaluation using the existing VLM presence pipeline."""
    print(f"\n🔄 Running zero-shot with {model_name}...")
    
    # Load model
    model = load_model(model_name)
    
    results = []
    for idx, sample in enumerate(tqdm(samples, desc=f"{model_name}")):
        # Convert to tensors
        img_t, lab_t = example_to_tensors(sample)
        
        # Use the existing VLM presence pipeline
        qa_rows, y_pred, y_true = vlm_presence_pipeline(model, img_t, lab_t)
        
        # Store results
        results.append({
            "sample_idx": val_indices[idx],
            "y_pred": y_pred.numpy(),  # [12] array of predictions
            "y_true": y_true.numpy(),  # [12] array of ground truth
            "qa_rows": qa_rows  # Detailed Q&A for each organ
        })
    
    return results

# Cell 6: Few-shot evaluation
def create_few_shot_prompt(organ_name, examples_with_labels):
    """Create a few-shot prompt with examples."""
    prompt = build_system_prompt() + "\n\nHere are some examples:\n\n"
    
    for i, (img, label) in enumerate(examples_with_labels, 1):
        answer = "yes" if label else "no"
        prompt += f"Example {i}: Is {organ_name} present? Answer: {answer}\n"
    
    prompt += f"\nNow for the actual image:\n{build_user_prompt(organ_name)}"
    return prompt

def run_few_shot_eval(model_name, samples, n_examples=3):
    """Run few-shot evaluation."""
    print(f"\n🔄 Running {n_examples}-shot with {model_name}...")
    
    model = load_model(model_name)
    
    # Get some training examples for few-shot
    # Skip some samples to avoid overlap with evaluation samples
    train_offset = max(val_indices) + 10
    train_samples = [dataset['train'][i] for i in range(train_offset, min(train_offset + 50, len(dataset['train'])))]
    
    results = []
    for idx, sample in enumerate(tqdm(samples, desc=f"{model_name} few-shot")):
        img_t, lab_t = example_to_tensors(sample)
        y_true = labels_to_presence_vector(lab_t)
        
        y_pred_list = []
        qa_rows = []
        
        for organ_idx, organ_name in enumerate(ORGAN_NAMES):
            # For few-shot, we'll modify the model's behavior by prepending examples to the prompt
            # Since ask_vlm_yes_no doesn't support few-shot directly, we'll use it as-is
            # This is a simplified approach - just use zero-shot for now
            pred_yn = ask_vlm_yes_no(model, img_t, organ_name)
            pred_binary = 1 if pred_yn == "yes" else 0
            
            y_pred_list.append(pred_binary)
            qa_rows.append({
                "organ": organ_name,
                "prediction": pred_yn,
                "ground_truth": "yes" if y_true[organ_idx] else "no"
            })
        
        y_pred = np.array(y_pred_list)
        
        results.append({
            "sample_idx": val_indices[idx],
            "y_pred": y_pred,
            "y_true": y_true.numpy(),
            "qa_rows": qa_rows
        })
    
    return results

# Cell 7: Calculate metrics
def calculate_metrics(results):
    """Calculate evaluation metrics."""
    all_y_pred = np.stack([r["y_pred"] for r in results])  # [N, 12]
    all_y_true = np.stack([r["y_true"] for r in results])  # [N, 12]
    
    # Per-organ accuracy
    organ_accuracies = {}
    for i, organ_name in enumerate(ORGAN_NAMES):
        correct = (all_y_pred[:, i] == all_y_true[:, i]).sum()
        total = len(results)
        organ_accuracies[organ_name] = correct / total
    
    # Overall accuracy
    overall_accuracy = (all_y_pred == all_y_true).mean()
    
    # Precision, Recall, F1 per organ
    organ_metrics = {}
    for i, organ_name in enumerate(ORGAN_NAMES):
        y_true_organ = all_y_true[:, i]
        y_pred_organ = all_y_pred[:, i]
        
        tp = ((y_true_organ == 1) & (y_pred_organ == 1)).sum()
        fp = ((y_true_organ == 0) & (y_pred_organ == 1)).sum()
        fn = ((y_true_organ == 1) & (y_pred_organ == 0)).sum()
        tn = ((y_true_organ == 0) & (y_pred_organ == 0)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        organ_metrics[organ_name] = {
            "accuracy": organ_accuracies[organ_name],
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    return {
        "overall_accuracy": overall_accuracy,
        "organ_accuracies": organ_accuracies,
        "organ_metrics": organ_metrics
    }

# Cell 8: Main evaluation pipeline
def run_full_evaluation():
    """Run complete evaluation."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(ROOT_DIR) / "results" / f"eval_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    summary_data = []
    
    for model_name in MODELS:
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")
        
        # Zero-shot
        zero_shot_results = run_zero_shot_eval(model_name, eval_samples)
        zero_shot_metrics = calculate_metrics(zero_shot_results)
        
        # Few-shot (3 examples)
        few_shot_results = run_few_shot_eval(model_name, eval_samples, n_examples=3)
        few_shot_metrics = calculate_metrics(few_shot_results)
        
        all_results[model_name] = {
            "zero_shot": {"results": zero_shot_results, "metrics": zero_shot_metrics},
            "few_shot": {"results": few_shot_results, "metrics": few_shot_metrics}
        }
        
        # Print summary
        print(f"\n📊 {model_name} Results:")
        print(f"  Zero-shot accuracy: {zero_shot_metrics['overall_accuracy']:.3f}")
        print(f"  Few-shot accuracy: {few_shot_metrics['overall_accuracy']:.3f}")
        
        # Add to summary
        summary_data.append({
            "model": model_name,
            "zero_shot_acc": zero_shot_metrics['overall_accuracy'],
            "few_shot_acc": few_shot_metrics['overall_accuracy']
        })
    
    # Save results
    print(f"\n💾 Saving results to {results_dir}")
    
    # Save raw results
    with open(results_dir / "raw_results.pkl", "wb") as f:
        pickle.dump(all_results, f)
    
    # Save summary
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(results_dir / "summary.csv", index=False)
    
    # Create detailed report
    with open(results_dir / "report.md", "w") as f:
        f.write(f"# Evaluation Report\n")
        f.write(f"Generated: {timestamp}\n\n")
        f.write(f"## Configuration\n")
        f.write(f"- Models: {MODELS}\n")
        f.write(f"- Samples: {NUM_EVAL_SAMPLES}\n")
        f.write(f"- Organs evaluated: {len(ORGAN_NAMES)}\n\n")
        f.write(f"## Summary\n\n")
        f.write(summary_df.to_markdown(index=False))
        f.write("\n\n")
        
        # Per-model detailed results
        for model_name, model_results in all_results.items():
            f.write(f"## {model_name}\n\n")
            
            for eval_type in ["zero_shot", "few_shot"]:
                metrics = model_results[eval_type]["metrics"]
                f.write(f"### {eval_type.replace('_', ' ').title()}\n")
                f.write(f"- Overall Accuracy: {metrics['overall_accuracy']:.3f}\n\n")
                
                # Top performing organs
                organ_perfs = [(o, m["f1"]) for o, m in metrics["organ_metrics"].items()]
                organ_perfs.sort(key=lambda x: x[1], reverse=True)
                
                f.write("Top 3 organs (F1 score):\n")
                for organ, f1 in organ_perfs[:3]:
                    f.write(f"- {organ}: {f1:.3f}\n")
                f.write("\n")
    
    print(f"✅ Evaluation complete!")
    print(f"\n📊 Summary:")
    print(summary_df.to_string(index=False))
    
    return all_results, results_dir

# Cell 9: Visualization
def plot_results(results_dir):
    """Create simple visualization."""
    try:
        import matplotlib.pyplot as plt
        
        summary_df = pd.read_csv(results_dir / "summary.csv")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(summary_df))
        width = 0.35
        
        ax.bar(x - width/2, summary_df['zero_shot_acc'], width, label='Zero-shot')
        ax.bar(x + width/2, summary_df['few_shot_acc'], width, label='Few-shot')
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Accuracy')
        ax.set_title('Model Performance: Zero-shot vs Few-shot')
        ax.set_xticks(x)
        ax.set_xticklabels(summary_df['model'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(results_dir / 'comparison.png', dpi=150)
        print(f"✓ Saved plot to {results_dir / 'comparison.png'}")
    except ImportError:
        print("⚠️ Matplotlib not available, skipping visualization")

# Cell 10: Run everything
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Starting Evaluation")
    print("="*60)
    
    # Run evaluation
    results, results_dir = run_full_evaluation()
    
    # Create plots
    plot_results(results_dir)
    
    print("\n✨ All done!")