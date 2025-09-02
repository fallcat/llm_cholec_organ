"""Main evaluator class for pointing tasks."""

import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

from tqdm import tqdm
import torch

from ..datasets.cholecseg8k import CholecSeg8kAdapter, ID2LABEL, LABEL_IDS
# Comment out missing model imports
from ..models import OpenAIAdapter, AnthropicAdapter, GoogleAdapter
from .pointing import run_pointing_on_canvas
from ..prompts.builders import (
    build_pointing_system_prompt,
    build_pointing_user_prompt,
)


class PointingEvaluator:
    """Main evaluator for pointing tasks."""
    
    def __init__(
        self,
        models: List[str],
        dataset,  # HuggingFace dataset object
        dataset_adapter: Optional[CholecSeg8kAdapter] = None,
        canvas_width: int = 768,
        canvas_height: int = 768,
        output_dir: Optional[Path] = None,
        use_cache: bool = True,
    ):
        """Initialize evaluator.
        
        Args:
            models: List of model names to evaluate
            dataset: HuggingFace dataset object
            dataset_adapter: Dataset adapter instance
            canvas_width: Canvas width for pointing
            canvas_height: Canvas height for pointing
            output_dir: Output directory for results
            use_cache: Whether to use cache for model responses
        """
        self.models = models
        self.dataset = dataset
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.use_cache = use_cache
        
        # Setup output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path.home() / "results" / f"pointing_{timestamp}"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup adapter
        if dataset_adapter is None:
            self.adapter = CholecSeg8kAdapter()
        else:
            self.adapter = dataset_adapter
        
        # Organ names
        self.organ_names = [ID2LABEL[i] for i in LABEL_IDS]
        
        print(f"Initialized evaluator:")
        print(f"  Models: {self.models}")
        print(f"  Organs: {len(self.organ_names)}")
        print(f"  Canvas: {canvas_width}x{canvas_height}")
        print(f"  Output: {self.output_dir}")
    
    def load_model(self, model_name: str, use_cache: bool = True):
        """Load a model adapter.
        
        Args:
            model_name: Model name
            use_cache: Whether to use cache (default: True)
            
        Returns:
            Model adapter instance
        """
        if "gpt" in model_name.lower():
            return OpenAIAdapter(
                model_name=model_name,
                max_tokens=100,
                use_cache=use_cache,
            )
        elif "claude" in model_name.lower():
            return AnthropicAdapter(
                model_name=model_name, 
                max_tokens=100,
                use_cache=use_cache,
            )
        elif "gemini" in model_name.lower():
            return GoogleAdapter(
                model_name=model_name,
                max_tokens=100,
                use_cache=use_cache,
            )
        elif "llava" in model_name.lower():
            # Import LLaVA model from vllm module
            from ..models.vllm import LLaVAModel
            return LLaVAModel(
                model_name=model_name,
                use_vllm=True,  # Use vLLM for faster inference
                max_tokens=100,
                temperature=0.0,
                use_cache=use_cache,
                verbose=True  # Enable verbose for debugging
            )
        elif "qwen" in model_name.lower():
            # Import Qwen-VL model from vllm module
            from ..models.vllm import QwenVLModel
            # Note: Qwen vLLM integration has issues with image tokens
            # Using transformers backend for now
            return QwenVLModel(
                model_name=model_name,
                use_vllm=False,  # Use transformers due to vLLM image token issues
                max_tokens=100,
                temperature=0.0,
                use_cache=use_cache,
                verbose=True  # Enable verbose for debugging
            )
        elif "pixtral" in model_name.lower():
            # Import Pixtral model from vllm module
            from ..models.vllm import PixtralModel
            # PixtralModel always uses vLLM, no need for use_vllm parameter
            return PixtralModel(
                model_name=model_name,
                max_tokens=100,
                temperature=0.0,
                use_cache=use_cache,
                verbose=True  # Enable verbose for debugging
            )
        elif "deepseek" in model_name.lower():
            # Import DeepSeek-VL2 model from vllm module
            from ..models.vllm import DeepSeekVL2Model
            return DeepSeekVL2Model(
                model_name=model_name,
                max_tokens=100,
                temperature=0.0,
                use_cache=use_cache,
                verbose=True  # Enable verbose for debugging
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def prepare_few_shot_examples(
        self,
        fewshot_plan: Dict,
        dataset_split: str = "train",
    ) -> Dict[str, List[Tuple[torch.Tensor, Dict]]]:
        """Prepare few-shot examples for each organ.
        
        Args:
            fewshot_plan: Few-shot plan dictionary
            dataset_split: Dataset split to use
            
        Returns:
            Dictionary mapping organ names to list of (image, response) tuples
        """
        few_shot_examples = {}
        
        # Handle different plan formats
        if 'plan' in fewshot_plan:
            # New format: {'plan': {'1': {...}, '2': {...}}}
            actual_plan = fewshot_plan['plan']
        else:
            # Old format: {'organ_name': {...}}
            actual_plan = fewshot_plan
        
        for organ_name in self.organ_names:
            examples = []
            
            # Try to find the organ plan by ID or name
            organ_id = str(self.adapter.label2id.get(organ_name, -1))
            
            # First try by ID (new format)
            if organ_id in actual_plan:
                organ_data = actual_plan[organ_id]
                # Extract examples from new format
                # positives is list of dicts, negatives are list of ints
                positives = organ_data.get('positives', [])
                pos_indices = [item['idx'] if isinstance(item, dict) else item for item in positives]
                
                negatives_easy = organ_data.get('negatives_easy', [])
                neg_easy_indices = [item['idx'] if isinstance(item, dict) else item for item in negatives_easy]
                
                negatives_hard = organ_data.get('negatives_hard', [])
                neg_hard_indices = [item['idx'] if isinstance(item, dict) else item for item in negatives_hard]
                
                organ_plan = {
                    'positive': pos_indices,
                    'negative_easy': neg_easy_indices,
                    'negative_hard': neg_hard_indices
                }
            # Fallback to name (old format)
            elif organ_name in actual_plan:
                organ_plan = actual_plan[organ_name]
            else:
                organ_plan = {}
            
            # Add positive examples
            for idx in organ_plan.get("positive", []):
                example = self.dataset[dataset_split][idx]
                img_t, lab_t = self.adapter.example_to_tensors(example)
                
                # Get ground truth presence
                presence = self.adapter.labels_to_presence_vector(lab_t)
                organ_idx = LABEL_IDS.index(self.adapter.label2id[organ_name])
                is_present = presence[organ_idx].item()
                
                if is_present:
                    # For positive examples, we'd ideally compute the actual point
                    # For now, use a placeholder in the center
                    response = {
                        "name": organ_name,
                        "present": 1,
                        "point_canvas": [self.canvas_width // 2, self.canvas_height // 2]
                    }
                    examples.append((img_t, response))
            
            # Add negative examples (easy)
            for idx in organ_plan.get("negative_easy", []):
                example = self.dataset[dataset_split][idx]
                img_t, lab_t = self.adapter.example_to_tensors(example)
                
                response = {
                    "name": organ_name,
                    "present": 0,
                    "point_canvas": None
                }
                examples.append((img_t, response))
            
            # Add hard negative examples
            for idx in organ_plan.get("negative_hard", []):
                example = self.dataset[dataset_split][idx]
                img_t, lab_t = self.adapter.example_to_tensors(example)
                
                response = {
                    "name": organ_name,
                    "present": 0,
                    "point_canvas": None
                }
                examples.append((img_t, response))
            
            if examples:
                few_shot_examples[organ_name] = examples
        
        return few_shot_examples
    
    def run_zero_shot(
        self,
        model_name: str,
        test_indices: List[int],
        split: str = "train",
    ) -> Dict:
        """Run zero-shot pointing evaluation.
        
        Args:
            model_name: Model name
            test_indices: Indices to evaluate
            split: Dataset split
            
        Returns:
            Results dictionary
        """
        print(f"\n🔄 Running zero-shot with {model_name}...")
        
        model = self.load_model(model_name, use_cache=self.use_cache)
        results = []
        
        for idx in tqdm(test_indices, desc=f"{model_name} zero-shot"):
            example = self.dataset[split][idx]
            img_t, lab_t = self.adapter.example_to_tensors(example)
            
            # Run pointing for each organ
            organ_results = []
            for organ_name in self.organ_names:
                result = run_pointing_on_canvas(
                    model=model,
                    img_t=img_t,
                    lab_t=lab_t,
                    organ_name=organ_name,
                    canvas_width=self.canvas_width,
                    canvas_height=self.canvas_height,
                    system_prompt_builder=build_pointing_system_prompt,
                    user_prompt_builder=build_pointing_user_prompt,
                    few_shot_examples=None,
                )
                organ_results.append(result)
            
            # Get ground truth
            y_true = self.adapter.labels_to_presence_vector(lab_t).numpy()
            y_pred = np.array([r["present"] for r in organ_results])
            
            results.append({
                "sample_idx": idx,
                "organ_results": organ_results,
                "y_pred": y_pred,
                "y_true": y_true,
            })
        
        # Calculate metrics
        all_y_pred = np.stack([r["y_pred"] for r in results])
        all_y_true = np.stack([r["y_true"] for r in results])
        
        metrics = self.calculate_metrics(all_y_pred, all_y_true)
        
        return {
            "results": results,
            "metrics": metrics,
        }
    
    def run_few_shot(
        self,
        model_name: str,
        test_indices: List[int],
        fewshot_plan: Dict,
        plan_name: str = "standard",
        split: str = "train",
    ) -> Dict:
        """Run few-shot pointing evaluation.
        
        Args:
            model_name: Model name
            test_indices: Indices to evaluate
            fewshot_plan: Few-shot plan dictionary
            plan_name: Name for this plan (for logging)
            split: Dataset split
            
        Returns:
            Results dictionary
        """
        print(f"\n🔄 Running few-shot ({plan_name}) with {model_name}...")
        
        model = self.load_model(model_name, use_cache=self.use_cache)
        few_shot_examples = self.prepare_few_shot_examples(fewshot_plan, split)
        
        results = []
        
        for idx in tqdm(test_indices, desc=f"{model_name} {plan_name}"):
            example = self.dataset[split][idx]
            img_t, lab_t = self.adapter.example_to_tensors(example)
            
            # Run pointing for each organ with few-shot examples
            organ_results = []
            for organ_name in self.organ_names:
                organ_examples = few_shot_examples.get(organ_name, [])
                
                # Pass few-shot examples if available
                result = run_pointing_on_canvas(
                    model=model,
                    img_t=img_t,
                    lab_t=lab_t,
                    organ_name=organ_name,
                    canvas_width=self.canvas_width,
                    canvas_height=self.canvas_height,
                    system_prompt_builder=build_pointing_system_prompt,
                    user_prompt_builder=build_pointing_user_prompt,
                    few_shot_examples=organ_examples if organ_examples else None,
                )
                organ_results.append(result)
            
            # Get ground truth
            y_true = self.adapter.labels_to_presence_vector(lab_t).numpy()
            y_pred = np.array([r["present"] for r in organ_results])
            
            results.append({
                "sample_idx": idx,
                "organ_results": organ_results,
                "y_pred": y_pred,
                "y_true": y_true,
            })
        
        # Calculate metrics
        all_y_pred = np.stack([r["y_pred"] for r in results])
        all_y_true = np.stack([r["y_true"] for r in results])
        
        metrics = self.calculate_metrics(all_y_pred, all_y_true)
        
        return {
            "results": results,
            "metrics": metrics,
        }
    
    def calculate_metrics(self, y_pred: np.ndarray, y_true: np.ndarray) -> Dict:
        """Calculate evaluation metrics.
        
        Args:
            y_pred: Predictions [N, K]
            y_true: Ground truth [N, K]
            
        Returns:
            Metrics dictionary
        """
        organ_metrics = {}
        
        for i, organ_name in enumerate(self.organ_names):
            y_true_organ = y_true[:, i]
            y_pred_organ = y_pred[:, i]
            
            tp = ((y_true_organ == 1) & (y_pred_organ == 1)).sum()
            fp = ((y_true_organ == 0) & (y_pred_organ == 1)).sum()
            fn = ((y_true_organ == 1) & (y_pred_organ == 0)).sum()
            tn = ((y_true_organ == 0) & (y_pred_organ == 0)).sum()
            
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            organ_metrics[organ_name] = {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn),
            }
        
        # Overall metrics
        overall_accuracy = (y_pred == y_true).mean()
        
        # Average metrics
        avg_precision = np.mean([m["precision"] for m in organ_metrics.values()])
        avg_recall = np.mean([m["recall"] for m in organ_metrics.values()])
        avg_f1 = np.mean([m["f1"] for m in organ_metrics.values()])
        
        return {
            "overall_accuracy": float(overall_accuracy),
            "avg_precision": float(avg_precision),
            "avg_recall": float(avg_recall),
            "avg_f1": float(avg_f1),
            "organ_metrics": organ_metrics,
        }
    
    def run_full_evaluation(
        self,
        test_indices: List[int],
        fewshot_plans: Dict[str, Dict],
        skip_zero_shot: bool = False,
    ) -> Dict:
        """Run complete evaluation pipeline.
        
        Args:
            test_indices: List of test sample indices
            fewshot_plans: Dictionary mapping plan names to plan data
            skip_zero_shot: Whether to skip zero-shot evaluation
            
        Returns:
            All results dictionary
        """
        print(f"\n📊 Evaluating {len(test_indices)} test samples")
        
        all_results = {}
        
        for model_name in self.models:
            print(f"\n{'='*60}")
            print(f"Evaluating: {model_name}")
            print(f"{'='*60}")
            
            model_results = {}
            
            # Zero-shot evaluation (if not skipped)
            if not skip_zero_shot:
                zero_shot_results = self.run_zero_shot(model_name, test_indices)
                model_results["zero_shot"] = zero_shot_results
            
            # Few-shot evaluations
            for plan_name, plan_data in fewshot_plans.items():
                few_shot_results = self.run_few_shot(
                    model_name, test_indices, plan_data, plan_name
                )
                model_results[f"few_shot_{plan_name}"] = few_shot_results
            
            all_results[model_name] = model_results
            
            # Print summary
            print(f"\n📊 {model_name} Results:")
            for eval_type, results in model_results.items():
                metrics = results["metrics"]
                print(f"  {eval_type}:")
                print(f"    Overall Accuracy: {metrics['overall_accuracy']:.3f}")
                print(f"    Avg F1: {metrics['avg_f1']:.3f}")
        
        # Save results
        self.save_results(all_results)
        
        return all_results
    
    def save_results(self, all_results: Dict):
        """Save evaluation results.
        
        Args:
            all_results: Complete results dictionary
        """
        print(f"\n💾 Saving results to {self.output_dir}")
        
        # Save raw results
        with open(self.output_dir / "raw_results.pkl", "wb") as f:
            pickle.dump(all_results, f)
        
        # Create summary DataFrame if pandas is available
        if pd is not None:
            summary_data = []
            for model_name, model_results in all_results.items():
                for eval_type, results in model_results.items():
                    metrics = results["metrics"]
                    summary_data.append({
                        "model": model_name,
                        "eval_type": eval_type,
                        "overall_accuracy": metrics["overall_accuracy"],
                        "avg_precision": metrics["avg_precision"],
                        "avg_recall": metrics["avg_recall"],
                        "avg_f1": metrics["avg_f1"],
                    })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(self.output_dir / "summary.csv", index=False)
        
        # Create detailed report
        with open(self.output_dir / "report.md", "w") as f:
            f.write("# Pointing Evaluation Report\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            
            f.write("## Configuration\n")
            f.write(f"- Models: {self.models}\n")
            f.write(f"- Canvas: {self.canvas_width}x{self.canvas_height}\n")
            f.write(f"- Organs evaluated: {len(self.organ_names)}\n\n")
            
            if pd is not None and 'summary_df' in locals():
                f.write("## Summary\n\n")
                f.write(summary_df.to_markdown(index=False))
                f.write("\n\n")
            
            # Per-model detailed results
            for model_name, model_results in all_results.items():
                f.write(f"## {model_name}\n\n")
                
                for eval_type, results in model_results.items():
                    metrics = results["metrics"]
                    f.write(f"### {eval_type}\n")
                    f.write(f"- Overall Accuracy: {metrics['overall_accuracy']:.3f}\n")
                    f.write(f"- Avg Precision: {metrics['avg_precision']:.3f}\n")
                    f.write(f"- Avg Recall: {metrics['avg_recall']:.3f}\n")
                    f.write(f"- Avg F1: {metrics['avg_f1']:.3f}\n\n")
                    
                    # Top performing organs
                    organ_perfs = [
                        (o, m["f1"]) 
                        for o, m in metrics["organ_metrics"].items()
                    ]
                    organ_perfs.sort(key=lambda x: x[1], reverse=True)
                    
                    f.write("Top 3 organs (F1 score):\n")
                    for organ, f1 in organ_perfs[:3]:
                        f.write(f"- {organ}: {f1:.3f}\n")
                    
                    f.write("\nBottom 3 organs (F1 score):\n")
                    for organ, f1 in organ_perfs[-3:]:
                        f.write(f"- {organ}: {f1:.3f}\n")
                    f.write("\n")
        
        print(f"✅ Results saved successfully!")