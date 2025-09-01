"""Enhanced evaluator with comprehensive metrics matching the notebook approach."""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import torch

from .evaluator import PointingEvaluator
from .pointing_metrics import (
    calculate_comprehensive_metrics,
    print_metrics_table,
    save_metrics_json,
    check_point_hit,
)
from ..datasets.cholecseg8k import CholecSeg8kAdapter, ID2LABEL, LABEL_IDS


class EnhancedPointingEvaluator(PointingEvaluator):
    """Enhanced evaluator with comprehensive metrics."""
    
    def __init__(self, *args, **kwargs):
        """Initialize enhanced evaluator."""
        super().__init__(*args, **kwargs)
        self.detailed_results = {}  # Store detailed results for analysis
        
    def run_zero_shot_enhanced(
        self,
        model_name: str,
        test_indices: List[int],
        split: str = "train",
        save_per_sample: bool = True,
    ) -> Dict:
        """Run zero-shot evaluation with enhanced metrics.
        
        Args:
            model_name: Model name
            test_indices: Indices to evaluate
            split: Dataset split
            save_per_sample: Whether to save per-sample results
            
        Returns:
            Enhanced results dictionary
        """
        print(f"\n🔄 Running zero-shot with {model_name} (enhanced metrics)...")
        
        model = self.load_model(model_name, use_cache=self.use_cache)
        
        # Create output directory for this evaluation
        eval_dir = self.output_dir / "zero_shot" / model_name / f"cholecseg8k_pointing"
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        records = []
        
        for idx in tqdm(test_indices, desc=f"{model_name} zero-shot"):
            example = self.dataset[split][idx]
            img_t, lab_t = self.adapter.example_to_tensors(example)
            
            # Get ground truth
            y_true = self.adapter.labels_to_presence_vector(lab_t).numpy()
            
            # Run pointing for each organ
            organ_results = []
            y_pred = []
            hits = []
            
            for i, organ_name in enumerate(self.organ_names):
                from .pointing import run_pointing_on_canvas
                from ..prompts.builders import (
                    build_pointing_system_prompt,
                    build_pointing_user_prompt,
                )
                
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
                y_pred.append(result["present"])
                
                # Check if point hits the organ
                if result["present"] == 1 and result.get("point_canvas") is not None:
                    # Get organ mask
                    organ_id = LABEL_IDS[i]
                    organ_mask = (lab_t.numpy() == organ_id).astype(np.uint8)
                    
                    # Check hit
                    hit = check_point_hit(
                        result["point_canvas"],
                        organ_mask,
                        self.canvas_width,
                        self.canvas_height,
                    )
                    hits.append(1 if hit else 0)
                else:
                    hits.append(0)
            
            # Create record for this sample
            record = {
                "sample_idx": idx,
                "y_true": y_true.tolist(),
                "y_pred": y_pred,
                "hits": hits,
                "rows": organ_results,
                "prompt": {"name": "zero_shot"},
            }
            records.append(record)
            
            # Save per-sample result
            if save_per_sample:
                sample_file = eval_dir / f"{split}_{idx:05d}.json"
                with open(sample_file, "w") as f:
                    json.dump(record, f, indent=2)
        
        # Calculate comprehensive metrics
        rows, totals, n_examples = calculate_comprehensive_metrics(records)
        
        # Print metrics table
        table_output = print_metrics_table(
            rows, totals, n_examples,
            model_name, "zero_shot", split
        )
        print(table_output)
        
        # Save metrics summary
        metrics_file = eval_dir / f"metrics_summary_{split}.json"
        save_metrics_json(
            rows, totals, n_examples,
            model_name, "zero_shot", split,
            metrics_file
        )
        
        return {
            "records": records,
            "metrics_rows": rows,
            "metrics_totals": totals,
            "n_examples": n_examples,
            "table_output": table_output,
        }
    
    def run_few_shot_enhanced(
        self,
        model_name: str,
        test_indices: List[int],
        fewshot_plan: Dict,
        plan_name: str = "standard",
        split: str = "train",
        save_per_sample: bool = True,
    ) -> Dict:
        """Run few-shot evaluation with enhanced metrics.
        
        Args:
            model_name: Model name
            test_indices: Indices to evaluate
            fewshot_plan: Few-shot plan dictionary
            plan_name: Name for this plan
            split: Dataset split
            save_per_sample: Whether to save per-sample results
            
        Returns:
            Enhanced results dictionary
        """
        print(f"\n🔄 Running few-shot ({plan_name}) with {model_name} (enhanced metrics)...")
        
        model = self.load_model(model_name, use_cache=self.use_cache)
        few_shot_examples = self.prepare_few_shot_examples(fewshot_plan, split)
        
        # Create output directory
        prompt_name = f"fewshot_{plan_name}"
        eval_dir = self.output_dir / prompt_name / model_name / f"cholecseg8k_pointing"
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        records = []
        
        for idx in tqdm(test_indices, desc=f"{model_name} {plan_name}"):
            example = self.dataset[split][idx]
            img_t, lab_t = self.adapter.example_to_tensors(example)
            
            # Get ground truth
            y_true = self.adapter.labels_to_presence_vector(lab_t).numpy()
            
            # Run pointing for each organ with few-shot
            organ_results = []
            y_pred = []
            hits = []
            
            for i, organ_name in enumerate(self.organ_names):
                from .pointing import run_pointing_on_canvas
                from ..prompts.builders import (
                    build_pointing_system_prompt,
                    build_pointing_user_prompt,
                )
                
                organ_examples = few_shot_examples.get(organ_name, [])
                
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
                y_pred.append(result["present"])
                
                # Check if point hits the organ
                if result["present"] == 1 and result.get("point_canvas") is not None:
                    # Get organ mask
                    organ_id = LABEL_IDS[i]
                    organ_mask = (lab_t.numpy() == organ_id).astype(np.uint8)
                    
                    # Check hit
                    hit = check_point_hit(
                        result["point_canvas"],
                        organ_mask,
                        self.canvas_width,
                        self.canvas_height,
                    )
                    hits.append(1 if hit else 0)
                else:
                    hits.append(0)
            
            # Create record
            record = {
                "sample_idx": idx,
                "y_true": y_true.tolist(),
                "y_pred": y_pred,
                "hits": hits,
                "rows": organ_results,
                "prompt": {"name": prompt_name},
            }
            records.append(record)
            
            # Save per-sample result
            if save_per_sample:
                sample_file = eval_dir / f"{split}_{idx:05d}.json"
                with open(sample_file, "w") as f:
                    json.dump(record, f, indent=2)
        
        # Calculate comprehensive metrics
        rows, totals, n_examples = calculate_comprehensive_metrics(records)
        
        # Print metrics table
        table_output = print_metrics_table(
            rows, totals, n_examples,
            model_name, prompt_name, split
        )
        print(table_output)
        
        # Save metrics summary
        metrics_file = eval_dir / f"metrics_summary_{split}.json"
        save_metrics_json(
            rows, totals, n_examples,
            model_name, prompt_name, split,
            metrics_file
        )
        
        return {
            "records": records,
            "metrics_rows": rows,
            "metrics_totals": totals,
            "n_examples": n_examples,
            "table_output": table_output,
        }
    
    def run_full_evaluation_enhanced(
        self,
        test_indices: List[int],
        fewshot_plans: Dict[str, Dict],
        split: str = "train",
    ) -> Dict:
        """Run complete evaluation with enhanced metrics.
        
        Args:
            test_indices: List of test sample indices
            fewshot_plans: Dictionary mapping plan names to plan data
            split: Dataset split
            
        Returns:
            All results with enhanced metrics
        """
        print(f"\n📊 Evaluating {len(test_indices)} test samples with enhanced metrics")
        
        all_results = {}
        
        for model_name in self.models:
            print(f"\n{'='*60}")
            print(f"Evaluating: {model_name}")
            print(f"{'='*60}")
            
            model_results = {}
            
            # Zero-shot evaluation
            zero_shot_results = self.run_zero_shot_enhanced(
                model_name, test_indices, split
            )
            model_results["zero_shot"] = zero_shot_results
            
            # Few-shot evaluations
            for plan_name, plan_data in fewshot_plans.items():
                few_shot_results = self.run_few_shot_enhanced(
                    model_name, test_indices, plan_data, plan_name, split
                )
                model_results[f"few_shot_{plan_name}"] = few_shot_results
            
            all_results[model_name] = model_results
        
        # Save comprehensive comparison
        self.save_comparison_table(all_results)
        
        return all_results
    
    def save_comparison_table(self, all_results: Dict):
        """Save a comparison table across all models and prompts.
        
        Args:
            all_results: Complete results dictionary
        """
        comparison_file = self.output_dir / "metrics_comparison.txt"
        
        lines = []
        lines.append("="*80)
        lines.append("COMPREHENSIVE METRICS COMPARISON")
        lines.append("="*80)
        
        for model_name, model_results in all_results.items():
            lines.append(f"\n{model_name}:")
            
            for eval_type, results in model_results.items():
                if "table_output" in results:
                    lines.append(results["table_output"])
        
        with open(comparison_file, "w") as f:
            f.write("\n".join(lines))
        
        print(f"\n💾 Saved comparison table to {comparison_file}")