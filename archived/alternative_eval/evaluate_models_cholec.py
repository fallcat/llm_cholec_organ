#!/usr/bin/env python3
"""
Evaluate all models on CholecSeg8k dataset with proper metrics.
Compares API models (GPT, Claude, Gemini) with open-source VLMs.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import time
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from PIL import Image
import torch

from llms import load_model
from cholecseg8k_utils import (
    load_cholecseg8k_sample,
    get_balanced_sample_indices,
    compute_presence_matrix
)


# Organ classes in CholecSeg8k
ORGAN_CLASSES = [
    "Liver", "Gallbladder", "Hepatocystic Triangle", "Fat",
    "Grasper", "Connective Tissue", "Blood", "Cystic Artery",
    "Cystic Vein", "Cystic Pedicle", "Gallbladder Plate", "Abdominal Wall"
]

# Model configurations
MODELS = {
    # API Models
    "gpt-4o-mini": {"type": "api", "provider": "openai"},
    "gpt-4o": {"type": "api", "provider": "openai"},
    "claude-3-5-sonnet-20241022": {"type": "api", "provider": "anthropic"},
    "gemini-2.0-flash-exp": {"type": "api", "provider": "google"},
    
    # Open-source VLMs
    "llava-hf/llava-v1.6-mistral-7b-hf": {"type": "vlm", "provider": "llava"},
    "Qwen/Qwen2.5-VL-7B-Instruct": {"type": "vlm", "provider": "qwen"},
    "mistralai/Pixtral-12B-2409": {"type": "vlm", "provider": "pixtral"},
}


class ModelEvaluator:
    """Evaluate models on organ detection and pointing tasks."""
    
    def __init__(self, model_name: str, verbose: bool = True):
        self.model_name = model_name
        self.verbose = verbose
        self.model = None
        self.results = {
            "model": model_name,
            "detection": [],
            "pointing": [],
            "timing": [],
            "errors": []
        }
    
    def load_model(self):
        """Load the model with appropriate settings."""
        try:
            config = MODELS.get(self.model_name, {})
            
            # Model loading parameters
            params = {
                "use_cache": False,
                "max_tokens": 300,
                "temperature": 0.0,  # Deterministic
                "verbose": self.verbose
            }
            
            # VLM-specific settings
            if config.get("type") == "vlm":
                if torch.cuda.is_available():
                    params["use_vllm"] = True
                else:
                    params["use_vllm"] = False
                    if self.verbose:
                        print(f"⚠️  No CUDA, using CPU for {self.model_name}")
            
            if self.verbose:
                print(f"Loading {self.model_name}...")
            
            self.model = load_model(self.model_name, **params)
            
            if self.verbose:
                print(f"✅ Model loaded successfully")
            
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"❌ Failed to load {self.model_name}: {e}")
            self.results["errors"].append(f"Load error: {str(e)}")
            return False
    
    def create_detection_prompt(self) -> str:
        """Create standardized detection prompt."""
        return f"""Analyze this laparoscopic surgery image and identify which organs/structures are visible.

Check for these specific organs:
{', '.join(ORGAN_CLASSES)}

You MUST respond with ONLY a JSON object where keys are the exact organ names above and values are boolean:
{{"Liver": true, "Gallbladder": false, "Hepatocystic Triangle": false, ...}}

Include ALL organs listed above in your response."""
    
    def create_pointing_prompt(self, organ: str, width: int, height: int) -> str:
        """Create standardized pointing prompt."""
        return f"""Look at this surgical image ({width}x{height} pixels).
If you can see a {organ}, point to its approximate center.

Respond with ONLY this JSON format:
{{"present": true/false, "x": <x-coordinate or null>, "y": <y-coordinate or null>}}

- Set present=true and provide x,y if you see the {organ}
- Set present=false and x,y=null if you don't see it
- Coordinates must be within image bounds (0-{width} for x, 0-{height} for y)"""
    
    def evaluate_detection(self, image: Image.Image, true_labels: List[str]) -> Dict:
        """Evaluate organ detection task."""
        result = {
            "true_labels": true_labels,
            "predicted": {},
            "metrics": {}
        }
        
        try:
            # Generate prompt
            prompt = self.create_detection_prompt()
            
            # System prompt for better performance
            system_prompt = """You are an expert medical image analyst specializing in laparoscopic surgery.
Carefully analyze the image and identify organs accurately.
Always respond with valid JSON containing all requested organs."""
            
            # Get model response
            start_time = time.time()
            response = self.model((image, prompt), system_prompt=system_prompt)
            inference_time = time.time() - start_time
            
            result["response"] = response
            result["inference_time"] = inference_time
            
            # Parse JSON response
            try:
                # Extract JSON from response
                if "{" in response and "}" in response:
                    json_str = response[response.find("{"):response.rfind("}")+1]
                    predictions = json.loads(json_str)
                else:
                    predictions = json.loads(response)
                
                result["predicted"] = predictions
                
                # Calculate metrics
                true_positives = 0
                false_positives = 0
                true_negatives = 0
                false_negatives = 0
                
                for organ in ORGAN_CLASSES:
                    predicted = predictions.get(organ, False)
                    actual = organ in true_labels
                    
                    if predicted and actual:
                        true_positives += 1
                    elif predicted and not actual:
                        false_positives += 1
                    elif not predicted and actual:
                        false_negatives += 1
                    else:
                        true_negatives += 1
                
                # Compute metrics
                accuracy = (true_positives + true_negatives) / len(ORGAN_CLASSES)
                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                result["metrics"] = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "tp": true_positives,
                    "fp": false_positives,
                    "tn": true_negatives,
                    "fn": false_negatives
                }
                
                result["success"] = True
                
            except (json.JSONDecodeError, KeyError) as e:
                result["parse_error"] = str(e)
                result["success"] = False
                
        except Exception as e:
            result["error"] = str(e)
            result["success"] = False
        
        return result
    
    def evaluate_pointing(self, image: Image.Image, organ: str, mask: Optional[np.ndarray] = None) -> Dict:
        """Evaluate pointing task."""
        result = {
            "target_organ": organ,
            "metrics": {}
        }
        
        try:
            # Generate prompt
            prompt = self.create_pointing_prompt(organ, image.width, image.height)
            
            # System prompt
            system_prompt = """You are an expert at localizing organs in surgical images.
Provide accurate coordinates for organ centers when visible."""
            
            # Get model response
            start_time = time.time()
            response = self.model((image, prompt), system_prompt=system_prompt)
            inference_time = time.time() - start_time
            
            result["response"] = response
            result["inference_time"] = inference_time
            
            # Parse response
            try:
                if "{" in response and "}" in response:
                    json_str = response[response.find("{"):response.rfind("}")+1]
                    prediction = json.loads(json_str)
                else:
                    prediction = json.loads(response)
                
                result["predicted"] = prediction
                
                # Evaluate pointing accuracy if mask available
                if mask is not None and prediction.get("present"):
                    x = prediction.get("x")
                    y = prediction.get("y")
                    
                    if x is not None and y is not None:
                        # Check if point is within bounds
                        if 0 <= x < image.width and 0 <= y < image.height:
                            # Check if point hits the mask
                            hit = mask[int(y), int(x)] > 0
                            
                            # Calculate distance to nearest mask pixel
                            if not hit:
                                mask_points = np.argwhere(mask > 0)
                                if len(mask_points) > 0:
                                    distances = np.sqrt((mask_points[:, 0] - y)**2 + 
                                                      (mask_points[:, 1] - x)**2)
                                    min_distance = np.min(distances)
                                else:
                                    min_distance = float('inf')
                            else:
                                min_distance = 0
                            
                            result["metrics"] = {
                                "hit": hit,
                                "distance_to_mask": min_distance,
                                "x": x,
                                "y": y
                            }
                        else:
                            result["metrics"] = {
                                "hit": False,
                                "out_of_bounds": True,
                                "x": x,
                                "y": y
                            }
                
                result["success"] = True
                
            except (json.JSONDecodeError, KeyError) as e:
                result["parse_error"] = str(e)
                result["success"] = False
                
        except Exception as e:
            result["error"] = str(e)
            result["success"] = False
        
        return result
    
    def evaluate_on_samples(self, sample_indices: List[int]):
        """Evaluate model on multiple samples."""
        
        if not self.model:
            if not self.load_model():
                return self.results
        
        for idx in sample_indices:
            if self.verbose:
                print(f"\n  Sample {idx}:")
            
            try:
                # Load sample
                sample = load_cholecseg8k_sample(idx)
                image = sample['image']
                true_labels = sample['class_labels']
                masks = sample.get('masks', {})
                
                if self.verbose:
                    print(f"    True organs: {', '.join(true_labels[:3])}...")
                
                # Evaluate detection
                detection_result = self.evaluate_detection(image, true_labels)
                self.results["detection"].append(detection_result)
                
                if detection_result["success"] and self.verbose:
                    metrics = detection_result["metrics"]
                    print(f"    Detection: Acc={metrics['accuracy']:.2f}, F1={metrics['f1']:.2f}")
                
                # Evaluate pointing on first available organ
                if true_labels and true_labels[0] in masks:
                    organ = true_labels[0]
                    mask = masks[organ]
                    
                    pointing_result = self.evaluate_pointing(image, organ, mask)
                    self.results["pointing"].append(pointing_result)
                    
                    if pointing_result["success"] and self.verbose:
                        if "hit" in pointing_result["metrics"]:
                            hit = pointing_result["metrics"]["hit"]
                            print(f"    Pointing at {organ}: {'HIT' if hit else 'MISS'}")
                
            except Exception as e:
                if self.verbose:
                    print(f"    Error: {e}")
                self.results["errors"].append(f"Sample {idx}: {str(e)}")
        
        # Compute aggregate metrics
        self._compute_aggregate_metrics()
        
        return self.results
    
    def _compute_aggregate_metrics(self):
        """Compute aggregate metrics across all samples."""
        
        # Detection metrics
        detection_metrics = [r["metrics"] for r in self.results["detection"] if r.get("success")]
        if detection_metrics:
            self.results["aggregate_detection"] = {
                "mean_accuracy": np.mean([m["accuracy"] for m in detection_metrics]),
                "mean_precision": np.mean([m["precision"] for m in detection_metrics]),
                "mean_recall": np.mean([m["recall"] for m in detection_metrics]),
                "mean_f1": np.mean([m["f1"] for m in detection_metrics]),
                "success_rate": len(detection_metrics) / len(self.results["detection"])
            }
        
        # Pointing metrics
        pointing_metrics = [r["metrics"] for r in self.results["pointing"] 
                          if r.get("success") and "hit" in r.get("metrics", {})]
        if pointing_metrics:
            hits = [m["hit"] for m in pointing_metrics]
            distances = [m["distance_to_mask"] for m in pointing_metrics if not m["hit"]]
            
            self.results["aggregate_pointing"] = {
                "hit_rate": np.mean(hits),
                "mean_miss_distance": np.mean(distances) if distances else 0,
                "success_rate": len(pointing_metrics) / len(self.results["pointing"]) if self.results["pointing"] else 0
            }


def evaluate_all_models(num_samples: int = 10, models_to_test: Optional[List[str]] = None):
    """Evaluate all models on CholecSeg8k samples."""
    
    print("=" * 80)
    print("MODEL EVALUATION ON CHOLECSEG8K")
    print("=" * 80)
    
    # Check CUDA
    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA Available: {cuda_available}")
    if cuda_available:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Get sample indices
    print(f"\nSelecting {num_samples} balanced samples...")
    try:
        sample_indices = get_balanced_sample_indices(num_samples_per_organ=1)[:num_samples]
        print(f"✅ Selected {len(sample_indices)} samples")
    except Exception as e:
        print(f"⚠️  Could not get balanced samples: {e}")
        sample_indices = list(range(num_samples))
    
    # Models to evaluate
    if models_to_test is None:
        models_to_test = list(MODELS.keys())
    
    print(f"\nModels to evaluate: {len(models_to_test)}")
    
    all_results = {}
    
    for model_name in models_to_test:
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")
        
        # Skip VLM models if no CUDA
        if MODELS.get(model_name, {}).get("type") == "vlm" and not cuda_available:
            print("⏭️  Skipping (requires CUDA)")
            continue
        
        evaluator = ModelEvaluator(model_name, verbose=True)
        results = evaluator.evaluate_on_samples(sample_indices)
        all_results[model_name] = results
        
        # Clean up memory for VLM models
        if MODELS.get(model_name, {}).get("type") == "vlm":
            del evaluator.model
            if cuda_available:
                torch.cuda.empty_cache()
    
    # Print summary
    print_evaluation_summary(all_results)
    
    # Save results
    save_path = Path(__file__).parent / f"evaluation_results_{time.strftime('%Y%m%d_%H%M%S')}.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nResults saved to: {save_path}")
    
    return all_results


def print_evaluation_summary(results: Dict):
    """Print formatted evaluation summary."""
    
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    
    # Create summary table
    summary_data = []
    
    for model_name, model_results in results.items():
        row = {"Model": model_name.split("/")[-1][:25]}
        
        # Detection metrics
        if "aggregate_detection" in model_results:
            det = model_results["aggregate_detection"]
            row["Det Acc"] = f"{det['mean_accuracy']:.3f}"
            row["Det F1"] = f"{det['mean_f1']:.3f}"
            row["Det Success"] = f"{det['success_rate']:.1%}"
        else:
            row["Det Acc"] = "N/A"
            row["Det F1"] = "N/A"
            row["Det Success"] = "0%"
        
        # Pointing metrics
        if "aggregate_pointing" in model_results:
            point = model_results["aggregate_pointing"]
            row["Point Hit"] = f"{point['hit_rate']:.1%}"
            row["Point Dist"] = f"{point['mean_miss_distance']:.1f}"
        else:
            row["Point Hit"] = "N/A"
            row["Point Dist"] = "N/A"
        
        # Errors
        row["Errors"] = len(model_results.get("errors", []))
        
        summary_data.append(row)
    
    # Print as table
    if summary_data:
        df = pd.DataFrame(summary_data)
        print("\n" + df.to_string(index=False))
    
    # Best performers
    print("\n" + "-" * 40)
    print("BEST PERFORMERS")
    print("-" * 40)
    
    # Best detection accuracy
    best_det = max(results.items(), 
                   key=lambda x: x[1].get("aggregate_detection", {}).get("mean_accuracy", 0))
    if "aggregate_detection" in best_det[1]:
        print(f"Detection Accuracy: {best_det[0]} ({best_det[1]['aggregate_detection']['mean_accuracy']:.3f})")
    
    # Best pointing accuracy
    best_point = max(results.items(),
                    key=lambda x: x[1].get("aggregate_pointing", {}).get("hit_rate", 0))
    if "aggregate_pointing" in best_point[1]:
        print(f"Pointing Hit Rate:  {best_point[0]} ({best_point[1]['aggregate_pointing']['hit_rate']:.1%})")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate models on CholecSeg8k")
    parser.add_argument("--samples", type=int, default=5, help="Number of samples to evaluate")
    parser.add_argument("--models", nargs="+", help="Specific models to test")
    parser.add_argument("--api-only", action="store_true", help="Test only API models")
    parser.add_argument("--vlm-only", action="store_true", help="Test only VLM models")
    
    args = parser.parse_args()
    
    # Filter models based on arguments
    models_to_test = args.models
    
    if not models_to_test:
        if args.api_only:
            models_to_test = [m for m, c in MODELS.items() if c["type"] == "api"]
        elif args.vlm_only:
            models_to_test = [m for m, c in MODELS.items() if c["type"] == "vlm"]
        else:
            models_to_test = None  # Test all
    
    # Run evaluation
    results = evaluate_all_models(
        num_samples=args.samples,
        models_to_test=models_to_test
    )
    
    print("\n✨ Evaluation completed!")