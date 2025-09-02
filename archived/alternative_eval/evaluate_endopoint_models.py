#!/usr/bin/env python3
"""
Evaluate models using only endopoint modules.
Tests both API and VLM models from endopoint package on CholecSeg8k dataset.
"""

# Set multiprocessing start method for vLLM CUDA compatibility
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import time
import pickle
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import torch
from PIL import Image

# Import from endopoint only
from endopoint.datasets import build_dataset
from endopoint.models import (
    OpenAIAdapter,
    AnthropicAdapter, 
    GoogleAdapter,
    LLaVAModel,
    QwenVLModel,
    PixtralModel,
    DeepSeekVL2Model
)


# Model configurations
MODEL_CONFIGS = {
    # API Models
    "gpt-4o-mini": {"class": OpenAIAdapter, "type": "api"},
    "gpt-4o": {"class": OpenAIAdapter, "type": "api"},
    "claude-3-5-sonnet-20241022": {"class": AnthropicAdapter, "type": "api"},
    "gemini-2.0-flash-exp": {"class": GoogleAdapter, "type": "api"},
    
    # VLM Models
    "llava-hf/llava-v1.6-mistral-7b-hf": {"class": LLaVAModel, "type": "vlm"},
    "Qwen/Qwen2.5-VL-7B-Instruct": {"class": QwenVLModel, "type": "vlm"},
    "mistralai/Pixtral-12B-2409": {"class": PixtralModel, "type": "vlm"},
    "deepseek-ai/deepseek-vl2": {"class": DeepSeekVL2Model, "type": "vlm"},
}


class EndopointModelEvaluator:
    """Evaluate models using endopoint modules."""
    
    def __init__(self, model_name: str, verbose: bool = True):
        self.model_name = model_name
        self.verbose = verbose
        self.model = None
        self.dataset = None
        self.results = {
            "model": model_name,
            "detection": [],
            "pointing": [],
            "timing": [],
            "errors": []
        }
    
    def load_dataset(self):
        """Load CholecSeg8k dataset using endopoint."""
        try:
            if self.verbose:
                print("Loading CholecSeg8k dataset...")
            
            # Build dataset using endopoint
            self.dataset = build_dataset("cholecseg8k")
            
            # Get organ labels
            self.organ_classes = [
                self.dataset.id2label[i] 
                for i in self.dataset.label_ids
                if i != 0  # Skip background
            ]
            
            if self.verbose:
                print(f"✅ Dataset loaded with {len(self.organ_classes)} organ classes")
            
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"❌ Failed to load dataset: {e}")
            self.results["errors"].append(f"Dataset load error: {str(e)}")
            return False
    
    def load_model(self):
        """Load model using endopoint classes."""
        try:
            config = MODEL_CONFIGS.get(self.model_name, {})
            model_class = config.get("class")
            
            if not model_class:
                raise ValueError(f"Unknown model: {self.model_name}")
            
            # Model parameters
            params = {
                "model_name": self.model_name,
                "use_cache": False,
                "max_tokens": 300,
                "verbose": self.verbose
            }
            
            # API models need different initialization
            if config["type"] == "api":
                params["num_tries_per_request"] = 3
                params["batch_size"] = 1
                
                # For API adapters, temperature is set differently
                if model_class == GoogleAdapter:
                    params["temperature"] = 0.0
                elif model_class == AnthropicAdapter:
                    params["temperature"] = 0.0
            else:
                # VLM models
                params["temperature"] = 0.0
                params["batch_size"] = 1
                
                # Use vLLM if CUDA available
                if torch.cuda.is_available() and "deepseek" not in self.model_name.lower():
                    params["use_vllm"] = True
            
            if self.verbose:
                print(f"Loading {self.model_name}...")
            
            self.model = model_class(**params)
            
            if self.verbose:
                print(f"✅ Model loaded successfully")
            
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"❌ Failed to load model: {e}")
            self.results["errors"].append(f"Model load error: {str(e)}")
            return False
    
    def create_detection_prompt(self, organ_classes: List[str]) -> str:
        """Create organ detection prompt."""
        return f"""Analyze this laparoscopic surgery image.
Identify which of these organs/structures are visible:
{', '.join(organ_classes)}

Respond with ONLY a valid JSON object:
{{"{organ_classes[0]}": true/false, "{organ_classes[1]}": true/false, ...}}

Include ALL organs in your response."""
    
    def create_pointing_prompt(self, organ: str, width: int, height: int) -> str:
        """Create pointing prompt."""
        return f"""Look at this surgical image ({width}x{height} pixels).
Point to the center of: {organ}

Respond with ONLY JSON:
{{"present": true/false, "x": <number or null>, "y": <number or null>}}"""
    
    def evaluate_sample(self, split: str, index: int) -> Dict:
        """Evaluate a single sample."""
        result = {
            "split": split,
            "index": index,
            "detection": {},
            "pointing": {}
        }
        
        try:
            # Get example from dataset
            example = self.dataset.get_example(split, index)
            
            # Convert to tensors
            img_t, lab_t = self.dataset.example_to_tensors(example)
            
            # Get image as PIL
            image = example['image']
            if not isinstance(image, Image.Image):
                # Convert tensor to PIL if needed
                image = Image.fromarray((img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
            
            # Get presence vector
            presence = self.dataset.labels_to_presence_vector(lab_t)
            
            # Get true labels
            true_labels = [
                self.dataset.id2label[i]
                for i, p in enumerate(presence)
                if p > 0 and i in self.dataset.label_ids
            ]
            
            result["true_labels"] = true_labels
            
            if self.verbose:
                print(f"  True organs: {', '.join(true_labels[:3])}...")
            
            # Test 1: Detection
            detection_prompt = self.create_detection_prompt(self.organ_classes)
            
            # System prompt
            system_prompt = "You are an expert medical image analyst. Respond with valid JSON only."
            
            # Call model
            start_time = time.time()
            
            # Handle different model interfaces
            if isinstance(self.model, (OpenAIAdapter, AnthropicAdapter, GoogleAdapter)):
                # API adapters expect list of prompts
                responses = self.model([(image, detection_prompt)], system_prompt=system_prompt)
                response = responses[0] if responses else ""
            else:
                # VLM models use direct call
                response = self.model((image, detection_prompt), system_prompt=system_prompt)
            
            inference_time = time.time() - start_time
            
            result["detection"]["response"] = response
            result["detection"]["time"] = inference_time
            
            # Parse response
            try:
                if "{" in response and "}" in response:
                    json_str = response[response.find("{"):response.rfind("}")+1]
                    predictions = json.loads(json_str)
                else:
                    predictions = json.loads(response)
                
                # Calculate accuracy
                correct = 0
                total = 0
                for organ in self.organ_classes:
                    predicted = predictions.get(organ, False)
                    actual = organ in true_labels
                    if predicted == actual:
                        correct += 1
                    total += 1
                
                accuracy = correct / total if total > 0 else 0
                result["detection"]["accuracy"] = accuracy
                result["detection"]["predictions"] = predictions
                result["detection"]["success"] = True
                
                if self.verbose:
                    print(f"  Detection accuracy: {accuracy:.2%}")
                
            except Exception as e:
                result["detection"]["error"] = str(e)
                result["detection"]["success"] = False
                if self.verbose:
                    print(f"  Detection parse error: {e}")
            
            # Test 2: Pointing (on first organ if available)
            if true_labels:
                target_organ = true_labels[0]
                pointing_prompt = self.create_pointing_prompt(
                    target_organ, image.width, image.height
                )
                
                start_time = time.time()
                
                # Call model
                if isinstance(self.model, (OpenAIAdapter, AnthropicAdapter, GoogleAdapter)):
                    responses = self.model([(image, pointing_prompt)], system_prompt=system_prompt)
                    response = responses[0] if responses else ""
                else:
                    response = self.model((image, pointing_prompt), system_prompt=system_prompt)
                
                inference_time = time.time() - start_time
                
                result["pointing"]["response"] = response
                result["pointing"]["time"] = inference_time
                result["pointing"]["target"] = target_organ
                
                # Parse response
                try:
                    if "{" in response and "}" in response:
                        json_str = response[response.find("{"):response.rfind("}")+1]
                        prediction = json.loads(json_str)
                    else:
                        prediction = json.loads(response)
                    
                    result["pointing"]["prediction"] = prediction
                    result["pointing"]["success"] = True
                    
                    if self.verbose and prediction.get("present"):
                        x, y = prediction.get("x"), prediction.get("y")
                        print(f"  Pointing at {target_organ}: ({x}, {y})")
                    
                except Exception as e:
                    result["pointing"]["error"] = str(e)
                    result["pointing"]["success"] = False
                    if self.verbose:
                        print(f"  Pointing parse error: {e}")
            
        except Exception as e:
            result["error"] = str(e)
            if self.verbose:
                print(f"  Sample error: {e}")
        
        return result
    
    def evaluate(self, split: str = "train", num_samples: int = 5):
        """Evaluate model on dataset samples."""
        
        # Load dataset and model
        if not self.dataset and not self.load_dataset():
            return self.results
        
        if not self.model and not self.load_model():
            return self.results
        
        # Get random sample indices
        total_samples = self.dataset.total(split)
        sample_indices = random.sample(range(total_samples), min(num_samples, total_samples))
        
        if self.verbose:
            print(f"\nEvaluating on {len(sample_indices)} samples from {split} split...")
        
        # Evaluate each sample
        for i, idx in enumerate(sample_indices):
            if self.verbose:
                print(f"\nSample {i+1}/{len(sample_indices)} (index {idx}):")
            
            result = self.evaluate_sample(split, idx)
            self.results["detection"].append(result.get("detection", {}))
            self.results["pointing"].append(result.get("pointing", {}))
        
        # Compute aggregate metrics
        self._compute_aggregate_metrics()
        
        return self.results
    
    def _compute_aggregate_metrics(self):
        """Compute aggregate metrics."""
        
        # Detection metrics
        detection_accuracies = [
            r["accuracy"] for r in self.results["detection"]
            if r.get("success") and "accuracy" in r
        ]
        
        if detection_accuracies:
            self.results["aggregate"] = {
                "detection_accuracy": np.mean(detection_accuracies),
                "detection_success_rate": len(detection_accuracies) / len(self.results["detection"])
            }
        
        # Pointing success rate
        pointing_successes = [
            r.get("success", False) for r in self.results["pointing"]
        ]
        
        if pointing_successes:
            self.results["aggregate"]["pointing_success_rate"] = np.mean(pointing_successes)


def main(num_samples: int = 5, test_api: bool = True, test_vlm: bool = True):
    """Main evaluation function."""
    
    print("=" * 80)
    print("ENDOPOINT MODEL EVALUATION")
    print("=" * 80)
    
    # Check CUDA
    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA Available: {cuda_available}")
    if cuda_available:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Select models to test
    models_to_test = []
    
    if test_api:
        models_to_test.extend([
            "gpt-4o-mini",
            "claude-3-5-sonnet-20241022",
            "gemini-2.0-flash-exp",
        ])
    
    if test_vlm and cuda_available:
        models_to_test.extend([
            "llava-hf/llava-v1.6-mistral-7b-hf",
            # "Qwen/Qwen2.5-VL-7B-Instruct",
            # "mistralai/Pixtral-12B-2409",
        ])
    
    print(f"\nModels to test: {len(models_to_test)}")
    for model in models_to_test:
        print(f"  - {model}")
    
    all_results = {}
    
    # Evaluate each model
    for model_name in models_to_test:
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")
        
        evaluator = EndopointModelEvaluator(model_name, verbose=True)
        results = evaluator.evaluate(split="train", num_samples=num_samples)
        all_results[model_name] = results
        
        # Clean up VLM models
        if MODEL_CONFIGS.get(model_name, {}).get("type") == "vlm":
            del evaluator.model
            if cuda_available:
                torch.cuda.empty_cache()
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    summary_data = []
    for model_name, results in all_results.items():
        row = {
            "Model": model_name.split("/")[-1][:30],
            "Type": MODEL_CONFIGS.get(model_name, {}).get("type", "unknown")
        }
        
        if "aggregate" in results:
            agg = results["aggregate"]
            row["Detection Acc"] = f"{agg.get('detection_accuracy', 0):.2%}"
            row["Detection Success"] = f"{agg.get('detection_success_rate', 0):.1%}"
            row["Pointing Success"] = f"{agg.get('pointing_success_rate', 0):.1%}"
        else:
            row["Detection Acc"] = "N/A"
            row["Detection Success"] = "N/A"
            row["Pointing Success"] = "N/A"
        
        row["Errors"] = len(results.get("errors", []))
        summary_data.append(row)
    
    if summary_data:
        df = pd.DataFrame(summary_data)
        print("\n" + df.to_string(index=False))
    
    # Save results
    save_path = Path(__file__).parent / f"endopoint_eval_{time.strftime('%Y%m%d_%H%M%S')}.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nResults saved to: {save_path}")
    
    print("\n✨ Evaluation completed!")
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate endopoint models")
    parser.add_argument("--samples", type=int, default=3, help="Number of samples")
    parser.add_argument("--api-only", action="store_true", help="Test API models only")
    parser.add_argument("--vlm-only", action="store_true", help="Test VLM models only")
    
    args = parser.parse_args()
    
    test_api = True
    test_vlm = True
    
    if args.api_only:
        test_vlm = False
    elif args.vlm_only:
        test_api = False
    
    main(
        num_samples=args.samples,
        test_api=test_api,
        test_vlm=test_vlm
    )