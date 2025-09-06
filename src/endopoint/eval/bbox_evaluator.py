"""
Bounding box evaluator supporting both separate and combined detection modes.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm

from ..datasets.base import DatasetAdapter
from ..models import create_model
from ..prompts.bbox_prompts import get_bbox_prompt, get_combined_bbox_prompt


@dataclass
class Canvas:
    """Simple canvas representation."""
    width: int
    height: int


class BBoxPrediction:
    """Container for bounding box prediction."""
    
    def __init__(self, present: int, bboxes: List[List[int]]):
        self.present = present
        self.bboxes = bboxes
    
    @classmethod
    def from_json(cls, response: str, organ_name: str) -> 'BBoxPrediction':
        """Parse prediction from JSON response.
        
        Args:
            response: JSON response string
            organ_name: Name of the organ to extract
            
        Returns:
            BBoxPrediction instance
        """
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                
                # Handle both formats: direct organ key or nested under 'name'
                if organ_name in data:
                    organ_data = data[organ_name]
                elif isinstance(data, dict) and data.get('name') == organ_name:
                    organ_data = data
                else:
                    # Search in nested structure
                    for key, value in data.items():
                        if isinstance(value, dict) and (key == organ_name or value.get('name') == organ_name):
                            organ_data = value
                            break
                    else:
                        return cls(0, [])
                
                # Extract presence and bbox
                present = 1 if organ_data.get('present', False) else 0
                
                # Handle different bbox formats
                bbox = organ_data.get('bbox', organ_data.get('bboxes', []))
                if bbox and isinstance(bbox[0], list):
                    bboxes = bbox  # Already list of bboxes
                elif bbox and present:
                    bboxes = [bbox]  # Single bbox
                else:
                    bboxes = []
                
                return cls(present, bboxes)
        except Exception as e:
            pass
        
        # Default to not present
        return cls(0, [])


def compute_iou(bbox1: List[int], bbox2: List[int]) -> float:
    """Compute IoU between two bounding boxes.
    
    Args:
        bbox1: [x1, y1, x2, y2]
        bbox2: [x1, y1, x2, y2]
        
    Returns:
        IoU value between 0 and 1
    """
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def compute_best_iou(pred_bboxes: List[List[int]], gt_bboxes: List[List[int]]) -> float:
    """Compute best IoU between predicted and ground truth bboxes.
    
    Args:
        pred_bboxes: List of predicted bboxes
        gt_bboxes: List of ground truth bboxes
        
    Returns:
        Best IoU value
    """
    if not pred_bboxes or not gt_bboxes:
        return 0.0
    
    best_iou = 0.0
    for pred_bbox in pred_bboxes:
        for gt_bbox in gt_bboxes:
            iou = compute_iou(pred_bbox, gt_bbox)
            best_iou = max(best_iou, iou)
    
    return best_iou


def compute_bbox_to_mask_iou(pred_bboxes: List[List[int]], mask: np.ndarray) -> float:
    """Compute best IoU between predicted bboxes and segmentation mask.
    
    Args:
        pred_bboxes: List of predicted bboxes [x1, y1, x2, y2]
        mask: Binary segmentation mask (H, W)
        
    Returns:
        Best IoU value between any predicted bbox and the mask
    """
    if not pred_bboxes or mask.sum() == 0:
        return 0.0
    
    mask_binary = (mask > 0).astype(np.uint8)
    best_iou = 0.0
    
    for pred_bbox in pred_bboxes:
        x1, y1, x2, y2 = pred_bbox
        
        # Ensure bbox is within image bounds
        h, w = mask.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            continue
        
        # Create bbox mask
        bbox_mask = np.zeros_like(mask_binary, dtype=np.uint8)
        bbox_mask[y1:y2, x1:x2] = 1
        
        # Compute IoU
        intersection = np.logical_and(bbox_mask, mask_binary).sum()
        union = np.logical_or(bbox_mask, mask_binary).sum()
        
        iou = intersection / union if union > 0 else 0.0
        best_iou = max(best_iou, iou)
    
    return best_iou


class BoundingBoxEvaluator:
    """Evaluator for bounding box detection supporting multiple modes."""
    
    def __init__(
        self,
        models: List[str],
        dataset: Optional[Any],
        dataset_adapter: DatasetAdapter,
        canvas_width: int = 768,
        canvas_height: int = 768,
        output_dir: Optional[Path] = None,
        use_cache: bool = True,
        min_pixels: int = 50,
        use_timestamp: bool = True
    ):
        """Initialize evaluator.
        
        Args:
            models: List of model names to evaluate
            dataset: Dataset object (can be None if using adapter only)
            dataset_adapter: Dataset adapter
            canvas_width: Canvas width
            canvas_height: Canvas height
            output_dir: Output directory for results
            use_cache: Whether to use cache
            min_pixels: Minimum pixels for valid detection
            use_timestamp: Whether to use timestamped output directory
        """
        self.models = models
        self.dataset = dataset
        self.adapter = dataset_adapter
        self.canvas = Canvas(width=canvas_width, height=canvas_height)
        self.use_cache = use_cache
        self.min_pixels = min_pixels
        self.use_timestamp = use_timestamp
        
        # Set output directory based on timestamp flag
        if output_dir:
            self.output_dir = output_dir
        elif use_timestamp:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dataset_tag = getattr(dataset_adapter, 'dataset_tag', 'unknown')
            self.output_dir = Path(f"results/bbox_{dataset_tag}_{timestamp}")
        else:
            # Persistent directory without timestamp
            dataset_tag = getattr(dataset_adapter, 'dataset_tag', 'unknown')
            self.output_dir = Path(f"results/bbox_{dataset_tag}")
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup cache directory
        self.cache_dir = Path("cache") / "bbox_eval"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, model_name: str, prompt: str, image_idx: int = None,
                      detection_mode: str = None, use_fewshot: bool = False) -> str:
        """Generate cache key for a prompt with mode and fewshot status.
        
        This ensures different evaluation modes (zero/few-shot, separate/combined)
        have distinct cache entries to prevent overwrites.
        """
        parts = [model_name]
        if image_idx is not None:
            parts.append(str(image_idx))
        if detection_mode:
            parts.append(detection_mode)
        parts.append('fewshot' if use_fewshot else 'zeroshot')
        parts.append(prompt[:500])  # Use first 500 chars of prompt
        
        content = ":".join(parts)
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cached_response(self, model_name: str, prompt: str, image_idx: int = None) -> Optional[str]:
        """Get cached response if available."""
        if not self.use_cache:
            return None
        
        cache_key = self._get_cache_key(model_name, prompt, image_idx)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                data = json.load(f)
                return data.get('response')
        
        return None
    
    def _save_cached_response(self, model_name: str, prompt: str, response: str, image_idx: int = None):
        """Save response to cache."""
        if not self.use_cache:
            return
        
        cache_key = self._get_cache_key(model_name, prompt, image_idx)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        with open(cache_file, 'w') as f:
            json.dump({
                'model': model_name,
                'prompt': prompt[:200],  # Save first 200 chars for reference
                'response': response,
                'image_idx': image_idx
            }, f)
    
    def _extract_ground_truth_bboxes(self, image_idx: int, organ_id: int, split: str = "train") -> List[List[int]]:
        """Extract ground truth bounding boxes for an organ.
        
        Args:
            image_idx: Index of the image
            organ_id: ID of the organ
            split: Dataset split
            
        Returns:
            List of bounding boxes [x1, y1, x2, y2]
        """
        # Get example using global index
        if self.dataset is not None:
            # For HuggingFace datasets, image_idx is already global
            example = self.dataset[split][image_idx]
        else:
            # Use global index directly with the adapter
            example = self.adapter.get_example_by_global_index(image_idx)
        
        # Get the semantic mask using the adapter's conversion
        if hasattr(self.adapter, 'example_to_tensors'):
            # Use adapter's conversion method
            import torch
            _, lab_tensor = self.adapter.example_to_tensors(example)
            mask = lab_tensor.numpy()
        else:
            raise NotImplementedError("Adapter must have example_to_tensors method")
        
        # Get organ mask
        organ_mask = (mask == organ_id).astype(np.uint8)
        
        if organ_mask.sum() < self.min_pixels:
            return []
        
        # Find connected components using CV2
        try:
            import cv2
            contours, _ = cv2.findContours(organ_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Combine all contours into one bbox
                x_min, y_min = float('inf'), float('inf')
                x_max, y_max = 0, 0
                
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x + w)
                    y_max = max(y_max, y + h)
                
                return [[int(x_min), int(y_min), int(x_max), int(y_max)]]
        except ImportError:
            # Fallback if cv2 not available
            rows, cols = np.where(organ_mask)
            if len(rows) > 0:
                return [[int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())]]
        
        return []
    
    def _build_combined_prompt(self, organ_classes: Dict[int, str], prompt_type: str = "standard") -> str:
        """Build a combined prompt for all organs.
        
        Args:
            organ_classes: Dictionary of organ IDs to names
            prompt_type: Type of prompt
            
        Returns:
            Combined prompt string
        """
        organ_names = list(organ_classes.values())
        return get_combined_bbox_prompt(
            organ_names=organ_names,
            prompt_type=prompt_type,
            use_fewshot=False,
            canvas_width=self.canvas.width,
            canvas_height=self.canvas.height
        )
    
    def _build_fewshot_prompt(
        self, 
        organ_classes: Dict[int, str],
        fewshot_examples: List[Dict],
        prompt_type: str = "standard"
    ) -> str:
        """Build few-shot prompt with examples.
        
        Args:
            organ_classes: Dictionary of organ IDs to names  
            fewshot_examples: List of few-shot examples
            prompt_type: Type of prompt
            
        Returns:
            Few-shot prompt string
        """
        organ_names = list(organ_classes.values())
        return get_combined_bbox_prompt(
            organ_names=organ_names,
            prompt_type=prompt_type,
            use_fewshot=True,
            examples=fewshot_examples,
            canvas_width=self.canvas.width,
            canvas_height=self.canvas.height
        )
    
    def evaluate_model(
        self,
        model_name: str,
        test_indices: List[int],
        detection_mode: str = "separate",  # "separate" or "combined"
        prompt_type: str = "standard",
        use_fewshot: bool = False,
        fewshot_plan: Optional[Dict] = None,
        fewshot_examples: Optional[List[Dict]] = None,
        split: str = "test"
    ) -> Dict:
        """Evaluate a model on the test set.
        
        Args:
            model_name: Name of the model to evaluate
            test_indices: List of test sample indices
            detection_mode: "separate" (one query per organ) or "combined" (all organs in one query)
            prompt_type: Type of prompt to use
            use_fewshot: Whether to use few-shot examples
            fewshot_plan: Few-shot plan for separate mode
            fewshot_examples: Few-shot examples for combined mode
            split: Dataset split
            
        Returns:
            Dictionary with evaluation results
        """
        # Initialize model
        model = create_model(model_name, use_cache=self.use_cache)
        system_prompt = "You are an expert medical image analyst specializing in laparoscopic surgery organ detection."
        
        # Results storage
        predictions = []
        
        # Get organ classes (only those in label_ids, excluding background)
        organ_classes = {id: self.adapter.id2label[id] for id in self.adapter.label_ids}
        
        # Choose evaluation mode
        if detection_mode == "combined":
            return self._evaluate_combined(
                model, model_name, test_indices, organ_classes,
                prompt_type, use_fewshot, fewshot_examples,
                split, system_prompt, predictions
            )
        else:  # separate
            return self._evaluate_separate(
                model, model_name, test_indices, organ_classes,
                prompt_type, use_fewshot, fewshot_plan,
                split, system_prompt, predictions
            )
    
    def _evaluate_combined(
        self, model, model_name, test_indices, organ_classes,
        prompt_type, use_fewshot, fewshot_examples,
        split, system_prompt, predictions
    ):
        """Evaluate using combined detection (all organs in one query)."""
        
        pbar = tqdm(total=len(test_indices), desc=f"Evaluating {model_name} (combined)")
        
        for test_idx in test_indices:
            # Check if result already exists
            eval_type = f"{'fewshot' if use_fewshot else 'zeroshot'}_combined"
            model_dir = self.output_dir / eval_type / model_name.replace("/", "_")
            sample_file = model_dir / f"test_{test_idx:05d}.json"
            
            if sample_file.exists() and self.use_cache:
                # Skip if already processed
                pbar.update(1)
                continue
            
            # Get example using global index
            if self.dataset is not None:
                # For HuggingFace datasets, test_idx should already be global
                example = self.dataset[split][test_idx]
            else:
                # Use global index directly with the adapter
                example = self.adapter.get_example_by_global_index(test_idx)
            image = example['image']
            
            # Get segmentation mask for IoU calculations
            _, lab_tensor = self.adapter.example_to_tensors(example)
            
            # Build prompt (combined for all organs)
            if use_fewshot and fewshot_examples:
                prompt = self._build_fewshot_prompt(organ_classes, fewshot_examples, prompt_type)
            else:
                prompt = self._build_combined_prompt(organ_classes, prompt_type)
            
            # Check cache with detection mode and fewshot status
            cached_response = self._get_cached_response(model_name, prompt, test_idx)
            
            if cached_response:
                response = cached_response
            else:
                # Query model - single call for all organs
                response = model([(image, prompt)], system_prompt=system_prompt)[0]
                self._save_cached_response(model_name, prompt, response, test_idx)
            
            # Parse predictions for all organs
            for organ_id, organ_name in organ_classes.items():
                # Get ground truth
                gt_bboxes = self._extract_ground_truth_bboxes(test_idx, organ_id, split)
                gt_present = 1 if gt_bboxes else 0
                
                # Parse prediction for this organ
                pred = BBoxPrediction.from_json(response, organ_name)
                
                # Compute both IoU types for comprehensive evaluation
                iou_bbox_to_bbox = 0.0
                iou_bbox_to_mask = 0.0
                
                if gt_present and pred.present and gt_bboxes and pred.bboxes:
                    # Bbox-to-Bbox IoU (current standard)
                    iou_bbox_to_bbox = compute_best_iou(pred.bboxes, gt_bboxes)
                    
                    # Bbox-to-Mask IoU (alternative metric)
                    # Get organ mask for this specific organ
                    organ_mask = (lab_tensor.numpy() == organ_id).astype(np.uint8)
                    iou_bbox_to_mask = compute_bbox_to_mask_iou(pred.bboxes, organ_mask)
                
                # Store results with both IoU metrics
                predictions.append({
                    'test_idx': test_idx,
                    'organ_id': organ_id,
                    'organ_name': organ_name,
                    'predicted_present': pred.present,
                    'predicted_bboxes': pred.bboxes,
                    'ground_truth_present': gt_present,
                    'ground_truth_bboxes': gt_bboxes,
                    'iou_bbox_to_bbox': iou_bbox_to_bbox,
                    'iou_bbox_to_mask': iou_bbox_to_mask
                })
            
            pbar.update(1)
        
        pbar.close()
        
        # Load any existing predictions from files
        eval_type = f"{'fewshot' if use_fewshot else 'zeroshot'}_combined"
        model_dir = self.output_dir / eval_type / model_name.replace("/", "_")
        
        if model_dir.exists():
            # Load existing predictions to include in metrics
            existing_files = sorted(model_dir.glob("test_*.json"))
            for pred_file in existing_files:
                with open(pred_file, 'r') as f:
                    sample_data = json.load(f)
                    # Add to predictions if not already there
                    test_idx = sample_data['sample_idx']
                    if not any(p['test_idx'] == test_idx for p in predictions):
                        for organ_data in sample_data['organs']:
                            predictions.append({
                                'test_idx': test_idx,
                                'organ_id': organ_data['organ_id'],
                                'organ_name': organ_data['organ_name'],
                                'predicted_present': organ_data['predicted_present'],
                                'predicted_bboxes': organ_data['predicted_bboxes'],
                                'ground_truth_present': organ_data['ground_truth_present'],
                                'ground_truth_bboxes': organ_data['ground_truth_bboxes']
                            })
        
        # Compute metrics and save
        metrics = self._compute_metrics(predictions)
        self._save_results(model_name, prompt_type, use_fewshot, predictions, metrics, "combined")
        
        return {'predictions': predictions, 'metrics': metrics}
    
    def _evaluate_separate(
        self, model, model_name, test_indices, organ_classes,
        prompt_type, use_fewshot, fewshot_plan,
        split, system_prompt, predictions
    ):
        """Evaluate using separate detection (one query per organ)."""
        
        total_evals = len(test_indices) * len(organ_classes)
        pbar = tqdm(total=total_evals, desc=f"Evaluating {model_name} (separate)")
        
        for test_idx in test_indices:
            # Check if result already exists
            eval_type = f"{'fewshot' if use_fewshot else 'zeroshot'}_separate"
            model_dir = self.output_dir / eval_type / model_name.replace("/", "_")
            sample_file = model_dir / f"test_{test_idx:05d}.json"
            
            if sample_file.exists() and self.use_cache:
                # Skip if already processed
                pbar.update(1)
                continue
            
            # Get example using global index
            if self.dataset is not None:
                # For HuggingFace datasets, test_idx should already be global
                example = self.dataset[split][test_idx]
            else:
                # Use global index directly with the adapter
                example = self.adapter.get_example_by_global_index(test_idx)
            image = example['image']
            
            # Get segmentation mask for IoU calculations
            _, lab_tensor = self.adapter.example_to_tensors(example)
            
            for organ_id, organ_name in organ_classes.items():
                # Get ground truth
                gt_bboxes = self._extract_ground_truth_bboxes(test_idx, organ_id, split)
                gt_present = 1 if gt_bboxes else 0
                
                # Generate prompt for this specific organ
                if use_fewshot and fewshot_plan:
                    # Get few-shot examples for this organ (plan is organized by organ ID)
                    organ_examples = fewshot_plan.get('plan', {}).get(str(organ_id), {})
                    
                    # Warn if no examples found for this organ
                    if not organ_examples:
                        print(f"⚠️ WARNING: No few-shot examples found for organ '{organ_name}' (ID: {organ_id})")
                        print(f"   Available organ IDs in plan: {list(fewshot_plan.get('plan', {}).keys())}")
                    elif not any(organ_examples.get(k) for k in ['positives', 'negatives_absent', 'negatives_wrong_bbox']):
                        print(f"⚠️ WARNING: Few-shot examples for '{organ_name}' are empty")
                    
                    prompt = get_bbox_prompt(
                        organ_name=organ_name,
                        prompt_type=prompt_type,
                        use_fewshot=True,
                        examples=organ_examples,
                        canvas_width=self.canvas.width,
                        canvas_height=self.canvas.height
                    )
                else:
                    prompt = get_bbox_prompt(
                        organ_name=organ_name,
                        prompt_type=prompt_type,
                        use_fewshot=False,
                        canvas_width=self.canvas.width,
                        canvas_height=self.canvas.height
                    )
                
                # Check cache with unique key for this configuration
                cache_key = f"{test_idx}_{organ_id}"
                cached_response = self._get_cached_response(model_name, prompt + cache_key)
                
                if cached_response:
                    response = cached_response
                else:
                    # Query model
                    response = model([(image, prompt)], system_prompt=system_prompt)[0]
                    self._save_cached_response(model_name, prompt + cache_key, response)
                
                # Parse prediction
                pred = BBoxPrediction.from_json(response, organ_name)
                
                # Compute both IoU types for comprehensive evaluation
                iou_bbox_to_bbox = 0.0
                iou_bbox_to_mask = 0.0
                
                if gt_present and pred.present and gt_bboxes and pred.bboxes:
                    # Bbox-to-Bbox IoU (current standard)
                    iou_bbox_to_bbox = compute_best_iou(pred.bboxes, gt_bboxes)
                    
                    # Bbox-to-Mask IoU (alternative metric)
                    # Get organ mask for this specific organ
                    organ_mask = (lab_tensor.numpy() == organ_id).astype(np.uint8)
                    iou_bbox_to_mask = compute_bbox_to_mask_iou(pred.bboxes, organ_mask)
                
                # Store results with both IoU metrics
                predictions.append({
                    'test_idx': test_idx,
                    'organ_id': organ_id,
                    'organ_name': organ_name,
                    'predicted_present': pred.present,
                    'predicted_bboxes': pred.bboxes,
                    'ground_truth_present': gt_present,
                    'ground_truth_bboxes': gt_bboxes,
                    'iou_bbox_to_bbox': iou_bbox_to_bbox,
                    'iou_bbox_to_mask': iou_bbox_to_mask
                })
                
                pbar.update(1)
        
        pbar.close()
        
        # Load any existing predictions from files
        eval_type = f"{'fewshot' if use_fewshot else 'zeroshot'}_separate"
        model_dir = self.output_dir / eval_type / model_name.replace("/", "_")
        
        if model_dir.exists():
            # Load existing predictions to include in metrics
            existing_files = sorted(model_dir.glob("test_*.json"))
            for pred_file in existing_files:
                with open(pred_file, 'r') as f:
                    sample_data = json.load(f)
                    # Add to predictions if not already there
                    test_idx = sample_data['sample_idx']
                    if not any(p['test_idx'] == test_idx for p in predictions):
                        for organ_data in sample_data['organs']:
                            predictions.append({
                                'test_idx': test_idx,
                                'organ_id': organ_data['organ_id'],
                                'organ_name': organ_data['organ_name'],
                                'predicted_present': organ_data['predicted_present'],
                                'predicted_bboxes': organ_data['predicted_bboxes'],
                                'ground_truth_present': organ_data['ground_truth_present'],
                                'ground_truth_bboxes': organ_data['ground_truth_bboxes']
                            })
        
        # Compute metrics and save
        metrics = self._compute_metrics(predictions)
        self._save_results(model_name, prompt_type, use_fewshot, predictions, metrics, "separate")
        
        return {'predictions': predictions, 'metrics': metrics}
    
    def _compute_metrics(self, predictions: List[Dict]) -> Dict:
        """Compute evaluation metrics.
        
        Args:
            predictions: List of prediction dictionaries
            
        Returns:
            Dictionary with computed metrics
        """
        # Aggregate by organ
        organ_results = {}
        
        for pred in predictions:
            organ_name = pred['organ_name']
            if organ_name not in organ_results:
                organ_results[organ_name] = {
                    'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0,
                    'ious_bbox_to_bbox': [],
                    'ious_bbox_to_mask': []
                }
            
            gt_present = pred['ground_truth_present']
            pred_present = pred['predicted_present']
            
            if gt_present and pred_present:
                # True positive - get both IoU types
                organ_results[organ_name]['tp'] += 1
                
                # Use pre-computed IoU values if available (new format)
                if 'iou_bbox_to_bbox' in pred:
                    iou_bbox_to_bbox = pred['iou_bbox_to_bbox']
                    iou_bbox_to_mask = pred['iou_bbox_to_mask']
                else:
                    # Fallback to legacy computation (old format)
                    iou_bbox_to_bbox = compute_best_iou(pred['predicted_bboxes'], pred['ground_truth_bboxes'])
                    iou_bbox_to_mask = 0.0  # Not available in legacy format
                
                organ_results[organ_name]['ious_bbox_to_bbox'].append(iou_bbox_to_bbox)
                organ_results[organ_name]['ious_bbox_to_mask'].append(iou_bbox_to_mask)
            elif not gt_present and not pred_present:
                organ_results[organ_name]['tn'] += 1
            elif gt_present and not pred_present:
                organ_results[organ_name]['fn'] += 1
            else:
                organ_results[organ_name]['fp'] += 1
        
        # Compute per-organ metrics
        organ_metrics = {}
        all_ious_bbox_to_bbox = []
        all_ious_bbox_to_mask = []
        
        for organ_name, results in organ_results.items():
            tp, fp, tn, fn = results['tp'], results['fp'], results['tn'], results['fn']
            ious_bbox_to_bbox = results['ious_bbox_to_bbox']
            ious_bbox_to_mask = results['ious_bbox_to_mask']
            
            # Presence accuracy
            total = tp + fp + tn + fn
            presence_acc = (tp + tn) / total if total > 0 else 0
            
            # Bbox-to-Bbox IoU metrics
            if ious_bbox_to_bbox:
                mean_iou_b2b = np.mean(ious_bbox_to_bbox)
                iou_at_03_b2b = np.mean([iou >= 0.3 for iou in ious_bbox_to_bbox])
                iou_at_05_b2b = np.mean([iou >= 0.5 for iou in ious_bbox_to_bbox])
                iou_at_075_b2b = np.mean([iou >= 0.75 for iou in ious_bbox_to_bbox])
                all_ious_bbox_to_bbox.extend(ious_bbox_to_bbox)
            else:
                mean_iou_b2b = iou_at_03_b2b = iou_at_05_b2b = iou_at_075_b2b = 0
            
            # Bbox-to-Mask IoU metrics
            if ious_bbox_to_mask:
                mean_iou_b2m = np.mean(ious_bbox_to_mask)
                iou_at_03_b2m = np.mean([iou >= 0.3 for iou in ious_bbox_to_mask])
                iou_at_05_b2m = np.mean([iou >= 0.5 for iou in ious_bbox_to_mask])
                iou_at_075_b2m = np.mean([iou >= 0.75 for iou in ious_bbox_to_mask])
                all_ious_bbox_to_mask.extend(ious_bbox_to_mask)
            else:
                mean_iou_b2m = iou_at_03_b2m = iou_at_05_b2m = iou_at_075_b2m = 0
            
            organ_metrics[organ_name] = {
                'presence_accuracy': presence_acc,
                # Bbox-to-Bbox IoU metrics (current standard)
                'mean_iou_bbox_to_bbox': mean_iou_b2b,
                'iou_at_0.3_bbox_to_bbox': iou_at_03_b2b,
                'iou_at_0.5_bbox_to_bbox': iou_at_05_b2b,
                'iou_at_0.75_bbox_to_bbox': iou_at_075_b2b,
                # Bbox-to-Mask IoU metrics (alternative)
                'mean_iou_bbox_to_mask': mean_iou_b2m,
                'iou_at_0.3_bbox_to_mask': iou_at_03_b2m,
                'iou_at_0.5_bbox_to_mask': iou_at_05_b2m,
                'iou_at_0.75_bbox_to_mask': iou_at_075_b2m,
                # Legacy fields for backward compatibility
                'mean_iou': mean_iou_b2b,
                'iou_at_0.3': iou_at_03_b2b,
                'iou_at_0.5': iou_at_05_b2b,
                'iou_at_0.75': iou_at_075_b2b,
                'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
            }
        
        # Overall metrics
        overall_metrics = {
            'presence_accuracy': np.mean([m['presence_accuracy'] for m in organ_metrics.values()]),
            # Bbox-to-Bbox IoU metrics (current standard)
            'mean_iou_bbox_to_bbox': np.mean(all_ious_bbox_to_bbox) if all_ious_bbox_to_bbox else 0,
            'iou_at_0.3_bbox_to_bbox': np.mean([iou >= 0.3 for iou in all_ious_bbox_to_bbox]) if all_ious_bbox_to_bbox else 0,
            'iou_at_0.5_bbox_to_bbox': np.mean([iou >= 0.5 for iou in all_ious_bbox_to_bbox]) if all_ious_bbox_to_bbox else 0,
            'iou_at_0.75_bbox_to_bbox': np.mean([iou >= 0.75 for iou in all_ious_bbox_to_bbox]) if all_ious_bbox_to_bbox else 0,
            # Bbox-to-Mask IoU metrics (alternative)
            'mean_iou_bbox_to_mask': np.mean(all_ious_bbox_to_mask) if all_ious_bbox_to_mask else 0,
            'iou_at_0.3_bbox_to_mask': np.mean([iou >= 0.3 for iou in all_ious_bbox_to_mask]) if all_ious_bbox_to_mask else 0,
            'iou_at_0.5_bbox_to_mask': np.mean([iou >= 0.5 for iou in all_ious_bbox_to_mask]) if all_ious_bbox_to_mask else 0,
            'iou_at_0.75_bbox_to_mask': np.mean([iou >= 0.75 for iou in all_ious_bbox_to_mask]) if all_ious_bbox_to_mask else 0,
            # Legacy fields for backward compatibility
            'mean_iou': np.mean(all_ious_bbox_to_bbox) if all_ious_bbox_to_bbox else 0,
            'iou_at_0.3': np.mean([iou >= 0.3 for iou in all_ious_bbox_to_bbox]) if all_ious_bbox_to_bbox else 0,
            'iou_at_0.5': np.mean([iou >= 0.5 for iou in all_ious_bbox_to_bbox]) if all_ious_bbox_to_bbox else 0,
            'iou_at_0.75': np.mean([iou >= 0.75 for iou in all_ious_bbox_to_bbox]) if all_ious_bbox_to_bbox else 0,
            'per_organ': organ_metrics
        }
        
        return overall_metrics
    
    def _save_results(
        self, model_name: str, prompt_type: str, use_fewshot: bool,
        predictions: List[Dict], metrics: Dict, mode: str
    ):
        """Save evaluation results with individual prediction files."""
        # Create output directory
        eval_type = f"{'fewshot' if use_fewshot else 'zeroshot'}_{mode}"
        model_dir = self.output_dir / eval_type / model_name.replace("/", "_")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Group predictions by test index
        predictions_by_idx = {}
        for pred in predictions:
            test_idx = pred['test_idx']
            if test_idx not in predictions_by_idx:
                predictions_by_idx[test_idx] = {
                    'test_idx': test_idx,
                    'organs': []
                }
            predictions_by_idx[test_idx]['organs'].append(pred)
        
        # Save individual prediction files
        for test_idx, sample_data in predictions_by_idx.items():
            # Create structured output for this sample
            sample_output = {
                'sample_idx': test_idx,
                'y_true': [],  # Ground truth presence for each organ
                'y_pred': [],  # Predicted presence for each organ
                'ious': [],     # IoU scores for true positives
                'bboxes_true': {},  # Ground truth bboxes by organ
                'bboxes_pred': {},  # Predicted bboxes by organ
                'organs': []    # Detailed per-organ results
            }
            
            # Process each organ for this sample
            for organ_pred in sample_data['organs']:
                organ_name = organ_pred['organ_name']
                gt_present = organ_pred['ground_truth_present']
                pred_present = organ_pred['predicted_present']
                
                sample_output['y_true'].append(gt_present)
                sample_output['y_pred'].append(pred_present)
                
                # Store bboxes
                if gt_present and organ_pred['ground_truth_bboxes']:
                    sample_output['bboxes_true'][organ_name] = organ_pred['ground_truth_bboxes']
                if pred_present and organ_pred['predicted_bboxes']:
                    sample_output['bboxes_pred'][organ_name] = organ_pred['predicted_bboxes']
                
                # Get both IoU types (use pre-computed if available)
                iou_bbox_to_bbox = 0.0
                iou_bbox_to_mask = 0.0
                
                if gt_present and pred_present:
                    if 'iou_bbox_to_bbox' in organ_pred:
                        # Use pre-computed values from new format
                        iou_bbox_to_bbox = organ_pred['iou_bbox_to_bbox']
                        iou_bbox_to_mask = organ_pred['iou_bbox_to_mask']
                    else:
                        # Fallback to legacy computation (bbox-to-bbox only)
                        iou_bbox_to_bbox = compute_best_iou(
                            organ_pred['predicted_bboxes'],
                            organ_pred['ground_truth_bboxes']
                        )
                    sample_output['ious'].append(iou_bbox_to_bbox)  # Legacy field
                
                # Add detailed organ info with dual IoU
                sample_output['organs'].append({
                    'organ_id': organ_pred['organ_id'],
                    'organ_name': organ_name,
                    'ground_truth_present': gt_present,
                    'predicted_present': pred_present,
                    'ground_truth_bboxes': organ_pred['ground_truth_bboxes'],
                    'predicted_bboxes': organ_pred['predicted_bboxes'],
                    # Legacy field for backward compatibility
                    'iou': iou_bbox_to_bbox if (gt_present and pred_present) else None,
                    # New dual IoU fields
                    'iou_bbox_to_bbox': iou_bbox_to_bbox if (gt_present and pred_present) else None,
                    'iou_bbox_to_mask': iou_bbox_to_mask if (gt_present and pred_present) else None
                })
            
            # Save individual file
            sample_file = model_dir / f"test_{test_idx:05d}.json"
            with open(sample_file, 'w') as f:
                json.dump(sample_output, f, indent=2)
        
        # Save aggregated predictions (for backward compatibility)
        with open(model_dir / "predictions.json", 'w') as f:
            json.dump(predictions, f, indent=2)
        
        # Save metrics
        with open(model_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"    ✓ Saved {len(predictions_by_idx)} individual predictions to {model_dir}")
        print(f"    ✓ Saved aggregated metrics")
    
    def load_and_compute_metrics(self, results_dir: Path) -> Dict:
        """Load individual prediction files and compute metrics.
        
        Args:
            results_dir: Directory containing test_*.json files
            
        Returns:
            Dictionary with computed metrics
        """
        # Find all individual prediction files
        prediction_files = sorted(results_dir.glob("test_*.json"))
        
        if not prediction_files:
            print(f"No prediction files found in {results_dir}")
            return {}
        
        # Load all predictions
        all_predictions = []
        for pred_file in prediction_files:
            with open(pred_file, 'r') as f:
                sample_data = json.load(f)
                
                # Convert to flat prediction format
                for organ_data in sample_data['organs']:
                    all_predictions.append({
                        'test_idx': sample_data['sample_idx'],
                        'organ_id': organ_data['organ_id'],
                        'organ_name': organ_data['organ_name'],
                        'predicted_present': organ_data['predicted_present'],
                        'predicted_bboxes': organ_data['predicted_bboxes'],
                        'ground_truth_present': organ_data['ground_truth_present'],
                        'ground_truth_bboxes': organ_data['ground_truth_bboxes']
                    })
        
        # Compute metrics from loaded predictions
        metrics = self._compute_metrics(all_predictions)
        
        # Save updated metrics
        metrics_file = results_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"    ✓ Computed metrics from {len(prediction_files)} files")
        print(f"    ✓ Saved metrics to {metrics_file}")
        
        return metrics