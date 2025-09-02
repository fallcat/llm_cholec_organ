"""Pointing evaluation utilities."""

import json
from typing import Dict, List, Optional, Tuple, Union
import torch
import numpy as np
from PIL import Image

from ..models.base import ModelAdapter
from ..prompts.builders import (
    build_pointing_system_prompt,
    build_pointing_user_prompt,
    build_cell_selection_system_prompt,
    build_cell_selection_system_prompt_strict,
    build_cell_selection_user_prompt,
)
from .parser import parse_pointing_json, parse_cell_selection_json
from .cell_selection import compute_cell_ground_truth, compute_cell_metrics


def run_pointing_on_canvas(
    model: ModelAdapter,
    img_t: torch.Tensor,
    lab_t: torch.Tensor,
    organ_name: str,
    canvas_width: int = 768,
    canvas_height: int = 768,
    system_prompt_builder=None,
    user_prompt_builder=None,
    few_shot_examples: Optional[List[Tuple[torch.Tensor, Dict]]] = None,
) -> Dict:
    """Run pointing task on a single organ.
    
    Args:
        model: Model adapter instance
        img_t: Image tensor [3,H,W]
        lab_t: Label tensor [H,W]
        organ_name: Name of organ to detect
        canvas_width: Canvas width in pixels
        canvas_height: Canvas height in pixels
        system_prompt_builder: Function to build system prompt
        user_prompt_builder: Function to build user prompt
        few_shot_examples: Optional list of (image_tensor, response_dict) for few-shot
        
    Returns:
        Dictionary with:
            - organ: organ name
            - present: 0 or 1
            - point_canvas: (x, y) tuple or None
            - raw: Raw model response
    """
    # Use default builders if not provided
    if system_prompt_builder is None:
        system_prompt_builder = build_pointing_system_prompt
    if user_prompt_builder is None:
        user_prompt_builder = build_pointing_user_prompt
    
    # Build prompts
    system_prompt = system_prompt_builder(canvas_width, canvas_height)
    user_prompt = user_prompt_builder(organ_name)
    
    # Convert current image to PIL
    img_pil = tensor_to_pil(img_t)
    
    # Query model - use the adapter's __call__ method
    # Format for the batch API
    if few_shot_examples:
        # Build a single tuple with all few-shot examples and the current query
        # Format: (text1, image1, text2, image2, ..., current_text, current_image)
        prompt_parts = []
        
        # Add few-shot examples
        prompt_parts.append("Here are some examples:\n")
        
        for i, (ex_img, ex_response) in enumerate(few_shot_examples, 1):
            # Convert example image to PIL
            ex_pil = tensor_to_pil(ex_img)
            
            # Build the example prompt
            ex_prompt = f"\nExample {i}: {user_prompt_builder(ex_response['name'])}"
            prompt_parts.append(ex_prompt)
            prompt_parts.append(ex_pil)
            
            # Build the expected response
            point_val = ex_response.get("point_canvas")
            if point_val and isinstance(point_val, (list, tuple)):
                point_str = f"[{point_val[0]},{point_val[1]}]"
            else:
                point_str = "null"
            
            ex_response_text = (
                f"\nResponse: {{"
                f'"name":"{ex_response["name"]}",'
                f'"present":{ex_response["present"]},'
                f'"point_canvas":{point_str}'
                f"}}\n"
            )
            prompt_parts.append(ex_response_text)
        
        # Add the current query
        prompt_parts.append(f"\nNow for the actual query: {user_prompt}")
        prompt_parts.append(img_pil)
        
        # Convert to tuple and call model
        response = model([tuple(prompt_parts)], system_prompt=system_prompt)[0]
    else:
        response = model([(user_prompt, img_pil)], system_prompt=system_prompt)[0]
    
    # Parse response
    result = parse_pointing_json(response, canvas_width, canvas_height)
    result["organ"] = organ_name
    
    return result


def tensor_to_pil(img_t: torch.Tensor) -> Image.Image:
    """Convert tensor to PIL Image.
    
    Args:
        img_t: Tensor [3,H,W] or [H,W,3], values in [0,1]
        
    Returns:
        PIL Image
    """
    if isinstance(img_t, torch.Tensor):
        img_np = img_t.detach().cpu().numpy()
    else:
        img_np = img_t
    
    # Handle different shapes
    if img_np.ndim == 3:
        if img_np.shape[0] == 3:  # [3,H,W]
            img_np = np.transpose(img_np, (1, 2, 0))  # -> [H,W,3]
    
    # Ensure value range [0, 255]
    if img_np.dtype == np.float32 or img_np.dtype == np.float64:
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)
    
    return Image.fromarray(img_np, mode='RGB')


def pointing_pipeline(
    model: ModelAdapter,
    img_t: torch.Tensor,
    lab_t: torch.Tensor,
    organ_names: List[str],
    canvas_width: int = 768,
    canvas_height: int = 768,
    system_prompt_builder=None,
    user_prompt_builder=None,
    few_shot_examples_per_organ: Optional[Dict[str, List]] = None,
) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
    """Run pointing pipeline for multiple organs.
    
    Args:
        model: Model adapter instance
        img_t: Image tensor [3,H,W]
        lab_t: Label tensor [H,W]
        organ_names: List of organ names to detect
        canvas_width: Canvas width
        canvas_height: Canvas height
        system_prompt_builder: Optional custom system prompt builder
        user_prompt_builder: Optional custom user prompt builder
        few_shot_examples_per_organ: Dict mapping organ names to few-shot examples
        
    Returns:
        Tuple of:
            - List of result dictionaries
            - Predicted presence vector [N_organs]
            - Ground truth presence vector [N_organs]
    """
    results = []
    y_pred = []
    
    for organ_name in organ_names:
        # Get few-shot examples for this organ
        few_shot_examples = None
        if few_shot_examples_per_organ:
            few_shot_examples = few_shot_examples_per_organ.get(organ_name)
        
        # Run pointing for this organ
        result = run_pointing_on_canvas(
            model=model,
            img_t=img_t,
            lab_t=lab_t,
            organ_name=organ_name,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            system_prompt_builder=system_prompt_builder,
            user_prompt_builder=user_prompt_builder,
            few_shot_examples=few_shot_examples,
        )
        
        results.append(result)
        y_pred.append(result["present"])
    
    # Convert to numpy arrays
    y_pred = np.array(y_pred)
    
    # Get ground truth from labels (if adapter available)
    from ..datasets.cholecseg8k import CholecSeg8kAdapter
    adapter = CholecSeg8kAdapter()
    y_true = adapter.labels_to_presence_vector(lab_t).numpy()
    
    return results, y_pred, y_true


def calculate_pointing_metrics(
    results: List[Dict],
    y_true: np.ndarray,
    organ_names: List[str],
) -> Dict:
    """Calculate pointing evaluation metrics.
    
    Args:
        results: List of pointing results
        y_true: Ground truth presence vector
        organ_names: List of organ names
        
    Returns:
        Dictionary with metrics
    """
    # Extract predictions
    y_pred = np.array([r["present"] for r in results])
    
    # Per-organ metrics
    organ_metrics = {}
    for i, organ_name in enumerate(organ_names):
        tp = ((y_true[i] == 1) & (y_pred[i] == 1)).sum()
        fp = ((y_true[i] == 0) & (y_pred[i] == 1)).sum()
        fn = ((y_true[i] == 1) & (y_pred[i] == 0)).sum()
        tn = ((y_true[i] == 0) & (y_pred[i] == 0)).sum()
        
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Check pointing accuracy (only for true positives)
        pointing_correct = 0
        pointing_total = 0
        for j, r in enumerate(results):
            if r["organ"] == organ_name and r["present"] == 1 and y_true[i] == 1:
                pointing_total += 1
                if r.get("point_canvas") is not None:
                    # Here we could validate if point is actually inside the organ
                    # For now, just check if a point was provided
                    pointing_correct += 1
        
        pointing_acc = pointing_correct / pointing_total if pointing_total > 0 else None
        
        organ_metrics[organ_name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "pointing_accuracy": pointing_acc,
        }
    
    # Overall metrics
    overall_accuracy = (y_pred == y_true).mean()
    
    return {
        "overall_accuracy": overall_accuracy,
        "organ_metrics": organ_metrics,
    }


def run_cell_selection_on_canvas(
    model: ModelAdapter,
    img_t: torch.Tensor,
    lab_t: torch.Tensor,
    organ_name: str,
    grid_size: int = 3,
    top_k: int = 1,
    canvas_width: int = 768,
    canvas_height: int = 768,
    prompt_style: str = "standard",
    few_shot_examples: Optional[List[Tuple[torch.Tensor, Dict]]] = None,
    min_pixels: int = 50,
) -> Dict:
    """Run cell selection task on a single organ.
    
    Args:
        model: Model adapter instance
        img_t: Image tensor [3,H,W]
        lab_t: Label tensor [H,W]
        organ_name: Name of organ to detect
        grid_size: Grid size (3 or 4)
        top_k: Maximum number of cells to return
        canvas_width: Canvas width in pixels
        canvas_height: Canvas height in pixels
        prompt_style: "standard" or "strict"
        few_shot_examples: Optional list of (image_tensor, response_dict) for few-shot
        min_pixels: Minimum pixels for presence detection
        
    Returns:
        Dictionary with:
            - organ: organ name
            - present: 0 or 1
            - cells: List of cell labels
            - gt_cells: Ground truth cell labels
            - gt_present: Ground truth presence
            - metrics: Cell selection metrics
            - raw: Raw model response
    """
    # Build prompts
    if prompt_style == "strict":
        system_prompt = build_cell_selection_system_prompt_strict(
            canvas_width, canvas_height, grid_size, top_k
        )
    else:
        system_prompt = build_cell_selection_system_prompt(
            canvas_width, canvas_height, grid_size, top_k
        )
    
    user_prompt = build_cell_selection_user_prompt(organ_name, grid_size)
    
    # Convert current image to PIL
    img_pil = tensor_to_pil(img_t)
    
    # Query model - use the adapter's __call__ method with batch format
    # This follows the same pattern as run_pointing_on_canvas
    if few_shot_examples:
        # Build a single tuple with all few-shot examples and the current query
        # Format: (text1, image1, text2, image2, ..., current_text, current_image)
        prompt_parts = []
        
        # Add few-shot examples
        prompt_parts.append("Here are some examples:\n")
        
        for i, (ex_img, ex_response) in enumerate(few_shot_examples, 1):
            # Convert example image to PIL
            ex_pil = tensor_to_pil(ex_img)
            
            # Build the example prompt
            ex_prompt = f"\nExample {i}: {build_cell_selection_user_prompt(ex_response['name'], grid_size)}"
            prompt_parts.append(ex_prompt)
            prompt_parts.append(ex_pil)
            
            # Build the expected response
            response_json = {
                "name": ex_response['name'],
                "present": ex_response.get('present', 0),
                "cells": ex_response.get('cells', [])
            }
            ex_response_text = f"\nResponse: {json.dumps(response_json)}\n"
            prompt_parts.append(ex_response_text)
        
        # Add the current query
        prompt_parts.append(f"\nNow for the actual query: {user_prompt}")
        prompt_parts.append(img_pil)
        
        # Convert to tuple and call model with batch format
        raw_response = model([tuple(prompt_parts)], system_prompt=system_prompt)[0]
    else:
        # Zero-shot with batch format
        raw_response = model([(user_prompt, img_pil)], system_prompt=system_prompt)[0]
    
    # Parse response
    parsed = parse_cell_selection_json(raw_response, grid_size, top_k)
    
    # Get organ ID from name (assuming CholecSeg8k)
    from ..datasets.cholecseg8k import LABEL2ID
    organ_id = LABEL2ID.get(organ_name, 0)
    
    # Compute ground truth
    if organ_id > 0:
        organ_mask = (lab_t == organ_id).numpy().astype(np.uint8)
        gt_info = compute_cell_ground_truth(organ_mask, grid_size, min_pixels)
        
        # Compute metrics
        metrics = compute_cell_metrics(
            parsed['cells'],
            gt_info['cells'],
            gt_info['present'],
            parsed['present'],
            top_k
        )
    else:
        # Unknown organ - assume absent
        gt_info = {'present': False, 'cells': set(), 'dominant_cell': None}
        metrics = compute_cell_metrics(
            parsed['cells'],
            set(),
            False,
            parsed['present'],
            top_k
        )
    
    return {
        "organ": organ_name,
        "present": parsed["present"],
        "cells": parsed["cells"],
        "gt_present": gt_info["present"],
        "gt_cells": list(gt_info["cells"]),
        "gt_dominant_cell": gt_info.get("dominant_cell"),
        "metrics": metrics,
        "raw": raw_response,
        "grid_size": grid_size,
        "top_k": top_k,
    }