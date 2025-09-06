#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test PeskaVLP for organ detection on CholecSeg8k dataset.

This script loads the CholecSeg8k dataset and uses PeskaVLP to detect
organ presence by computing similarity scores between image and text embeddings.
Organs with similarity scores above a threshold are considered present.
"""

import sys
import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from typing import List, Dict, Tuple
from pathlib import Path

# Add SurgVLP to path
sys.path.insert(0, '/shared_data0/weiqiuy/github/SurgVLP')

# Add current project to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import surgvlp
from mmengine.config import Config

# Import from our project
from src.endopoint.datasets.cholecseg8k import CholecSeg8kAdapter
from src.endopoint.utils.io import read_json, write_json


# CholecSeg8k organ classes
ORGAN_CLASSES = [
    'Abdominal Wall',
    'Liver', 
    'Gastrointestinal Tract',
    'Fat',
    'Grasper',
    'Connective Tissue',
    'Blood',
    'Cystic Artery',
    'Clip',
    'Gallbladder',
    'Hepatocystic Triangle',
    'Liver Ligament'
]


def load_peskavlp_model(device='cuda'):
    """Load PeskaVLP model and preprocessing function."""
    configs = Config.fromfile('/shared_data0/weiqiuy/github/SurgVLP/tests/config_peskavlp.py')['config']
    model, preprocess = surgvlp.load(
        configs.model_config, 
        device=device,
        pretrain='/shared_data0/weiqiuy/github/SurgVLP/model_weights/PeskaVLP.pth'
    )
    model.eval()
    return model, preprocess


def encode_text_prompts(model, texts: List[str], device='cuda'):
    """
    Encode text prompts using PeskaVLP.
    
    Args:
        model: PeskaVLP model
        texts: List of text prompts
        device: Device to run on
        
    Returns:
        Normalized text features tensor
    """
    text_inputs = surgvlp.tokenize(texts)
    
    with torch.no_grad():
        feats_text_local, feats_text_global, sents = model.extract_feat_text(
            ids=text_inputs['input_ids'].to(device),
            attn_mask=text_inputs['attention_mask'].to(device),
            token_type=text_inputs['token_type_ids'].to(device)
        )
    
    text_features = feats_text_global
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    return text_features


def encode_image(model, preprocess, image_path: str, device='cuda'):
    """
    Encode an image using PeskaVLP.
    
    Args:
        model: PeskaVLP model
        preprocess: Preprocessing function
        image_path: Path to image
        device: Device to run on
        
    Returns:
        Normalized image features tensor
    """
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Extract image features
        feats_img_local, feats_img_global = model.extract_feat_img(image_tensor)
    
    image_features = feats_img_global
    image_features /= image_features.norm(dim=-1, keepdim=True)
    
    return image_features


def compute_similarity_scores(image_features, text_features):
    """
    Compute cosine similarity between image and text features.
    
    Args:
        image_features: Image feature tensor (1, D)
        text_features: Text feature tensor (N, D) for N text prompts
        
    Returns:
        Similarity scores (N,)
    """
    # Compute cosine similarity
    similarity = image_features @ text_features.T
    return similarity.squeeze(0)


def create_organ_prompts(organ_classes: List[str], prompt_style='simple'):
    """
    Create text prompts for each organ class.
    
    Args:
        organ_classes: List of organ class names
        prompt_style: Style of prompt ('simple', 'descriptive', 'surgical')
        
    Returns:
        List of text prompts
    """
    prompts = []
    
    if prompt_style == 'simple':
        # Simple prompts
        for organ in organ_classes:
            prompts.append(f"an image of {organ.lower()}")
    
    elif prompt_style == 'descriptive':
        # More descriptive prompts
        for organ in organ_classes:
            prompts.append(f"a surgical view showing the {organ.lower()}")
    
    elif prompt_style == 'surgical':
        # Surgical context prompts
        for organ in organ_classes:
            prompts.append(f"the {organ.lower()} is visible in this laparoscopic surgery")
    
    elif prompt_style == 'mixed':
        # Mix of different prompt styles
        template_map = {
            'Abdominal Wall': "the abdominal wall tissue",
            'Liver': "the liver organ",
            'Gastrointestinal Tract': "the gastrointestinal tract",
            'Fat': "adipose tissue or fat",
            'Grasper': "a surgical grasper instrument",
            'Connective Tissue': "connective tissue structures",
            'Blood': "blood or bleeding",
            'Cystic Artery': "the cystic artery",
            'Clip': "a surgical clip",
            'Gallbladder': "the gallbladder",
            'Hepatocystic Triangle': "the hepatocystic triangle area",
            'Liver Ligament': "liver ligament structures"
        }
        for organ in organ_classes:
            prompts.append(template_map.get(organ, f"the {organ.lower()}"))
    
    return prompts


def evaluate_organ_detection(
    model, 
    preprocess,
    dataset_adapter,
    indices: List[int],
    threshold: float = 0.3,
    prompt_style: str = 'mixed',
    device: str = 'cuda',
    verbose: bool = True
):
    """
    Evaluate PeskaVLP for organ detection on CholecSeg8k.
    
    Args:
        model: PeskaVLP model
        preprocess: Preprocessing function
        dataset_adapter: CholecSeg8k dataset adapter
        indices: List of sample indices to evaluate
        threshold: Similarity threshold for detection
        prompt_style: Style of text prompts
        device: Device to run on
        verbose: Whether to print progress
        
    Returns:
        Results dictionary with predictions and metrics
    """
    # Create text prompts for organs
    organ_prompts = create_organ_prompts(ORGAN_CLASSES, prompt_style)
    
    # Encode text prompts
    if verbose:
        print(f"Encoding {len(organ_prompts)} text prompts...")
    text_features = encode_text_prompts(model, organ_prompts, device)
    
    # Store results
    results = {
        'predictions': [],
        'ground_truth': [],
        'similarity_scores': [],
        'indices': indices,
        'threshold': threshold,
        'prompt_style': prompt_style
    }
    
    # Process each image
    for idx in tqdm(indices, desc="Processing images", disable=not verbose):
        example = dataset_adapter[idx]
        
        # Get ground truth presence vector
        gt_presence = example['presence_vector']
        
        # Encode image
        image_features = encode_image(model, preprocess, example['image_path'], device)
        
        # Compute similarity scores
        similarity_scores = compute_similarity_scores(image_features, text_features)
        
        # Apply threshold for predictions
        predictions = (similarity_scores > threshold).cpu().numpy().astype(int)
        
        # Store results
        results['predictions'].append(predictions.tolist())
        results['ground_truth'].append(gt_presence.tolist())
        results['similarity_scores'].append(similarity_scores.cpu().numpy().tolist())
    
    # Compute metrics
    results['predictions'] = np.array(results['predictions'])
    results['ground_truth'] = np.array(results['ground_truth'])
    results['similarity_scores'] = np.array(results['similarity_scores'])
    
    # Calculate per-class and overall metrics
    metrics = calculate_metrics(results['predictions'], results['ground_truth'])
    results['metrics'] = metrics
    
    return results


def calculate_metrics(predictions, ground_truth):
    """
    Calculate detection metrics.
    
    Args:
        predictions: Binary predictions (N, C)
        ground_truth: Binary ground truth (N, C)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Per-class metrics
    per_class_metrics = []
    for i, organ in enumerate(ORGAN_CLASSES):
        pred_class = predictions[:, i]
        gt_class = ground_truth[:, i]
        
        # Calculate metrics
        tp = np.sum((pred_class == 1) & (gt_class == 1))
        fp = np.sum((pred_class == 1) & (gt_class == 0))
        fn = np.sum((pred_class == 0) & (gt_class == 1))
        tn = np.sum((pred_class == 0) & (gt_class == 0))
        
        accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        per_class_metrics.append({
            'organ': organ,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn)
        })
    
    metrics['per_class'] = per_class_metrics
    
    # Overall metrics
    overall_accuracy = np.mean([m['accuracy'] for m in per_class_metrics])
    overall_precision = np.mean([m['precision'] for m in per_class_metrics])
    overall_recall = np.mean([m['recall'] for m in per_class_metrics])
    overall_f1 = np.mean([m['f1'] for m in per_class_metrics])
    
    metrics['overall'] = {
        'accuracy': overall_accuracy,
        'precision': overall_precision,
        'recall': overall_recall,
        'f1': overall_f1
    }
    
    return metrics


def find_optimal_threshold(
    model,
    preprocess,
    dataset_adapter,
    indices: List[int],
    thresholds: np.ndarray = None,
    prompt_style: str = 'mixed',
    device: str = 'cuda',
    verbose: bool = True
):
    """
    Find optimal threshold for organ detection.
    
    Args:
        model: PeskaVLP model
        preprocess: Preprocessing function
        dataset_adapter: Dataset adapter
        indices: Sample indices for validation
        thresholds: Array of thresholds to test
        prompt_style: Style of text prompts
        device: Device to run on
        verbose: Whether to print progress
        
    Returns:
        Optimal threshold and results for each threshold
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.6, 0.05)
    
    # Create text prompts
    organ_prompts = create_organ_prompts(ORGAN_CLASSES, prompt_style)
    text_features = encode_text_prompts(model, organ_prompts, device)
    
    # Collect similarity scores and ground truth
    all_scores = []
    all_gt = []
    
    for idx in tqdm(indices, desc="Computing similarities", disable=not verbose):
        example = dataset_adapter[idx]
        gt_presence = example['presence_vector']
        
        image_features = encode_image(model, preprocess, example['image_path'], device)
        similarity_scores = compute_similarity_scores(image_features, text_features)
        
        all_scores.append(similarity_scores.cpu().numpy())
        all_gt.append(gt_presence)
    
    all_scores = np.array(all_scores)
    all_gt = np.array(all_gt)
    
    # Test different thresholds
    threshold_results = []
    
    for threshold in thresholds:
        predictions = (all_scores > threshold).astype(int)
        metrics = calculate_metrics(predictions, all_gt)
        
        threshold_results.append({
            'threshold': threshold,
            'metrics': metrics,
            'f1_score': metrics['overall']['f1']
        })
        
        if verbose:
            print(f"Threshold: {threshold:.2f}, F1: {metrics['overall']['f1']:.3f}")
    
    # Find optimal threshold based on F1 score
    best_idx = np.argmax([r['f1_score'] for r in threshold_results])
    optimal_threshold = threshold_results[best_idx]['threshold']
    
    if verbose:
        print(f"\nOptimal threshold: {optimal_threshold:.2f}")
        print(f"Best F1 score: {threshold_results[best_idx]['f1_score']:.3f}")
    
    return optimal_threshold, threshold_results


def visualize_results(results, dataset_adapter, num_samples=4):
    """
    Visualize detection results.
    
    Args:
        results: Results dictionary from evaluate_organ_detection
        dataset_adapter: Dataset adapter
        num_samples: Number of samples to visualize
    """
    indices = results['indices'][:num_samples]
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(indices):
        example = dataset_adapter[idx]
        
        # Load image
        image = Image.open(example['image_path'])
        
        # Plot image
        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f"Sample {idx}")
        axes[i, 0].axis('off')
        
        # Plot similarity scores
        scores = results['similarity_scores'][i]
        axes[i, 1].barh(range(len(ORGAN_CLASSES)), scores)
        axes[i, 1].set_yticks(range(len(ORGAN_CLASSES)))
        axes[i, 1].set_yticklabels(ORGAN_CLASSES)
        axes[i, 1].axvline(x=results['threshold'], color='r', linestyle='--', label=f"Threshold={results['threshold']:.2f}")
        axes[i, 1].set_xlabel('Similarity Score')
        axes[i, 1].set_title('PeskaVLP Scores')
        axes[i, 1].legend()
        
        # Plot predictions vs ground truth
        pred = results['predictions'][i]
        gt = results['ground_truth'][i]
        
        x = np.arange(len(ORGAN_CLASSES))
        width = 0.35
        
        axes[i, 2].bar(x - width/2, gt, width, label='Ground Truth', alpha=0.8)
        axes[i, 2].bar(x + width/2, pred, width, label='Prediction', alpha=0.8)
        axes[i, 2].set_xticks(x)
        axes[i, 2].set_xticklabels(ORGAN_CLASSES, rotation=45, ha='right')
        axes[i, 2].set_ylabel('Present (1) / Absent (0)')
        axes[i, 2].set_title('Predictions vs Ground Truth')
        axes[i, 2].legend()
        axes[i, 2].set_ylim([0, 1.2])
    
    plt.tight_layout()
    plt.show()


def main():
    """Main function to run PeskaVLP organ detection on CholecSeg8k."""
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load PeskaVLP model
    print("Loading PeskaVLP model...")
    model, preprocess = load_peskavlp_model(device)
    
    # Load CholecSeg8k dataset
    print("Loading CholecSeg8k dataset...")
    dataset_adapter = CholecSeg8kAdapter(
        split='train',
        download=False,  # Assuming data is already downloaded
        root_dir='./data/cholecseg8k'
    )
    
    # Load balanced indices if available
    indices_file = 'data_info/cholecseg8k/balanced_indices_train_100_cap70_seed7.json'
    if os.path.exists(indices_file):
        print(f"Loading balanced indices from {indices_file}")
        indices = read_json(indices_file)[:20]  # Use first 20 for demo
    else:
        print("Using first 20 samples")
        indices = list(range(20))
    
    # Test different prompt styles
    prompt_styles = ['simple', 'descriptive', 'surgical', 'mixed']
    
    for prompt_style in prompt_styles:
        print(f"\n{'='*60}")
        print(f"Testing prompt style: {prompt_style}")
        print('='*60)
        
        # Find optimal threshold on validation set
        val_indices = indices[:10]
        print(f"\nFinding optimal threshold on {len(val_indices)} validation samples...")
        optimal_threshold, threshold_results = find_optimal_threshold(
            model, preprocess, dataset_adapter, val_indices,
            thresholds=np.arange(0.15, 0.45, 0.05),
            prompt_style=prompt_style,
            device=device,
            verbose=False
        )
        
        # Evaluate on test set
        test_indices = indices[10:20]
        print(f"\nEvaluating on {len(test_indices)} test samples...")
        results = evaluate_organ_detection(
            model, preprocess, dataset_adapter, test_indices,
            threshold=optimal_threshold,
            prompt_style=prompt_style,
            device=device,
            verbose=False
        )
        
        # Print results
        print(f"\nResults for prompt style: {prompt_style}")
        print(f"Threshold: {optimal_threshold:.2f}")
        print(f"Overall Metrics:")
        print(f"  Accuracy: {results['metrics']['overall']['accuracy']:.3f}")
        print(f"  Precision: {results['metrics']['overall']['precision']:.3f}")
        print(f"  Recall: {results['metrics']['overall']['recall']:.3f}")
        print(f"  F1 Score: {results['metrics']['overall']['f1']:.3f}")
        
        # Print per-class F1 scores
        print("\nPer-class F1 Scores:")
        for class_metrics in results['metrics']['per_class']:
            print(f"  {class_metrics['organ']:25s}: {class_metrics['f1']:.3f}")
        
        # Save results
        output_dir = Path('results/peskavlp_baseline')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f'results_{prompt_style}.json'
        write_json(output_file, results)
        print(f"\nResults saved to {output_file}")
        
        # Visualize some results (only for first prompt style)
        if prompt_style == prompt_styles[0]:
            print("\nVisualizing results...")
            visualize_results(results, dataset_adapter, num_samples=3)
    
    print("\n" + "="*60)
    print("PeskaVLP evaluation complete!")
    print("="*60)


if __name__ == "__main__":
    main()