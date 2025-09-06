#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple demo of PeskaVLP for organ detection on a single CholecSeg8k image.
"""

import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Add SurgVLP to path
sys.path.insert(0, '/shared_data0/weiqiuy/github/SurgVLP')
import surgvlp
from mmengine.config import Config

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


def demo_peskavlp():
    """Simple demo of PeskaVLP organ detection."""
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load PeskaVLP model
    print("Loading PeskaVLP model...")
    configs = Config.fromfile('/shared_data0/weiqiuy/github/SurgVLP/tests/config_peskavlp.py')['config']
    model, preprocess = surgvlp.load(
        configs.model_config, 
        device=device,
        pretrain='/shared_data0/weiqiuy/github/SurgVLP/model_weights/PeskaVLP.pth'
    )
    model.eval()
    
    # Create text prompts for each organ
    print("\nCreating text prompts for organs...")
    text_prompts = []
    for organ in ORGAN_CLASSES:
        # Using mixed style prompts
        if organ == 'Abdominal Wall':
            prompt = "the abdominal wall tissue in laparoscopic surgery"
        elif organ == 'Liver':
            prompt = "the liver organ visible in the surgical field"
        elif organ == 'Gastrointestinal Tract':
            prompt = "the gastrointestinal tract"
        elif organ == 'Fat':
            prompt = "adipose tissue or fat in the surgical view"
        elif organ == 'Grasper':
            prompt = "a surgical grasper instrument"
        elif organ == 'Connective Tissue':
            prompt = "connective tissue structures"
        elif organ == 'Blood':
            prompt = "blood or bleeding in the surgical field"
        elif organ == 'Cystic Artery':
            prompt = "the cystic artery"
        elif organ == 'Clip':
            prompt = "a surgical clip or clipping instrument"
        elif organ == 'Gallbladder':
            prompt = "the gallbladder organ"
        elif organ == 'Hepatocystic Triangle':
            prompt = "the hepatocystic triangle anatomical region"
        elif organ == 'Liver Ligament':
            prompt = "liver ligament structures"
        else:
            prompt = f"the {organ.lower()} in surgery"
        
        text_prompts.append(prompt)
        print(f"  {organ}: '{prompt}'")
    
    # Encode text prompts
    print("\nEncoding text prompts...")
    text_inputs = surgvlp.tokenize(text_prompts)
    
    with torch.no_grad():
        feats_text_local, feats_text_global, sents = model.extract_feat_text(
            ids=text_inputs['input_ids'].to(device),
            attn_mask=text_inputs['attention_mask'].to(device),
            token_type=text_inputs['token_type_ids'].to(device)
        )
    
    text_features = feats_text_global
    text_features /= text_features.norm(dim=-1, keepdim=True)
    print(f"Text features shape: {text_features.shape}")
    
    # Load a sample image from CholecSeg8k
    # You can replace this with any image path
    sample_image_path = "data/cholecseg8k/train/images/video01_00040.png"
    
    print(f"\nLoading image: {sample_image_path}")
    try:
        image = Image.open(sample_image_path).convert('RGB')
    except:
        print(f"Could not load {sample_image_path}")
        print("Using a dummy image for demonstration...")
        # Create a dummy surgical-like image
        image = Image.new('RGB', (640, 480), color=(120, 30, 30))
    
    # Preprocess and encode image
    print("Encoding image...")
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        feats_img_local, feats_img_global = model.extract_feat_img(image_tensor)
    
    image_features = feats_img_global
    image_features /= image_features.norm(dim=-1, keepdim=True)
    print(f"Image features shape: {image_features.shape}")
    
    # Compute similarity scores
    print("\nComputing similarity scores...")
    similarity_scores = (image_features @ text_features.T).squeeze(0)
    similarity_scores = similarity_scores.cpu().numpy()
    
    # Apply threshold for detection
    threshold = 0.25  # You can adjust this
    predictions = (similarity_scores > threshold).astype(int)
    
    # Display results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Threshold: {threshold}")
    print("\nOrgan Detection Results:")
    print("-" * 40)
    
    for i, organ in enumerate(ORGAN_CLASSES):
        score = similarity_scores[i]
        detected = "PRESENT" if predictions[i] else "absent"
        status_symbol = "✓" if predictions[i] else " "
        print(f"{status_symbol} {organ:25s}: {score:.3f} [{detected}]")
    
    print("\nOrgans detected as present:")
    detected_organs = [ORGAN_CLASSES[i] for i in range(len(ORGAN_CLASSES)) if predictions[i]]
    if detected_organs:
        for organ in detected_organs:
            print(f"  - {organ}")
    else:
        print("  None (all organs below threshold)")
    
    # Visualize results
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Show image
    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[0].axis('off')
    
    # Show similarity scores
    colors = ['green' if p else 'gray' for p in predictions]
    bars = axes[1].barh(range(len(ORGAN_CLASSES)), similarity_scores, color=colors)
    axes[1].set_yticks(range(len(ORGAN_CLASSES)))
    axes[1].set_yticklabels(ORGAN_CLASSES)
    axes[1].axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold={threshold}')
    axes[1].set_xlabel('Similarity Score')
    axes[1].set_title('PeskaVLP Organ Detection Scores')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, similarity_scores)):
        axes[1].text(score + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{score:.3f}', va='center')
    
    plt.tight_layout()
    plt.savefig('peskavlp_demo_results.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: peskavlp_demo_results.png")
    plt.show()
    
    # Print similarity matrix between text prompts (interesting analysis)
    print("\n" + "="*60)
    print("BONUS: Text-Text Similarity Matrix (first 5 organs)")
    print("="*60)
    text_similarity = text_features[:5] @ text_features[:5].T
    text_similarity = text_similarity.cpu().numpy()
    
    print("\n     ", end="")
    for i in range(5):
        print(f"{ORGAN_CLASSES[i][:8]:>10s}", end="")
    print()
    
    for i in range(5):
        print(f"{ORGAN_CLASSES[i][:8]:8s}", end="")
        for j in range(5):
            print(f"{text_similarity[i, j]:10.3f}", end="")
        print()
    
    print("\nNote: Higher values indicate more similar text embeddings")
    print("Diagonal values should be 1.0 (self-similarity)")


if __name__ == "__main__":
    demo_peskavlp()