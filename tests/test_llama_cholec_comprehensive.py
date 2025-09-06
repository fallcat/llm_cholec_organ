#!/usr/bin/env python3
"""
Comprehensive test script for Llama Vision model with CholecSeg8k dataset.
Includes zero-shot and few-shot pointing tasks matching the evaluation pipeline.
"""

import sys
import json
sys.path.append('src')

import torch
import numpy as np
from PIL import Image
from endopoint.models import LlamaAdapter
from endopoint.datasets import build_dataset
from endopoint.prompts.builders import (
    build_pointing_system_prompt,
    build_pointing_system_prompt_strict,
    build_pointing_user_prompt,
    build_existence_system_prompt,
    build_existence_user_prompt,
)

# CholecSeg8k organ classes (12 organs/tools, excluding background and abdominal wall)
ORGAN_NAMES = [
    "Liver",
    "Gastrointestinal Tract",
    "Fat", 
    "Grasper",
    "Connective Tissue",
    "Blood",
    "Cystic Duct",
    "L-hook Electrocautery",
    "Gallbladder",
    "Hepatic Vein",
    "Liver Ligament",
]


def parse_pointing_response(response: str, canvas_width: int, canvas_height: int) -> dict:
    """Parse model response for pointing task.
    
    Args:
        response: Raw model response
        canvas_width: Canvas width
        canvas_height: Canvas height
        
    Returns:
        Parsed result dictionary
    """
    result = {
        "present": 0,
        "point_canvas": None,
        "raw": response,
        "parse_error": None
    }
    
    try:
        # Try to find JSON in the response
        import re
        json_match = re.search(r'\{[^}]+\}', response)
        if json_match:
            parsed = json.loads(json_match.group())
            
            # Extract presence
            if "present" in parsed:
                result["present"] = int(parsed["present"])
            
            # Extract point if present
            if result["present"] == 1 and "point_canvas" in parsed:
                point = parsed["point_canvas"]
                if point and isinstance(point, (list, tuple)) and len(point) == 2:
                    x, y = int(point[0]), int(point[1])
                    # Validate coordinates
                    if 0 <= x < canvas_width and 0 <= y < canvas_height:
                        result["point_canvas"] = (x, y)
    except Exception as e:
        result["parse_error"] = str(e)
    
    return result


def test_zero_shot_pointing():
    """Test zero-shot pointing task matching the evaluation pipeline."""
    print("\n" + "=" * 60)
    print("Zero-Shot Pointing Evaluation")
    print("=" * 60)
    print("This tests the model's ability to detect organs and point to them")
    print("without any examples (zero-shot setting).")
    print("=" * 60)
    
    # Initialize model
    print("\nLoading Llama model...")
    model = LlamaAdapter(
        model_name="nvidia/Llama-3.2-11B-Vision-Surgical-CholecT50",
        temperature=0.0,
        max_tokens=256,
        use_cache=False,
        verbose=False
    )
    
    # Load dataset
    print("Loading CholecSeg8k dataset...")
    dataset = build_dataset("cholecseg8k")
    
    # Get a sample image
    example = dataset.get_example("train", 10)  # Use index 10 for variety
    image = example['image']
    
    # Get ground truth
    img_t, lab_t = dataset.example_to_tensors(example)
    presence_vector = dataset.labels_to_presence_vector(lab_t, min_pixels=100)
    
    # Get canvas dimensions
    width, height = image.size
    
    print(f"\nImage dimensions: {width}x{height}")
    organs_present = [ORGAN_NAMES[i] for i, p in enumerate(presence_vector) if p == 1]
    print(f"Ground truth organs present: {organs_present}")
    
    # Test 1: Base pointing prompt
    print("\n" + "-" * 40)
    print("Test 1: Base Pointing Prompt")
    print("-" * 40)
    
    system_prompt = build_pointing_system_prompt(width, height)
    print(f"\nSystem prompt:\n{system_prompt}")
    
    # Test for each organ
    results = []
    for organ in ["Liver", "Gallbladder", "Grasper"]:
        user_prompt = build_pointing_user_prompt(organ)
        print(f"\nTesting organ: {organ}")
        print(f"User prompt: {user_prompt}")
        
        query = (image, user_prompt)
        response = model([query], system_prompt=system_prompt)[0]
        
        print(f"Response: {response}")
        
        # Parse response
        parsed = parse_pointing_response(response, width, height)
        results.append({
            "organ": organ,
            "present": parsed["present"],
            "point": parsed["point_canvas"],
            "ground_truth": organ in organs_present
        })
        
        if parsed["present"] == 1 and parsed["point_canvas"]:
            print(f"✓ Detected at point: {parsed['point_canvas']}")
        elif parsed["present"] == 0:
            print(f"✗ Not detected")
        else:
            print(f"⚠ Detection without valid point")
    
    # Test 2: Strict pointing prompt
    print("\n" + "-" * 40)
    print("Test 2: Strict Pointing Prompt")
    print("-" * 40)
    
    system_prompt_strict = build_pointing_system_prompt_strict(width, height)
    print(f"\nStrict system prompt (excerpt):\n{system_prompt_strict[:300]}...")
    
    organ = "Fat"
    user_prompt = build_pointing_user_prompt(organ)
    print(f"\nTesting organ: {organ}")
    
    query = (image, user_prompt)
    response = model([query], system_prompt=system_prompt_strict)[0]
    
    print(f"Response: {response}")
    parsed = parse_pointing_response(response, width, height)
    
    if parsed["present"] == 1 and parsed["point_canvas"]:
        print(f"✓ Detected at point: {parsed['point_canvas']}")
    elif parsed["present"] == 0:
        print(f"✗ Not detected")
    
    # Summary
    print("\n" + "-" * 40)
    print("Zero-Shot Results Summary")
    print("-" * 40)
    for r in results:
        match = "✓" if (r["present"] == 1) == r["ground_truth"] else "✗"
        print(f"{r['organ']:20} Present: {r['present']} (GT: {int(r['ground_truth'])}) {match}")


def test_few_shot_pointing():
    """Test few-shot pointing with examples."""
    print("\n" + "=" * 60)
    print("Few-Shot Pointing Evaluation")
    print("=" * 60)
    print("This tests the model with few-shot examples showing how to")
    print("detect and point to organs in similar images.")
    print("=" * 60)
    
    # Initialize model
    print("\nLoading Llama model...")
    model = LlamaAdapter(
        model_name="nvidia/Llama-3.2-11B-Vision-Surgical-CholecT50",
        temperature=0.0,
        max_tokens=256,
        use_cache=False,
        verbose=False
    )
    
    # Load dataset
    print("Loading CholecSeg8k dataset...")
    dataset = build_dataset("cholecseg8k")
    
    # Prepare few-shot examples
    print("\nPreparing few-shot examples...")
    
    # Positive example (organ present)
    pos_example = dataset.get_example("train", 0)
    pos_image = pos_example['image']
    pos_img_t, pos_lab_t = dataset.example_to_tensors(pos_example)
    
    # Negative example (organ absent) - find one without gallbladder
    neg_idx = 5
    neg_example = dataset.get_example("train", neg_idx)
    neg_image = neg_example['image']
    neg_img_t, neg_lab_t = dataset.example_to_tensors(neg_example)
    
    # Test image
    test_example = dataset.get_example("train", 20)
    test_image = test_example['image']
    test_img_t, test_lab_t = dataset.example_to_tensors(test_example)
    test_presence = dataset.labels_to_presence_vector(test_lab_t, min_pixels=100)
    
    width, height = test_image.size
    
    # Build few-shot prompt
    print("\n" + "-" * 40)
    print("Few-Shot Test: Gallbladder Detection")
    print("-" * 40)
    
    system_prompt = build_pointing_system_prompt_strict(width, height)
    organ = "Gallbladder"
    
    # Check if gallbladder is present in examples
    pos_presence = dataset.labels_to_presence_vector(pos_lab_t, min_pixels=100)
    neg_presence = dataset.labels_to_presence_vector(neg_lab_t, min_pixels=100)
    
    gallbladder_idx = ORGAN_NAMES.index("Gallbladder")
    pos_has_gb = pos_presence[gallbladder_idx] == 1
    neg_has_gb = neg_presence[gallbladder_idx] == 1
    test_has_gb = test_presence[gallbladder_idx] == 1
    
    print(f"Positive example has gallbladder: {pos_has_gb}")
    print(f"Negative example has gallbladder: {neg_has_gb}")
    print(f"Test image has gallbladder: {test_has_gb}")
    
    # Build the few-shot prompt with examples
    prompt_parts = [
        "Here are some examples of how to detect and point to organs:\n",
        "\nExample 1: Organ: \"Gallbladder\". Return exactly: {\"name\":\"Gallbladder\", \"present\":0|1, \"point_canvas\":[x,y] or null}",
        pos_image,
        f'\nResponse: {{"name":"Gallbladder", "present":{int(pos_has_gb)}, "point_canvas":{"[400,300]" if pos_has_gb else "null"}}}\n',
        "\nExample 2: Organ: \"Gallbladder\". Return exactly: {\"name\":\"Gallbladder\", \"present\":0|1, \"point_canvas\":[x,y] or null}",
        neg_image,
        f'\nResponse: {{"name":"Gallbladder", "present":{int(neg_has_gb)}, "point_canvas":{"[350,250]" if neg_has_gb else "null"}}}\n',
        "\nNow for the actual query: Organ: \"Gallbladder\". Return exactly: {\"name\":\"Gallbladder\", \"present\":0|1, \"point_canvas\":[x,y] or null}",
        test_image
    ]
    
    print("\nSending few-shot query with 2 examples...")
    response = model([tuple(prompt_parts)], system_prompt=system_prompt)[0]
    
    print(f"\nResponse: {response}")
    
    # Parse response
    parsed = parse_pointing_response(response, width, height)
    
    match = "✓" if (parsed["present"] == 1) == test_has_gb else "✗"
    print(f"\nResult: Present={parsed['present']} (GT: {int(test_has_gb)}) {match}")
    if parsed["point_canvas"]:
        print(f"Point: {parsed['point_canvas']}")


def test_existence_detection():
    """Test simple existence detection (yes/no) for all organs."""
    print("\n" + "=" * 60)
    print("Existence Detection Test (All 12 Organs)")
    print("=" * 60)
    print("This tests simple yes/no detection for each organ class")
    print("=" * 60)
    
    # Initialize model
    print("\nLoading Llama model...")
    model = LlamaAdapter(
        model_name="nvidia/Llama-3.2-11B-Vision-Surgical-CholecT50",
        temperature=0.0,
        max_tokens=50,  # Shorter for yes/no
        use_cache=False,
        verbose=False
    )
    
    # Load dataset
    print("Loading CholecSeg8k dataset...")
    dataset = build_dataset("cholecseg8k")
    
    # Get a sample with multiple organs
    example = dataset.get_example("train", 15)
    image = example['image']
    
    # Get ground truth
    img_t, lab_t = dataset.example_to_tensors(example)
    presence_vector = dataset.labels_to_presence_vector(lab_t, min_pixels=100)
    
    print(f"\nImage shape: {image.size}")
    
    # Test existence detection
    system_prompt = build_existence_system_prompt()
    print(f"\nSystem prompt:\n{system_prompt}")
    
    print("\n" + "-" * 40)
    print("Testing all 12 organ classes:")
    print("-" * 40)
    
    results = []
    correct = 0
    total = 0
    
    for i, organ in enumerate(ORGAN_NAMES):
        user_prompt = build_existence_user_prompt(organ)
        
        query = (image, user_prompt)
        response = model([query], system_prompt=system_prompt)[0]
        
        # Parse yes/no response
        response_lower = response.lower().strip()
        predicted = 1 if "yes" in response_lower else 0
        ground_truth = int(presence_vector[i])
        
        is_correct = predicted == ground_truth
        if is_correct:
            correct += 1
        total += 1
        
        match = "✓" if is_correct else "✗"
        print(f"{organ:25} Predicted: {'Yes' if predicted else 'No ':3} | GT: {'Yes' if ground_truth else 'No ':3} {match}")
        
        results.append({
            "organ": organ,
            "predicted": predicted,
            "ground_truth": ground_truth,
            "correct": is_correct,
            "response": response
        })
    
    # Summary
    accuracy = correct / total * 100
    print("\n" + "-" * 40)
    print(f"Existence Detection Accuracy: {correct}/{total} = {accuracy:.1f}%")
    print("-" * 40)


def test_comprehensive_evaluation():
    """Run a comprehensive evaluation on multiple images."""
    print("\n" + "=" * 60)
    print("Comprehensive Multi-Image Evaluation")
    print("=" * 60)
    
    # Initialize model
    print("\nLoading Llama model...")
    model = LlamaAdapter(
        model_name="nvidia/Llama-3.2-11B-Vision-Surgical-CholecT50",
        temperature=0.0,
        max_tokens=128,
        use_cache=True,
        verbose=False
    )
    
    # Load dataset
    print("Loading CholecSeg8k dataset...")
    dataset = build_dataset("cholecseg8k")
    
    # Test on multiple images
    test_indices = [0, 5, 10, 15, 20]
    organs_to_test = ["Liver", "Gallbladder", "Grasper", "Fat"]
    
    print(f"\nTesting {len(organs_to_test)} organs on {len(test_indices)} images")
    print("-" * 40)
    
    # Build prompts
    system_prompt = build_pointing_system_prompt_strict(768, 768)  # Use standard canvas size
    
    all_results = []
    
    for idx in test_indices:
        example = dataset.get_example("train", idx)
        image = example['image']
        
        # Get ground truth
        img_t, lab_t = dataset.example_to_tensors(example)
        presence_vector = dataset.labels_to_presence_vector(lab_t, min_pixels=100)
        
        print(f"\nImage {idx}:")
        
        for organ in organs_to_test:
            user_prompt = build_pointing_user_prompt(organ)
            
            query = (image, user_prompt)
            response = model([query], system_prompt=system_prompt)[0]
            
            # Parse response
            parsed = parse_pointing_response(response, 768, 768)
            
            # Get ground truth for this organ
            if organ in ORGAN_NAMES:
                organ_idx = ORGAN_NAMES.index(organ)
                gt_present = int(presence_vector[organ_idx])
            else:
                gt_present = 0
            
            is_correct = (parsed["present"] == gt_present)
            match = "✓" if is_correct else "✗"
            
            print(f"  {organ:15} Pred: {parsed['present']} | GT: {gt_present} {match}")
            
            all_results.append({
                "image_idx": idx,
                "organ": organ,
                "predicted": parsed["present"],
                "ground_truth": gt_present,
                "correct": is_correct
            })
    
    # Calculate overall accuracy
    correct = sum(1 for r in all_results if r["correct"])
    total = len(all_results)
    accuracy = correct / total * 100
    
    print("\n" + "=" * 40)
    print(f"Overall Accuracy: {correct}/{total} = {accuracy:.1f}%")
    print("=" * 40)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Comprehensive Llama Vision model tests")
    parser.add_argument("--test", choices=["zero-shot", "few-shot", "existence", "comprehensive", "all"],
                        default="all", help="Which test to run")
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("Llama Vision Model - CholecSeg8k Comprehensive Tests")
    print("=" * 60)
    print("Model: nvidia/Llama-3.2-11B-Vision-Surgical-CholecT50")
    print("Dataset: CholecSeg8k (laparoscopic cholecystectomy)")
    print("=" * 60)
    
    if args.test == "zero-shot" or args.test == "all":
        test_zero_shot_pointing()
    
    if args.test == "few-shot" or args.test == "all":
        test_few_shot_pointing()
    
    if args.test == "existence" or args.test == "all":
        test_existence_detection()
    
    if args.test == "comprehensive" or args.test == "all":
        test_comprehensive_evaluation()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)