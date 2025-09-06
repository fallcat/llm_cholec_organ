#!/usr/bin/env python3
"""Test script for Llama Vision model with CholecSeg8k dataset."""

import sys
sys.path.append('src')

from PIL import Image
from endopoint.models import LlamaAdapter
from endopoint.datasets import build_dataset
from endopoint.prompts import PROMPT_REGISTRY


def test_llama_with_single_image():
    """Test Llama model with a single CholecSeg8k image."""
    print("=" * 60)
    print("Testing Llama with single CholecSeg8k image")
    print("=" * 60)
    
    # Initialize model
    print("\nLoading Llama model...")
    model = LlamaAdapter(
        model_name="nvidia/Llama-3.2-11B-Vision-Surgical-CholecT50",
        temperature=0.0,
        max_tokens=256,
        use_cache=False,  # Disable cache for testing
        verbose=True
    )
    
    # Load a sample image from CholecSeg8k
    print("\nLoading CholecSeg8k dataset...")
    dataset = build_dataset("cholecseg8k")
    
    # Get first example from training set
    example = dataset.get_example("train", 0)
    image = example['image']  # PIL Image
    
    # Get presence information
    img_t, lab_t = dataset.example_to_tensors(example)
    presence_vector = dataset.labels_to_presence_vector(lab_t, min_pixels=100)
    
    print(f"\nImage info:")
    print(f"  - Shape: {image.size}")
    organs_present = [dataset.id2label[i+1] for i, p in enumerate(presence_vector) if p == 1]
    print(f"  - Organs present: {organs_present}")
    
    # Test basic organ detection
    print("\n" + "=" * 40)
    print("Test 1: Basic organ detection")
    print("=" * 40)
    
    prompt = "List all surgical instruments and anatomical structures visible in this laparoscopic image."
    query = (image, prompt)
    
    responses = model([query], system_prompt="You are an expert surgeon analyzing laparoscopic cholecystectomy images.")
    print(f"\nResponse:\n{responses[0]}")
    
    # Test specific organ query
    print("\n" + "=" * 40)
    print("Test 2: Specific organ query (Gallbladder)")
    print("=" * 40)
    
    prompt = "Is the gallbladder visible in this image? If yes, describe its appearance and location."
    query = (image, prompt)
    
    responses = model([query], system_prompt="")
    print(f"\nResponse:\n{responses[0]}")
    
    # Test with prompt from registry
    print("\n" + "=" * 40)
    print("Test 3: Using prompt from registry")
    print("=" * 40)
    
    # Get a prompt config from registry
    prompt_config = PROMPT_REGISTRY.get("strict")
    if prompt_config:
        organ = "Liver"
        
        # Get image dimensions for canvas
        width, height = image.size
        
        # Build the prompts using the builders from config
        system_prompt = prompt_config["system_builder"](width, height)
        user_prompt = prompt_config["user_builder"](organ)
        
        # Create query with image and user prompt
        query = (image, user_prompt)
        
        # Query model with system prompt
        responses = model([query], system_prompt=system_prompt)
        print(f"\nOrgan: {organ}")
        print(f"Canvas size: {width}x{height}")
        print(f"User prompt: {user_prompt}")
        print(f"Response:\n{responses[0]}")
    else:
        print("No 'strict' prompt found in registry")


def test_llama_with_multiple_images():
    """Test Llama model with multiple images."""
    print("\n" + "=" * 60)
    print("Testing Llama with multiple images")
    print("=" * 60)
    
    # Initialize model
    print("\nLoading Llama model...")
    model = LlamaAdapter(
        model_name="nvidia/Llama-3.2-11B-Vision-Surgical-CholecT50",
        temperature=0.0,
        max_tokens=256,
        use_cache=False,
        verbose=True
    )
    
    # Load dataset
    print("\nLoading CholecSeg8k dataset...")
    dataset = build_dataset("cholecseg8k")
    
    # Get two different images from training set
    example1 = dataset.get_example("train", 0)
    example2 = dataset.get_example("train", 1)
    image1 = example1['image']
    image2 = example2['image']
    
    print(f"\nUsing images:")
    print(f"  - Image 1: index 0")
    print(f"  - Image 2: index 1")
    
    # Test with multiple images interleaved with text
    print("\n" + "=" * 40)
    print("Test 4: Multiple images with interleaved text")
    print("=" * 40)
    
    query = (
        "First image:",
        image1,
        "Second image:",
        image2,
        "Compare the visibility of the gallbladder in these two images. Which image shows it more clearly?"
    )
    
    responses = model([query], system_prompt="You are analyzing laparoscopic surgery images.")
    print(f"\nResponse:\n{responses[0]}")
    
    # Test few-shot learning scenario
    print("\n" + "=" * 40)
    print("Test 5: Few-shot learning with examples")
    print("=" * 40)
    
    example3 = dataset.get_example("train", 2)
    image3 = example3['image']
    
    query = (
        "Here are examples of images with gallbladder visible:",
        image1,
        "This image shows the gallbladder clearly.",
        image2,
        "This image also contains the gallbladder.",
        "Now analyze this new image:",
        image3,
        "Is the gallbladder visible in this new image?"
    )
    
    responses = model([query], system_prompt="")
    print(f"\nResponse:\n{responses[0]}")


def test_batch_processing():
    """Test batch processing of multiple queries."""
    print("\n" + "=" * 60)
    print("Testing batch processing")
    print("=" * 60)
    
    # Initialize model
    print("\nLoading Llama model...")
    model = LlamaAdapter(
        model_name="nvidia/Llama-3.2-11B-Vision-Surgical-CholecT50",
        temperature=0.0,
        max_tokens=128,
        use_cache=True,  # Enable cache for batch
        verbose=False
    )
    
    # Load dataset
    print("\nLoading CholecSeg8k dataset...")
    dataset = build_dataset("cholecseg8k")
    
    # Prepare batch of queries
    queries = []
    organs_to_check = ["Liver", "Gallbladder", "Hepatocystic Triangle"]
    
    for i in range(3):
        example = dataset.get_example("train", i)
        image = example['image']
        organ = organs_to_check[i % len(organs_to_check)]
        query = (
            image,
            f"Is the {organ.lower()} visible in this surgical image? Answer with just 'yes' or 'no'."
        )
        queries.append(query)
    
    print(f"\nProcessing batch of {len(queries)} queries...")
    
    # Process batch
    responses = model(queries, system_prompt="Answer concisely.")
    
    # Display results
    for i, response in enumerate(responses):
        organ = organs_to_check[i % len(organs_to_check)]
        print(f"\nQuery {i+1} - {organ}:")
        print(f"  Response: {response}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test Llama Vision model with CholecSeg8k")
    parser.add_argument("--test", choices=["single", "multiple", "batch", "all"], 
                        default="single", help="Which test to run")
    args = parser.parse_args()
    
    if args.test == "single" or args.test == "all":
        test_llama_with_single_image()
    
    if args.test == "multiple" or args.test == "all":
        test_llama_with_multiple_images()
    
    if args.test == "batch" or args.test == "all":
        test_batch_processing()
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)