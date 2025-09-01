#!/usr/bin/env python3
"""
Diagnose why few-shot and zero-shot are giving the same results.
"""

import os
import sys
import json
from pathlib import Path

# Add src to path
ROOT_DIR = ".."
sys.path.append(os.path.join(ROOT_DIR, 'src'))

# Load API keys
api_keys_file = Path(ROOT_DIR) / "API_KEYS2.json"
if api_keys_file.exists():
    with open(api_keys_file, "r") as file:
        api_keys = json.load(file)
    
    os.environ['OPENAI_API_KEY'] = api_keys.get('OPENAI_API_KEY', '')
    os.environ['ANTHROPIC_API_KEY'] = api_keys.get('ANTHROPIC_API_KEY', '')
    os.environ['GOOGLE_API_KEY'] = api_keys.get('GOOGLE_API_KEY', '')
else:
    print("❌ API_KEYS2.json not found")
    sys.exit(1)


def test_basic_fewshot():
    """Test that few-shot actually sends different prompts than zero-shot."""
    
    print("="*60)
    print("TESTING BASIC FEW-SHOT VS ZERO-SHOT")
    print("="*60)
    
    from endopoint.models import OpenAIAdapter
    from endopoint.eval.pointing import run_pointing_on_canvas
    import torch
    from PIL import Image
    import numpy as np
    
    # Create a dummy image
    dummy_img = torch.zeros(3, 256, 256)
    dummy_lab = torch.zeros(256, 256)
    
    # Create model without cache
    model = OpenAIAdapter(
        model_name="gpt-4o-mini",
        max_tokens=100,
        use_cache=False  # IMPORTANT: No cache for testing
    )
    
    # Monkey-patch the model to capture prompts instead of calling API
    captured_prompts = []
    
    original_call = model.__call__
    def capture_call(prompts, system_prompt=None):
        captured_prompts.append({
            "prompts": prompts,
            "system_prompt": system_prompt
        })
        return ["{'present': 1, 'point_canvas': [100, 100]}"]  # Dummy response
    
    model.__call__ = capture_call
    
    print("\n1. Testing ZERO-SHOT:")
    print("-" * 40)
    
    # Run zero-shot
    result_zero = run_pointing_on_canvas(
        model=model,
        img_t=dummy_img,
        lab_t=dummy_lab,
        organ_name="Liver",
        canvas_width=768,
        canvas_height=768,
        few_shot_examples=None  # No examples
    )
    
    zero_shot_prompt = captured_prompts[-1]["prompts"][0] if captured_prompts else None
    if zero_shot_prompt:
        if isinstance(zero_shot_prompt, tuple):
            print(f"   Prompt parts: {len(zero_shot_prompt)}")
            for i, part in enumerate(zero_shot_prompt):
                if isinstance(part, str):
                    print(f"   Part {i}: {part[:100]}..." if len(part) > 100 else f"   Part {i}: {part}")
                else:
                    print(f"   Part {i}: <image>")
    
    print("\n2. Testing FEW-SHOT:")
    print("-" * 40)
    
    # Create few-shot examples
    few_shot_examples = [
        (dummy_img, {"name": "Liver", "present": 1, "point_canvas": [400, 400]}),
        (dummy_img, {"name": "Liver", "present": 0, "point_canvas": None}),
    ]
    
    # Clear captured prompts
    captured_prompts.clear()
    
    # Run few-shot
    result_few = run_pointing_on_canvas(
        model=model,
        img_t=dummy_img,
        lab_t=dummy_lab,
        organ_name="Liver",
        canvas_width=768,
        canvas_height=768,
        few_shot_examples=few_shot_examples  # With examples
    )
    
    few_shot_prompt = captured_prompts[-1]["prompts"][0] if captured_prompts else None
    if few_shot_prompt:
        if isinstance(few_shot_prompt, tuple):
            print(f"   Prompt parts: {len(few_shot_prompt)}")
            for i, part in enumerate(few_shot_prompt):
                if isinstance(part, str):
                    print(f"   Part {i}: {part[:100]}..." if len(part) > 100 else f"   Part {i}: {part}")
                else:
                    print(f"   Part {i}: <image>")
    
    print("\n" + "="*60)
    print("RESULTS:")
    print("="*60)
    
    # Compare
    if zero_shot_prompt and few_shot_prompt:
        if isinstance(zero_shot_prompt, tuple) and isinstance(few_shot_prompt, tuple):
            print(f"Zero-shot parts: {len(zero_shot_prompt)}")
            print(f"Few-shot parts:  {len(few_shot_prompt)}")
            
            if len(zero_shot_prompt) == len(few_shot_prompt):
                print("❌ ERROR: Same number of parts! Few-shot examples not being added!")
            else:
                print("✅ Different number of parts - few-shot has examples")
        else:
            print("❌ ERROR: Prompts are not tuples")
    else:
        print("❌ ERROR: Could not capture prompts")


def check_evaluator_fewshot():
    """Check if the evaluator is actually creating few-shot examples."""
    
    print("\n" + "="*60)
    print("CHECKING EVALUATOR FEW-SHOT PREPARATION")
    print("="*60)
    
    from endopoint.datasets.cholecseg8k import CholecSeg8kAdapter
    from endopoint.eval.evaluator import PointingEvaluator
    from datasets import load_dataset
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("minwoosun/CholecSeg8k")
    
    # Create evaluator
    evaluator = PointingEvaluator(
        models=["gpt-4o-mini"],
        dataset=dataset,
        dataset_adapter=CholecSeg8kAdapter(),
        use_cache=False  # No cache
    )
    
    # Load few-shot plan
    data_dir = Path(ROOT_DIR) / "data_info" / "cholecseg8k"
    plan_file = data_dir / "fewshot_plan_train_pos1_neg1_seed43_excl100.json"
    
    if not plan_file.exists():
        print(f"❌ Few-shot plan not found: {plan_file}")
        return
    
    with open(plan_file) as f:
        fewshot_plan = json.load(f)
    
    print(f"✅ Loaded few-shot plan from {plan_file.name}")
    
    # Prepare few-shot examples
    few_shot_examples = evaluator.prepare_few_shot_examples(fewshot_plan, "train")
    
    print(f"\nFew-shot examples prepared:")
    for organ_name in ["Liver", "Gallbladder", "Fat"]:
        if organ_name in few_shot_examples:
            examples = few_shot_examples[organ_name]
            print(f"  {organ_name}: {len(examples)} examples")
            for i, (img_t, response) in enumerate(examples[:2]):  # First 2
                print(f"    Example {i+1}: present={response['present']}, point={response.get('point_canvas')}")
        else:
            print(f"  {organ_name}: NO EXAMPLES")
    
    if not few_shot_examples or all(len(v) == 0 for v in few_shot_examples.values()):
        print("\n❌ ERROR: No few-shot examples were prepared!")
        print("   This means the evaluator won't pass any examples to the model!")
    else:
        print("\n✅ Few-shot examples are being prepared")


def trace_actual_evaluation():
    """Trace an actual evaluation call to see what happens."""
    
    print("\n" + "="*60)
    print("TRACING ACTUAL EVALUATION CALL")
    print("="*60)
    
    # Monkey-patch the pointing function to trace calls
    import endopoint.eval.pointing as pointing_module
    
    original_run = pointing_module.run_pointing_on_canvas
    call_count = {"zero": 0, "few": 0}
    
    def traced_run(*args, **kwargs):
        few_shot_examples = kwargs.get("few_shot_examples")
        if few_shot_examples:
            call_count["few"] += 1
            print(f"  [TRACE] Few-shot call #{call_count['few']}: {len(few_shot_examples)} examples")
        else:
            call_count["zero"] += 1
            print(f"  [TRACE] Zero-shot call #{call_count['zero']}")
        
        return original_run(*args, **kwargs)
    
    pointing_module.run_pointing_on_canvas = traced_run
    
    # Now run a minimal evaluation
    from endopoint.datasets.cholecseg8k import CholecSeg8kAdapter
    from endopoint.eval.evaluator import PointingEvaluator
    from datasets import load_dataset
    
    print("Running minimal evaluation with tracing...")
    
    dataset = load_dataset("minwoosun/CholecSeg8k")
    
    evaluator = PointingEvaluator(
        models=["gpt-4o-mini"],
        dataset=dataset,
        dataset_adapter=CholecSeg8kAdapter(),
        use_cache=False
    )
    
    # Test indices (just 1 sample)
    test_indices = [0]
    
    # Load few-shot plan
    data_dir = Path(ROOT_DIR) / "data_info" / "cholecseg8k"
    plan_file = data_dir / "fewshot_plan_train_pos1_neg1_seed43_excl100.json"
    
    if plan_file.exists():
        with open(plan_file) as f:
            fewshot_plan = json.load(f)
        
        print("\n1. Running ZERO-SHOT:")
        evaluator.run_zero_shot("gpt-4o-mini", test_indices, "train")
        
        print(f"\n2. Running FEW-SHOT:")
        evaluator.run_few_shot("gpt-4o-mini", test_indices, fewshot_plan, "standard", "train")
        
        print("\n" + "="*60)
        print("TRACE SUMMARY:")
        print("="*60)
        print(f"Zero-shot calls: {call_count['zero']}")
        print(f"Few-shot calls:  {call_count['few']}")
        
        if call_count["few"] == 0:
            print("\n❌ ERROR: No few-shot calls were made!")
            print("   The evaluator is not using few-shot examples!")
        elif call_count["zero"] > 0 and call_count["few"] == 0:
            print("\n❌ ERROR: Few-shot evaluation ran as zero-shot!")
    else:
        print(f"❌ Few-shot plan not found")
    
    # Restore original function
    pointing_module.run_pointing_on_canvas = original_run


def main():
    print("\n" + "="*60)
    print("DIAGNOSING FEW-SHOT ISSUE")
    print("="*60)
    
    # Test 1: Basic few-shot
    test_basic_fewshot()
    
    # Test 2: Check evaluator preparation
    check_evaluator_fewshot()
    
    # Test 3: Trace actual evaluation
    trace_actual_evaluation()
    
    print("\n" + "="*60)
    print("DIAGNOSIS COMPLETE")
    print("="*60)
    print("""
If all tests pass but results are still the same:
1. The model might not be responding differently to few-shot prompts
2. Try with a different model (e.g., Claude instead of GPT)
3. Check the actual API responses in the cache files

To manually inspect cache entries:
  python3 -c "import diskcache; c = diskcache.Cache('../src/.llms.py.cache'); print(list(c.items())[:5])"
""")


if __name__ == "__main__":
    main()