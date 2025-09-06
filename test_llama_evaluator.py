#!/usr/bin/env python3
"""Quick test to verify Llama model can be loaded in evaluator."""

import sys
sys.path.append('src')

from endopoint.eval.evaluator import PointingEvaluator

# Create a dummy evaluator
evaluator = PointingEvaluator(
    models=[], 
    dataset=None, 
    dataset_adapter=None, 
    canvas_width=768, 
    canvas_height=768, 
    output_dir=None
)

# Test loading the Llama model
try:
    model = evaluator.load_model('nvidia/Llama-3.2-11B-Vision-Surgical-CholecT50', use_cache=False)
    print(f'✓ Successfully loaded Llama model: {model.__class__.__name__}')
    print(f'  Model name: {model.model_name}')
    print(f'  Max tokens: {model.max_tokens}')
    print(f'  Temperature: {model.temperature}')
except Exception as e:
    print(f'✗ Failed to load Llama model: {e}')
    import traceback
    traceback.print_exc()