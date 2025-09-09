#!/usr/bin/env python3
"""Test the BBoxPrediction parsing with the actual LLaVA response."""

import sys
sys.path.insert(0, '/shared_data0/weiqiuy/llm_cholec_organ/src')

from endopoint.eval.bbox_evaluator import BBoxPrediction

# Actual response from LLaVA
response = """
{
  "Organ Name": {
    "present": true/false,
    "bbox": [x1, y1, x2, y2]  // only if present is true
  },
  "Go": {
    "present": true,
    "bbox": [0.000,0.000,0.500,0.986]
  },
  "NoGo": {
    "present": true,
    "bbox": [0.450,0.000,0.998,0.986]
  }
}
<|im_end|>
"""

# Test parsing for both organs
organs = [
    ("Go (Safe to Incise)", 1),
    ("NoGo (Unsafe to Incise)", 2)
]

canvas_width = 640
canvas_height = 384

for organ_name, organ_id in organs:
    print(f"\nParsing for: {organ_name}")
    pred = BBoxPrediction.from_json(response, organ_name, canvas_width, canvas_height)
    print(f"  Present: {pred.present}")
    print(f"  Bboxes: {pred.bboxes}")
    
    if pred.present and pred.bboxes:
        bbox = pred.bboxes[0]
        print(f"  Converted bbox: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
        print(f"  Expected Go bbox: [0, 0, 320, 379]")
        print(f"  Expected NoGo bbox: [288, 0, 639, 379]")