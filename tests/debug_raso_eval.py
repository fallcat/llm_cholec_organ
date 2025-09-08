#!/usr/bin/env python3
"""
Debug why RASO evaluation is showing 0% accuracy.
"""

import sys
sys.path.insert(0, '/shared_data0/weiqiuy/llm_cholec_organ/src')

import json
from endopoint.eval.bbox_evaluator import BBoxPrediction

# Simulate what happens during evaluation
def test_bbox_parsing():
    """Test how BBoxPrediction.from_json works with RASO response."""
    
    print("="*80)
    print("DEBUGGING BBOX PREDICTION PARSING")
    print("="*80)
    
    # This is what RASO returns (from our test)
    raso_response = """{
  "Liver": {
    "present": true,
    "bbox": null
  },
  "Gallbladder": {
    "present": true,
    "bbox": null
  },
  "Hepatocystic Triangle": {
    "present": false,
    "bbox": null
  }
}"""
    
    # These are the organ names the evaluation is looking for
    organ_names = ["Liver", "Gallbladder", "Hepatocystic Triangle"]
    
    print("\nRASO Response:")
    print(raso_response)
    
    print("\n" + "-"*40)
    print("Parsing each organ:")
    print("-"*40)
    
    for organ_name in organ_names:
        pred = BBoxPrediction.from_json(raso_response, organ_name)
        print(f"\nOrgan: {organ_name}")
        print(f"  Predicted present: {pred.present}")
        print(f"  Predicted bboxes: {pred.bboxes}")
    
    # Let's also check what happens with case mismatch
    print("\n" + "-"*40)
    print("Testing case sensitivity:")
    print("-"*40)
    
    # Try lowercase organ name with capitalized response
    pred_lower = BBoxPrediction.from_json(raso_response, "liver")
    print(f"\nSearching for 'liver' in response with 'Liver':")
    print(f"  Predicted present: {pred_lower.present}")
    
    # Try exact match
    pred_exact = BBoxPrediction.from_json(raso_response, "Liver")
    print(f"\nSearching for 'Liver' in response with 'Liver':")
    print(f"  Predicted present: {pred_exact.present}")
    
    # Debug: Let's see what the parser extracts
    print("\n" + "-"*40)
    print("Manual parsing test:")
    print("-"*40)
    
    import re
    json_match = re.search(r'\{.*\}', raso_response, re.DOTALL)
    if json_match:
        data = json.loads(json_match.group())
        print(f"Parsed JSON keys: {list(data.keys())}")
        
        for organ_name in ["Liver", "liver"]:
            print(f"\nChecking '{organ_name}' in data:")
            print(f"  Is '{organ_name}' in data? {organ_name in data}")
            if organ_name in data:
                organ_data = data[organ_name]
                print(f"  organ_data: {organ_data}")
                present = organ_data.get('present', False)
                print(f"  present value: {present}")
                print(f"  present as int: {1 if present else 0}")

if __name__ == "__main__":
    test_bbox_parsing()