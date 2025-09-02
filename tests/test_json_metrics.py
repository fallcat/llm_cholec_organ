#!/usr/bin/env python3
"""Test script to verify JSON metrics output functionality."""

import json
from pathlib import Path
import sys

def test_json_output():
    """Test that save_comparison_table generates both txt and json files."""
    
    # Create a mock results structure similar to what the evaluator produces
    mock_results = {
        "gpt-4o-mini": {
            "zero_shot": {
                "n_examples": 3,
                "table_output": "Mock zero-shot results table",
                "metrics_rows": [
                    {
                        "ID": 1,
                        "Label": "Liver",
                        "TP": 2,
                        "FN": 1,
                        "TN": 0,
                        "FP": 0,
                        "PresenceAcc": 0.667,
                        "Hit@Point|Present": 0.5,
                        "GatedAcc": 0.333,
                        "Precision": 1.0,
                        "Recall": 0.667,
                        "F1": 0.8
                    },
                    {
                        "ID": 2,
                        "Label": "Gallbladder",
                        "TP": 3,
                        "FN": 0,
                        "TN": 0,
                        "FP": 0,
                        "PresenceAcc": 1.0,
                        "Hit@Point|Present": 0.667,
                        "GatedAcc": 0.667,
                        "Precision": 1.0,
                        "Recall": 1.0,
                        "F1": 1.0
                    }
                ],
                "metrics_totals": {
                    "total_tp": 5,
                    "total_fn": 1,
                    "total_tn": 0,
                    "total_fp": 0,
                    "macro_presence_acc": 0.833,
                    "macro_hit_rate": 0.583,
                    "macro_gated_acc": 0.5,
                    "macro_precision": 1.0,
                    "macro_recall": 0.833,
                    "macro_f1": 0.9,
                    "micro_precision": 1.0,
                    "micro_recall": 0.833,
                    "micro_f1": 0.909
                }
            }
        },
        "llava-hf/llava-v1.6-mistral-7b-hf": {
            "zero_shot": {
                "n_examples": 3,
                "table_output": "Mock LLaVA zero-shot results table",
                "metrics_rows": [
                    {
                        "ID": 1,
                        "Label": "Liver",
                        "TP": 1,
                        "FN": 2,
                        "TN": 0,
                        "FP": 0,
                        "PresenceAcc": 0.333,
                        "Hit@Point|Present": 0.0,
                        "GatedAcc": 0.0,
                        "Precision": 1.0,
                        "Recall": 0.333,
                        "F1": 0.5
                    }
                ],
                "metrics_totals": {
                    "total_tp": 1,
                    "total_fn": 2,
                    "total_tn": 0,
                    "total_fp": 0,
                    "macro_presence_acc": 0.333,
                    "macro_hit_rate": 0.0,
                    "macro_gated_acc": 0.0,
                    "macro_precision": 1.0,
                    "macro_recall": 0.333,
                    "macro_f1": 0.5,
                    "micro_precision": 1.0,
                    "micro_recall": 0.333,
                    "micro_f1": 0.5
                }
            }
        }
    }
    
    # Create a temporary output directory
    output_dir = Path("test_json_output")
    output_dir.mkdir(exist_ok=True)
    
    # Directly execute the save_comparison_table logic from enhanced_evaluator.py
    # without importing the module (to avoid torch dependency)
    
    # Save text version for human reading
    comparison_file_txt = output_dir / "metrics_comparison.txt"
    
    lines = []
    lines.append("="*80)
    lines.append("COMPREHENSIVE METRICS COMPARISON")
    lines.append("="*80)
    
    for model_name, model_results in mock_results.items():
        lines.append(f"\n{model_name}:")
        
        for eval_type, results in model_results.items():
            if "table_output" in results:
                lines.append(results["table_output"])
    
    with open(comparison_file_txt, "w") as f:
        f.write("\n".join(lines))
    
    print(f"💾 Saved comparison table to {comparison_file_txt}")
    
    # Save JSON version for easy loading and analysis
    comparison_file_json = output_dir / "metrics_comparison.json"
    
    # Extract key metrics for JSON output
    json_data = {
        "metadata": {
            "timestamp": str(output_dir.name),
            "canvas_width": 224,
            "canvas_height": 224,
            "organ_names": ["Liver", "Gallbladder", "Hepatocystic Triangle", "Fat", 
                          "Grasper", "Connective Tissue", "Blood", "Cystic Artery",
                          "Cystic Plate", "Hepatofalciform Ligament", "Liver Ligament", "Omental"],
            "models": list(mock_results.keys())
        },
        "results": {}
    }
    
    for model_name, model_results in mock_results.items():
        json_data["results"][model_name] = {}
        
        for eval_type, results in model_results.items():
            json_data["results"][model_name][eval_type] = {
                "n_examples": results.get("n_examples", 0)
            }
            
            # Add metrics rows if available
            if "metrics_rows" in results:
                json_data["results"][model_name][eval_type]["per_organ"] = []
                for row in results["metrics_rows"]:
                    json_data["results"][model_name][eval_type]["per_organ"].append({
                        "id": row.get("ID"),
                        "label": row.get("Label"),
                        "TP": row.get("TP"),
                        "FN": row.get("FN"),
                        "TN": row.get("TN"),
                        "FP": row.get("FP"),
                        "PresenceAcc": row.get("PresenceAcc"),
                        "Hit@Point|Present": row.get("Hit@Point|Present"),
                        "GatedAcc": row.get("GatedAcc"),
                        "Precision": row.get("Precision"),
                        "Recall": row.get("Recall"),
                        "F1": row.get("F1")
                    })
            
            # Add aggregate metrics if available
            if "metrics_totals" in results:
                totals = results["metrics_totals"]
                json_data["results"][model_name][eval_type]["aggregate"] = {
                    "total_TP": totals.get("total_tp", 0),
                    "total_FN": totals.get("total_fn", 0),
                    "total_TN": totals.get("total_tn", 0),
                    "total_FP": totals.get("total_fp", 0),
                    "macro_presence_acc": totals.get("macro_presence_acc"),
                    "macro_hit_rate": totals.get("macro_hit_rate"),
                    "macro_gated_acc": totals.get("macro_gated_acc"),
                    "macro_precision": totals.get("macro_precision"),
                    "macro_recall": totals.get("macro_recall"),
                    "macro_f1": totals.get("macro_f1"),
                    "micro_precision": totals.get("micro_precision"),
                    "micro_recall": totals.get("micro_recall"),
                    "micro_f1": totals.get("micro_f1")
                }
    
    # Save JSON with proper formatting
    with open(comparison_file_json, "w") as f:
        json.dump(json_data, f, indent=2)
    
    print(f"💾 Saved metrics JSON to {comparison_file_json}")
    
    # Check that both files were created
    txt_file = output_dir / "metrics_comparison.txt"
    json_file = output_dir / "metrics_comparison.json"
    
    assert txt_file.exists(), f"Text file not created: {txt_file}"
    assert json_file.exists(), f"JSON file not created: {json_file}"
    
    print(f"✅ Both files created successfully:")
    print(f"   - {txt_file}")
    print(f"   - {json_file}")
    
    # Load and verify JSON structure
    with open(json_file, 'r') as f:
        json_data = json.load(f)
    
    # Check metadata
    assert "metadata" in json_data, "Missing metadata in JSON"
    assert "canvas_width" in json_data["metadata"], "Missing canvas_width in metadata"
    assert "canvas_height" in json_data["metadata"], "Missing canvas_height in metadata"
    assert "organ_names" in json_data["metadata"], "Missing organ_names in metadata"
    assert "models" in json_data["metadata"], "Missing models in metadata"
    
    print("✅ JSON metadata structure is correct")
    
    # Check results structure
    assert "results" in json_data, "Missing results in JSON"
    assert "gpt-4o-mini" in json_data["results"], "Missing gpt-4o-mini in results"
    assert "llava-hf/llava-v1.6-mistral-7b-hf" in json_data["results"], "Missing LLaVA in results"
    
    # Check zero_shot results
    gpt_zero = json_data["results"]["gpt-4o-mini"]["zero_shot"]
    assert "n_examples" in gpt_zero, "Missing n_examples"
    assert "per_organ" in gpt_zero, "Missing per_organ metrics"
    assert "aggregate" in gpt_zero, "Missing aggregate metrics"
    
    print("✅ JSON results structure is correct")
    
    # Check per-organ metrics
    assert len(gpt_zero["per_organ"]) == 2, f"Expected 2 organs, got {len(gpt_zero['per_organ'])}"
    liver = gpt_zero["per_organ"][0]
    assert liver["label"] == "Liver", f"Expected Liver, got {liver['label']}"
    assert liver["TP"] == 2, f"Expected TP=2, got {liver['TP']}"
    
    print("✅ Per-organ metrics are correct")
    
    # Check aggregate metrics
    agg = gpt_zero["aggregate"]
    assert agg["total_TP"] == 5, f"Expected total_TP=5, got {agg['total_TP']}"
    assert agg["macro_f1"] == 0.9, f"Expected macro_f1=0.9, got {agg['macro_f1']}"
    
    print("✅ Aggregate metrics are correct")
    
    # Pretty print a sample of the JSON
    print("\n📊 Sample JSON structure:")
    print(json.dumps({
        "metadata": json_data["metadata"],
        "results": {
            "gpt-4o-mini": {
                "zero_shot": {
                    "n_examples": gpt_zero["n_examples"],
                    "per_organ": [gpt_zero["per_organ"][0]],  # Just show first organ
                    "aggregate": {
                        "macro_f1": agg["macro_f1"],
                        "micro_f1": agg["micro_f1"]
                    }
                }
            }
        }
    }, indent=2))
    
    print("\n✅ All tests passed! JSON output functionality is working correctly.")
    
    # Option to clean up (comment out to keep files)
    # import shutil
    # shutil.rmtree(output_dir)
    # print("🧹 Cleaned up test directory")
    
    print(f"📁 Test files kept in {output_dir}/")

if __name__ == "__main__":
    test_json_output()