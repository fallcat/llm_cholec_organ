#!/usr/bin/env python3
"""
Debug script to investigate Cystic Duct ground truth distribution
"""

import json
from pathlib import Path

def check_cystic_duct_distribution():
    """Check Cystic Duct ground truth distribution across models"""
    
    results_dir = Path('/shared_data0/weiqiuy/llm_cholec_organ/results/bbox_cholecseg8k_local_quick/zeroshot_combined')
    
    models_to_check = ['gpt-4.1', 'gonogonet', 'claude-sonnet-4-20250514', 'qwen_qwen2.5-vl-7b-instruct']
    
    for model_name in models_to_check:
        model_dir = results_dir / model_name
        if not model_dir.exists():
            print(f"Model directory not found: {model_dir}")
            continue
            
        print(f'\n=== {model_name} ===')
        
        total_files = 0
        cystic_duct_positives = 0
        cystic_duct_negatives = 0
        cystic_duct_pred_positives = 0
        cystic_duct_pred_negatives = 0
        
        sample_positive_files = []
        sample_negative_files = []
        
        for test_file in model_dir.glob('test_*.json'):
            total_files += 1
            try:
                with open(test_file) as f:
                    data = json.load(f)
                
                for organ in data.get('organs', []):
                    if organ.get('organ_name') == 'Cystic Duct':
                        gt_present = organ.get('ground_truth_present', 0)
                        pred_present = organ.get('predicted_present', 0)
                        
                        if gt_present == 1:
                            cystic_duct_positives += 1
                            if len(sample_positive_files) < 3:
                                sample_positive_files.append(str(test_file))
                        else:
                            cystic_duct_negatives += 1
                            if len(sample_negative_files) < 3:
                                sample_negative_files.append(str(test_file))
                        
                        if pred_present == 1:
                            cystic_duct_pred_positives += 1
                        else:
                            cystic_duct_pred_negatives += 1
                        break
            except Exception as e:
                print(f"Error reading {test_file}: {e}")
                continue
        
        print(f'Total files: {total_files}')
        print(f'Cystic Duct - GT positives: {cystic_duct_positives}, GT negatives: {cystic_duct_negatives}')
        if cystic_duct_positives + cystic_duct_negatives > 0:
            print(f'GT positive rate: {cystic_duct_positives/(cystic_duct_positives+cystic_duct_negatives):.3f}')
        
        print(f'Cystic Duct - Pred positives: {cystic_duct_pred_positives}, Pred negatives: {cystic_duct_pred_negatives}')
        if cystic_duct_pred_positives + cystic_duct_pred_negatives > 0:
            print(f'Pred positive rate: {cystic_duct_pred_positives/(cystic_duct_pred_positives+cystic_duct_pred_negatives):.3f}')
        
        if sample_positive_files:
            print(f'Sample files with GT positive Cystic Duct: {sample_positive_files}')
        else:
            print('No files found with GT positive Cystic Duct')
            
        if sample_negative_files:
            print(f'Sample files with GT negative Cystic Duct: {sample_negative_files[:2]}')

def analyze_gonogonet_confusion():
    """Analyze why GoNoGoNet shows all FP for Cystic Duct"""
    
    print(f'\n{"="*60}')
    print("GONOGONET CYSTIC DUCT CONFUSION ANALYSIS")
    print(f'{"="*60}')
    
    model_dir = Path('/shared_data0/weiqiuy/llm_cholec_organ/results/bbox_cholecseg8k_local_quick/zeroshot_combined/gonogonet')
    
    if not model_dir.exists():
        print(f"GoNoGoNet directory not found: {model_dir}")
        return
    
    tp = tn = fp = fn = 0
    
    sample_files = list(model_dir.glob('test_*.json'))[:10]  # Check first 10 files
    
    for test_file in sample_files:
        try:
            with open(test_file) as f:
                data = json.load(f)
            
            for organ in data.get('organs', []):
                if organ.get('organ_name') == 'Cystic Duct':
                    gt = organ.get('ground_truth_present', 0)
                    pred = organ.get('predicted_present', 0)
                    
                    if gt == 1 and pred == 1:
                        tp += 1
                    elif gt == 0 and pred == 0:
                        tn += 1
                    elif gt == 0 and pred == 1:
                        fp += 1
                    elif gt == 1 and pred == 0:
                        fn += 1
                    
                    print(f"{test_file.name}: GT={gt}, Pred={pred}")
                    break
        except Exception as e:
            print(f"Error reading {test_file}: {e}")
            continue
    
    print(f'\nConfusion Matrix for first {len(sample_files)} files:')
    print(f'TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}')
    print(f'This explains why we see all FP - GoNoGoNet predicts Cystic Duct when it\'s not present')

if __name__ == "__main__":
    check_cystic_duct_distribution()
    analyze_gonogonet_confusion()