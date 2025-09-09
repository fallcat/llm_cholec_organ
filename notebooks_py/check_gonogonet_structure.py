#!/usr/bin/env python3
"""Check GoNoGoNet file structure to debug IoU aggregation issue."""

import json
from pathlib import Path
import numpy as np

def check_gonogonet_structure():
    """Check the structure of GoNoGoNet result files."""
    
    # Check a few GoNoGoNet files
    gonogo_dir = Path('/shared_data0/weiqiuy/llm_cholec_organ/results/bbox_cholec_gonogo_quick/zeroshot_combined/gonogonet')
    files = sorted(gonogo_dir.glob('test_*.json'))[:10]  # Check first 10 files
    
    print('=' * 80)
    print('CHECKING GONOGONET FILE STRUCTURE')
    print('=' * 80)
    
    all_ious_bbox = []
    all_ious_mask = []
    files_with_organs = 0
    files_without_organs = 0
    
    for f in files:
        with open(f) as fp:
            data = json.load(fp)
        
        print(f'\nFile: {f.name}')
        print(f'  Keys in file: {list(data.keys())}')
        
        if 'organs' in data:
            files_with_organs += 1
            print(f'  ✓ Has "organs" key with {len(data["organs"])} organs')
            
            for org in data['organs']:
                organ_id = org.get('organ_id', 'N/A')
                organ_name = org.get('organ_name', 'N/A')
                
                # Check IoU keys
                iou_simple = org.get('iou')
                iou_bbox = org.get('iou_bbox_to_bbox')
                iou_mask = org.get('iou_bbox_to_mask')
                
                print(f'    Organ {organ_id} ({organ_name}):')
                print(f'      - iou: {iou_simple}')
                print(f'      - iou_bbox_to_bbox: {iou_bbox}')
                print(f'      - iou_bbox_to_mask: {iou_mask}')
                
                # Collect IoU values
                if iou_simple is not None:
                    all_ious_bbox.append(iou_simple)
                elif iou_bbox is not None:
                    all_ious_bbox.append(iou_bbox)
                    
                if iou_mask is not None:
                    all_ious_mask.append(iou_mask)
        else:
            files_without_organs += 1
            print(f'  ✗ NO "organs" key!')
    
    print('\n' + '=' * 80)
    print('SUMMARY')
    print('=' * 80)
    print(f'Files with "organs" key: {files_with_organs}/{len(files)}')
    print(f'Files without "organs" key: {files_without_organs}/{len(files)}')
    
    if all_ious_bbox:
        print(f'\nBbox-to-bbox IoU values collected: {len(all_ious_bbox)}')
        print(f'  Mean: {np.mean(all_ious_bbox):.3f}')
        print(f'  Min: {np.min(all_ious_bbox):.3f}')
        print(f'  Max: {np.max(all_ious_bbox):.3f}')
    else:
        print('\n⚠️ No bbox-to-bbox IoU values found!')
    
    if all_ious_mask:
        print(f'\nBbox-to-mask IoU values collected: {len(all_ious_mask)}')
        print(f'  Mean: {np.mean(all_ious_mask):.3f}')
        print(f'  Min: {np.min(all_ious_mask):.3f}')
        print(f'  Max: {np.max(all_ious_mask):.3f}')
    else:
        print('\n⚠️ No bbox-to-mask IoU values found!')
    
    # Now check what the notebook aggregation would find
    print('\n' + '=' * 80)
    print('TESTING NOTEBOOK AGGREGATION LOGIC')
    print('=' * 80)
    
    # Simulate the notebook's aggregation logic
    notebook_ious_bbox = []
    notebook_ious_mask = []
    
    for f in files:
        with open(f) as fp:
            data = json.load(fp)
        
        for organ in data.get('organs', []):
            # This is exactly what the notebook does
            if 'iou_bbox_to_mask' in organ and organ['iou_bbox_to_mask'] is not None:
                notebook_ious_mask.append(organ['iou_bbox_to_mask'])
            
            # Check both possible key names for bbox IoU
            if 'iou' in organ and organ['iou'] is not None:
                notebook_ious_bbox.append(organ['iou'])
            elif 'iou_bbox_to_bbox' in organ and organ['iou_bbox_to_bbox'] is not None:
                notebook_ious_bbox.append(organ['iou_bbox_to_bbox'])
    
    print(f'Notebook would find:')
    print(f'  Bbox-to-bbox IoU values: {len(notebook_ious_bbox)}')
    if notebook_ious_bbox:
        print(f'    Mean: {np.mean(notebook_ious_bbox):.3f}')
    print(f'  Bbox-to-mask IoU values: {len(notebook_ious_mask)}')
    if notebook_ious_mask:
        print(f'    Mean: {np.mean(notebook_ious_mask):.3f}')
    
    # Check all files to get complete statistics
    print('\n' + '=' * 80)
    print('CHECKING ALL GONOGONET FILES')
    print('=' * 80)
    
    all_files = list(gonogo_dir.glob('test_*.json'))
    total_bbox_ious = []
    total_mask_ious = []
    
    for f in all_files:
        with open(f) as fp:
            data = json.load(fp)
        
        for organ in data.get('organs', []):
            if 'iou' in organ and organ['iou'] is not None:
                total_bbox_ious.append(organ['iou'])
            elif 'iou_bbox_to_bbox' in organ and organ['iou_bbox_to_bbox'] is not None:
                total_bbox_ious.append(organ['iou_bbox_to_bbox'])
                
            if 'iou_bbox_to_mask' in organ and organ['iou_bbox_to_mask'] is not None:
                total_mask_ious.append(organ['iou_bbox_to_mask'])
    
    print(f'Total files: {len(all_files)}')
    print(f'Total bbox-to-bbox IoU values: {len(total_bbox_ious)}')
    if total_bbox_ious:
        print(f'  Mean IoU-B: {np.mean(total_bbox_ious):.3f}')
        print(f'  Std IoU-B: {np.std(total_bbox_ious):.3f}')
    
    print(f'Total bbox-to-mask IoU values: {len(total_mask_ious)}')
    if total_mask_ious:
        print(f'  Mean IoU-M: {np.mean(total_mask_ious):.3f}')
        print(f'  Std IoU-M: {np.std(total_mask_ious):.3f}')

if __name__ == '__main__':
    check_gonogonet_structure()