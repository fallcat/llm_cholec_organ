#!/bin/bash
# Test script to verify GoNoGoNet IoU fix

echo "Testing GoNoGoNet IoU fix..."
echo "============================"

cd /shared_data0/weiqiuy/llm_cholec_organ/notebooks_py

# Run with just 2 samples to use cached results
EVAL_MODEL=gonogonet \
EVAL_DATASET=cholec_gonogo \
EVAL_NUM_SAMPLES=2 \
EVAL_DETECTION_MODE=combined \
EVAL_FEWSHOT=0 \
python3 eval_bbox_unified.py

echo ""
echo "Checking metrics file..."
echo "========================"
python3 -c "
import json
with open('/shared_data0/weiqiuy/llm_cholec_organ/results/bbox_cholec_gonogo_quick/zeroshot_combined/gonogonet/metrics.json', 'r') as f:
    metrics = json.load(f)
print(f'Presence Accuracy: {metrics[\"presence_accuracy\"]:.3f}')
print(f'Bbox-to-Bbox IoU: {metrics[\"mean_iou_bbox_to_bbox\"]:.3f}')
print(f'Bbox-to-Mask IoU: {metrics[\"mean_iou_bbox_to_mask\"]:.3f}')
print()
print('Per-organ IoU values:')
for organ, data in metrics.get('per_organ', {}).items():
    print(f'  {organ}:')
    print(f'    Bbox-to-Bbox: {data[\"mean_iou_bbox_to_bbox\"]:.3f}')
    print(f'    Bbox-to-Mask: {data[\"mean_iou_bbox_to_mask\"]:.3f}')
"