#!/bin/bash

# Run naive baselines on CholecGoNoGo dataset

echo "Running Naive Baselines on CholecGoNoGo dataset..."
echo "================================================"

cd /shared_data0/weiqiuy/llm_cholec_organ/notebooks_py

# Run the Python script
python run_naive_baseline_cholec_gonogo.py

echo ""
echo "Done! Check the results in:"
echo "  /shared_data0/weiqiuy/llm_cholec_organ/results/bbox_cholec_gonogo_quick/zeroshot_combined/"