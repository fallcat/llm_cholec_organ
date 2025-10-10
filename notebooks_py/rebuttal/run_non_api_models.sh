#!/usr/bin/env bash
set -uo pipefail  # Continue on errors

# Helper script to run all non-API model families

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "================================================"
echo "Running Non-API Models - 3 Seeds Unbalanced"
echo "================================================"
echo ""

# Track successes and failures
declare -a PASSED=()
declare -a FAILED=()

# Run Local LVLMs
echo "Step 1/3: Running Local LVLMs..."
if bash "${SCRIPT_DIR}/run_local_lvmls.sh"; then
    echo "✓ Local LVLMs completed successfully"
    PASSED+=("Local LVLMs")
else
    echo "✗ Local LVLMs failed"
    FAILED+=("Local LVLMs")
fi
echo ""

# Run CLIP Models
echo "Step 2/3: Running CLIP Models..."
if bash "${SCRIPT_DIR}/run_clip_models.sh"; then
    echo "✓ CLIP Models completed successfully"
    PASSED+=("CLIP Models")
else
    echo "✗ CLIP Models failed"
    FAILED+=("CLIP Models")
fi
echo ""

# Run Segmentation Models
echo "Step 3/3: Running Segmentation Models..."
if bash "${SCRIPT_DIR}/run_seg_models.sh"; then
    echo "✓ Segmentation Models completed successfully"
    PASSED+=("Segmentation Models")
else
    echo "✗ Segmentation Models failed"
    FAILED+=("Segmentation Models")
fi
echo ""

echo "================================================"
echo "Non-API Models Evaluation Complete!"
echo "================================================"
echo ""

# Summary report
echo "Summary:"
echo "Passed: ${#PASSED[@]}/3"
if [ ${#PASSED[@]} -gt 0 ]; then
    echo "  ✓ ${PASSED[*]}"
fi

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo "Failed: ${#FAILED[@]}/3"
    echo "  ✗ ${FAILED[*]}"
    echo ""
    echo "Check logs for failed families:"
    echo "  /shared_data0/weiqiuy/llm_cholec_organ/results/logs/"
fi

echo ""
echo "Results location: /shared_data0/weiqiuy/llm_cholec_organ/results"
echo "Logs location: /shared_data0/weiqiuy/llm_cholec_organ/results/logs/"
echo ""
echo "To view aggregated results, run:"
echo "  python notebooks_py/aggregate_bbox_results.py --num-samples 200 --seeds \"42,7,2025\" --split-modes unbalanced"

# Exit with error if any family failed
if [ ${#FAILED[@]} -gt 0 ]; then
    exit 1
fi