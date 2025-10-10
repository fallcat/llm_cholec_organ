#!/usr/bin/env bash
set -uo pipefail  # Remove -e to continue on errors

# Helper script to run all model family evaluations in sequence

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "================================================"
echo "Running All Model Families - 5 Seeds Evaluation"
echo "================================================"
echo ""

# Track successes and failures
declare -a PASSED=()
declare -a FAILED=()

# Run API LVLMs
echo "Step 1/4: Running API LVLMs..."
if bash "${SCRIPT_DIR}/run_api_lvmls.sh"; then
    echo "✓ API LVLMs completed successfully"
    PASSED+=("API LVLMs")
else
    echo "✗ API LVLMs failed"
    FAILED+=("API LVLMs")
fi
echo ""

# Run Local LVLMs
echo "Step 2/4: Running Local LVLMs..."
if bash "${SCRIPT_DIR}/run_local_lvmls.sh"; then
    echo "✓ Local LVLMs completed successfully"
    PASSED+=("Local LVLMs")
else
    echo "✗ Local LVLMs failed"
    FAILED+=("Local LVLMs")
fi
echo ""

# Run CLIP Models
echo "Step 3/4: Running CLIP Models..."
if bash "${SCRIPT_DIR}/run_clip_models.sh"; then
    echo "✓ CLIP Models completed successfully"
    PASSED+=("CLIP Models")
else
    echo "✗ CLIP Models failed"
    FAILED+=("CLIP Models")
fi
echo ""

# Run Segmentation/Task-Specific Models
echo "Step 4/4: Running Segmentation/Task-Specific Models..."
if bash "${SCRIPT_DIR}/run_seg_models.sh"; then
    echo "✓ Segmentation Models completed successfully"
    PASSED+=("Segmentation Models")
else
    echo "✗ Segmentation Models failed"
    FAILED+=("Segmentation Models")
fi
echo ""

echo "================================================"
echo "All Model Families Evaluation Complete!"
echo "================================================"
echo ""

# Summary report
echo "Summary:"
echo "Passed: ${#PASSED[@]}/4"
if [ ${#PASSED[@]} -gt 0 ]; then
    echo "  ✓ ${PASSED[*]}"
fi

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo "Failed: ${#FAILED[@]}/4"
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
echo "  python notebooks_py/aggregate_bbox_results.py --num-samples 200"

# Exit with error if any family failed
if [ ${#FAILED[@]} -gt 0 ]; then
    exit 1
fi