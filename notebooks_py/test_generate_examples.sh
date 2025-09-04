#!/bin/bash

# Test script to generate paper examples with proper Python environment

# Try to find the right Python
if command -v python &> /dev/null; then
    PYTHON_CMD=python
elif command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
else
    echo "Python not found!"
    exit 1
fi

echo "Using Python: $PYTHON_CMD"

# Check for required packages
echo "Checking dependencies..."
$PYTHON_CMD -c "import numpy; print('✓ numpy')" 2>/dev/null || { echo "✗ numpy missing"; }
$PYTHON_CMD -c "import matplotlib; print('✓ matplotlib')" 2>/dev/null || { echo "✗ matplotlib missing"; }
$PYTHON_CMD -c "import torch; print('✓ torch')" 2>/dev/null || { echo "✗ torch missing"; }
$PYTHON_CMD -c "import datasets; print('✓ datasets')" 2>/dev/null || { echo "✗ datasets missing"; }
$PYTHON_CMD -c "import PIL; print('✓ PIL')" 2>/dev/null || { echo "✗ PIL missing"; }

echo ""
echo "Attempting to generate examples..."

# Try with conda environment if available
if [ -f /opt/conda/bin/conda ]; then
    echo "Activating conda environment..."
    source /opt/conda/bin/activate
    conda activate llm_cholec 2>/dev/null || conda activate base
fi

# Run the script
$PYTHON_CMD notebooks_py/generate_paper_examples.py \
    --results-dir results/pointing_original \
    --output-dir /shared_data0/weiqiuy/llm_cholec_organ_paper/images/examples \
    --eval-type zero_shot \
    --models gpt-4.1 gemini-2.0-flash claude-sonnet-4-20250514 \
    --num-examples 8