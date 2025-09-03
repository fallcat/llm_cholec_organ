#!/bin/bash
# Example usage of eval_pointing.py - works in both command line and notebooks

echo "Example usage of eval_pointing.py"
echo "================================="
echo ""
echo "COMMAND LINE USAGE:"
echo "-------------------"
echo ""

echo "1. Quick test (5 samples, GPT-4.1 only):"
echo "   EVAL_QUICK_TEST=true python3 eval_pointing.py"
echo ""

echo "2. Evaluate specific number of samples (e.g., 10 evenly spaced):"
echo "   EVAL_NUM_SAMPLES=10 python3 eval_pointing.py"
echo ""

echo "3. Evaluate specific models only:"
echo "   EVAL_MODELS='gpt-4.1,claude-sonnet-4-20250514' python3 eval_pointing.py"
echo ""

echo "4. Combine options (20 samples, specific models):"
echo "   EVAL_NUM_SAMPLES=20 EVAL_MODELS='gpt-4.1,gemini-2.0-flash' python3 eval_pointing.py"
echo ""

echo "5. Full evaluation (all test samples, all models):"
echo "   python3 eval_pointing.py"
echo ""

echo ""
echo "NOTEBOOK USAGE:"
echo "---------------"
echo ""
echo "# In a Jupyter notebook, just import and call main() directly:"
echo ""
echo "# Cell 1: Setup"
echo "%run eval_pointing.py"
echo ""
echo "# Cell 2: Quick test"
echo "main(num_samples=5, models=['gpt-4.1'])"
echo ""
echo "# Cell 3: Custom evaluation"
echo "main(num_samples=20, models=['gpt-4.1', 'claude-sonnet-4-20250514'])"
echo ""
echo "# Cell 4: Full evaluation"
echo "main()"
echo ""

echo "Note: The script uses numpy.linspace to select evenly spaced samples"
echo "      from the test set, ensuring good coverage across the dataset."