# Visualization Tools Documentation

This document describes the visualization tools available for generating publication-ready figures and tables from evaluation results.

## Overview

The visualization module (`src/endopoint/vis/`) provides tools to generate:
1. LaTeX tables with comprehensive metrics
2. PDF bar charts for model performance comparison
3. Complete LaTeX documents with all figures and tables

## Scripts

### 1. `regenerate_metrics_comparison.py`
Regenerates metrics comparison files from saved results.

**Usage:**
```bash
# Basic usage - regenerate metrics from existing results
python notebooks_py/regenerate_metrics_comparison.py /path/to/results/folder

# Example
python notebooks_py/regenerate_metrics_comparison.py results/pointing_original

# Fix LLaVA's normalized coordinates (post-hoc correction)
python notebooks_py/regenerate_metrics_comparison.py results/pointing_original --fix-llava

# Specify output directory
python notebooks_py/regenerate_metrics_comparison.py results/pointing_original --output-dir results/pointing_fixed
```

**Output Files:**
- `metrics_comparison.txt` - Detailed per-organ metrics for all models
- `metrics_comparison.json` - Structured JSON data
- `metrics_summary_table.txt` - Concise summary table with macro metrics only

**Features:**
- Handles nested model folder structures (e.g., `llava-hf/llava-v1.6-mistral-7b-hf`)
- Supports all evaluation types (zero_shot, fewshot_standard, fewshot_hard_negatives)
- Can fix LLaVA's normalized coordinates post-hoc (converts 0-1 range to pixel coordinates)

### 2. `generate_paper_figures.py`
Generates LaTeX tables and PDF figures for paper publication.

**Usage:**
```bash
# Generate all figures and tables
python src/endopoint/vis/generate_paper_figures.py /path/to/results /path/to/paper/base

# Example
python src/endopoint/vis/generate_paper_figures.py results/pointing_original /shared_data0/weiqiuy/llm_cholec_organ_paper

# Specify experiment name for subdirectories
python src/endopoint/vis/generate_paper_figures.py results/pointing_original /path/to/paper --experiment-name pointing_exp1
```

**Output Structure:**
```
/shared_data0/weiqiuy/llm_cholec_organ_paper/
├── figures/
│   ├── pointing_metrics_table.tex          # Main comprehensive metrics table
│   ├── pointing_zero_shot_table.tex        # Zero-shot comparison table
│   ├── pointing_fewshot_standard_table.tex # Few-shot comparison table
│   ├── pointing_fewshot_hard_negatives_table.tex
│   └── pointing_main.tex                   # Complete LaTeX document
└── images/
    └── pointing/
        ├── overall_comparison.pdf           # All models/methods comparison
        ├── gpt-4_1_performance.pdf         # Individual model charts
        ├── claude-sonnet-4-20250514_performance.pdf
        ├── gemini-2_0-flash_performance.pdf
        ├── llava-hf_llava-v1_6-mistral-7b-hf_performance.pdf
        ├── comparison_presence_accuracy_zero-shot.pdf
        ├── comparison_f1_score_zero-shot.pdf
        └── ...
```

## Visualization Components

### LaTeX Tables (`latex_tables.py`)

**Functions:**
- `generate_metrics_latex_table()` - Creates comprehensive table with all models and methods
- `generate_comparison_latex_table()` - Creates comparison table for specific evaluation type

**Features:**
- Automatic model name formatting for LaTeX
- Metric explanations in table notes
- Resizable tables to fit page width
- Support for booktabs styling

### Bar Charts (`bar_charts.py`)

**Functions:**
- `generate_model_performance_bars()` - Individual model performance across methods
- `generate_metric_comparison_bars()` - Compare all models for specific metric
- `generate_all_models_comparison()` - Comprehensive 2x2 subplot comparison

**Features:**
- Publication-quality PDF output (300 DPI)
- Grouped bar charts with value labels
- Consistent color scheme across figures
- Automatic model name shortening for readability

## Metrics Explained

The visualizations display four key metrics:

1. **Presence Accuracy (%)**: How well the model detects if an organ is present or absent
   - Formula: (TP + TN) / Total
   - Range: 0-100%, higher is better

2. **Hit@Point|Present (%)**: Pointing accuracy when organ is correctly detected as present
   - Only evaluated when model correctly identifies organ presence
   - Measures if predicted (x,y) coordinate falls within organ mask
   - 0% for models that don't provide valid coordinates

3. **Gated Accuracy (%)**: Combined detection and pointing performance
   - Requires BOTH correct presence detection AND accurate pointing
   - More stringent than Presence Accuracy
   - Best measure of end-to-end performance

4. **F1 Score (%)**: Harmonic mean of precision and recall for presence detection
   - Balances false positives and false negatives
   - Only considers detection, not pointing accuracy
   - Range: 0-100%, higher is better

## Installation Requirements

```bash
# Required packages
pip install numpy pandas matplotlib

# For LaTeX compilation (optional)
# Ubuntu/Debian:
sudo apt-get install texlive-full

# MacOS:
brew install --cask mactex
```

## Usage in LaTeX Documents

To include generated tables and figures in your paper:

```latex
% In your preamble
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{subcaption}

% Include table
\input{figures/pointing_metrics_table.tex}

% Include figure
\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{images/pointing/overall_comparison.pdf}
\caption{Model performance comparison across evaluation methods.}
\label{fig:overall_comparison}
\end{figure}

% Or compile the complete document
% pdflatex figures/pointing_main.tex
```

## Troubleshooting

### Font Issues
If you see "Generic family 'serif' not found" warnings:
- The script defaults to DejaVu Sans which is available on most systems
- Figures will still be generated correctly
- To use custom fonts, modify `plt.rcParams` in `bar_charts.py`

### Missing Matplotlib
```bash
pip install matplotlib
```

### Normalized Coordinates (LLaVA)
LLaVA models may return coordinates in normalized form (0-1 range). Use the `--fix-llava` flag with `regenerate_metrics_comparison.py` to convert them to pixel coordinates post-hoc.

## Examples

### Complete Workflow

```bash
# Step 1: Run evaluation (if not already done)
cd notebooks_py
python eval_pointing_original_size.py

# Step 2: Regenerate metrics (optional, if fixing LLaVA)
python regenerate_metrics_comparison.py ../results/pointing_original --fix-llava

# Step 3: Generate paper figures
cd ..
python src/endopoint/vis/generate_paper_figures.py results/pointing_original /path/to/paper

# Step 4: Compile LaTeX (optional)
cd /path/to/paper/figures
pdflatex pointing_main.tex
```

### Quick Metrics Summary

```bash
# Just get the concise metrics table
python notebooks_py/regenerate_metrics_comparison.py results/pointing_original
cat results/pointing_original/metrics_summary_table.txt
```

## Output Format Examples

### Concise Metrics Table
```
Model                               Method                    PresenceAcc     Hit@Pt|Pres     GatedAcc        F1             
------------------------------------------------------------------------------------------------------------------------
gpt-4.1                             zero shot                 70.17%          34.33%          60.92%          73.69%         
gpt-4.1                             few-shot                  78.25%          26.05%          55.58%          79.71%         
claude-sonnet-4                     zero shot                 71.33%          16.35%          49.00%          74.65%         
...
```

### LaTeX Table (excerpt)
```latex
\begin{table}[htbp]
\centering
\caption{Comprehensive evaluation metrics...}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llrrrr}
\toprule
\textbf{Model} & \textbf{Method} & \textbf{Presence Acc.} & ...
```

## Notes

- All percentages are macro-averaged across 12 organ classes
- Results based on 100 test samples by default
- Visualization tools are dataset-agnostic and work with any pointing evaluation results
- For custom visualizations, extend the functions in `bar_charts.py` and `latex_tables.py`