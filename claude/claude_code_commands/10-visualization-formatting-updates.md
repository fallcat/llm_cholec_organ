# Visualization and Table Formatting Updates

## Date: 2025-09-03

## Overview
This document details all the changes made to improve the visualization tools and LaTeX table generation for the organ pointing task results. The updates focus on better readability, consistent ordering, and publication-ready formatting for two-column papers.

## Key Changes

### 1. LaTeX Table Formatting

#### Column Reorganization
- **Before**: Metrics ordered as: Presence Acc., Hit@Pt|Pres, Gated Acc., F1
- **After**: Grouped into two categories:
  - **Presence**: Acc., F1
  - **Localization**: Hit@Pt|Pres, Gated Acc.
- Added multi-column headers with `\cmidrule` for visual grouping

#### Best Value Highlighting
- **Bold** formatting for best values in each metric
- **Italic** formatting for second-best values
- Updated captions to mention: "Best values are \textbf{bold}, second best are \textit{italicized} for each metric."

#### Model Ordering
- Consistent ordering across all tables and figures:
  1. Commercial APIs: GPT-4.1, Gemini-2.0, Claude-Sonnet-4
  2. Open-source: LLaVA-v1.6, Qwen2.5-VL, Pixtral-12B

#### Two-Column Paper Compatibility
- Changed from `\begin{table}` to `\begin{table*}[t]`
- Always use `[t]` placement instead of `[htbp]`
- Removed dependency on `tablenotes` package
- Used `\parbox` for footnote-style metric explanations

### 2. Figure Generation Updates

#### Font Size Increases (30-50% larger)
```python
plt.rcParams['font.size'] = 14          # was 11
plt.rcParams['axes.labelsize'] = 16     # was 12
plt.rcParams['axes.titlesize'] = 18     # was 12
plt.rcParams['xtick.labelsize'] = 14    # was 10
plt.rcParams['ytick.labelsize'] = 14    # was 10
plt.rcParams['legend.fontsize'] = 14    # was 10
plt.rcParams['figure.titlesize'] = 20   # was 14
```

#### Star Markers for Best Performance
- Added black stars (★) to mark highest performing bars
- **Individual model charts**: Star on highest bar for each metric
- **Comparison charts**: Single star on overall highest value
- **Overall comparison (2x2 grid)**: One star per subplot on absolute best

#### Visual Improvements
- **Legend placement**: Moved to lower-left of first subplot (Presence Accuracy)
- **Color coding**: 
  - Blue (#1f77b4) for commercial APIs
  - Green (#2ca02c) for open-source models
- **Separator line**: Gray dashed line at x=2.5 to separate API from open-source models
- **Star formatting**: 
  - Color changed from gold to black for better visibility
  - Properly centered on bars (not at edges)
  - Font sizes: 18pt for individual charts, 20pt for comparisons, 12pt for grid

#### Figure Sizes
- Individual model performance: (10, 6)
- Metric comparison: (14, 8)
- Overall comparison grid: (16, 10)

### 3. File Structure Updates

Created comprehensive guidelines in `/guidelines/` folder:
- `latex_formatting.md` - LaTeX best practices for two-column papers
- `figure_generation.md` - Matplotlib configuration and chart guidelines
- `package_requirements.md` - Dependencies and troubleshooting
- `visualization_best_practices.md` - Overall design principles

### 4. Script Improvements

#### `regenerate_metrics_comparison.py`
- Added post-hoc fixing for LLaVA coordinate normalization issue
- Detects normalized coordinates (0-1 range) and converts to pixel coordinates
- `--fix-llava` flag for selective correction

#### `generate_paper_figures.py`
- Modified to load from individual metrics_summary files instead of incomplete JSON
- Ensures all models are processed even if metrics_comparison.json is partial

#### `latex_tables.py`
- Added formatting logic for bold/italic highlighting
- Implemented consistent model ordering
- Column grouping with multi-column headers

#### `bar_charts.py`
- Metric ordering matches table columns
- Star positioning fixed to center of bars
- Legend placement improved
- Color scheme for model types

## Usage

### Generate All Figures and Tables
```bash
python3 src/endopoint/vis/generate_paper_figures.py \
    results/pointing_original \
    /shared_data0/weiqiuy/llm_cholec_organ_paper \
    --experiment-name pointing
```

### Regenerate Metrics with LLaVA Fix
```bash
python3 notebooks_py/regenerate_metrics_comparison.py \
    results/pointing_original \
    --fix-llava
```

## Output Files

### LaTeX Tables
- `figures/pointing_metrics_table.tex` - Main comprehensive table
- `figures/pointing_zero_shot_table.tex` - Zero-shot comparison
- `figures/pointing_fewshot_standard_table.tex` - Few-shot comparison
- `figures/pointing_fewshot_hard_negatives_table.tex` - Hard negatives comparison

### PDF Figures
- `images/pointing/overall_comparison.pdf` - 2x2 grid overview
- `images/pointing/*_performance.pdf` - Individual model charts
- `images/pointing/comparison_*.pdf` - Per-metric comparisons

## Key Improvements Summary

1. **Better Readability**: Larger fonts, clear star markers, logical grouping
2. **Consistent Ordering**: APIs first, then open-source across all outputs
3. **Publication Ready**: Proper table* usage, no problematic packages
4. **Visual Hierarchy**: Bold/italic for best values, stars for top performance
5. **Clear Documentation**: Comprehensive guidelines for future updates

## Notes for Paper Submission

- All tables use `table*[t]` for two-column format
- Figures use `figure*[t]` for wide figures
- No dependency on subfigure/subcaption packages
- Font sizes tested for readability at publication size
- Color scheme is colorblind-friendly with shape distinctions

## Troubleshooting

### Common Issues
1. **Missing matplotlib**: Install with `pip install matplotlib`
2. **Incomplete data**: Check that all evaluation folders have metrics_summary files
3. **LLaVA 0% accuracy**: Use `--fix-llava` flag in regeneration script
4. **Table formatting**: Ensure booktabs package is included in LaTeX preamble

## Future Enhancements
- Consider adding error bars for statistical significance
- Implement automatic detection of coordinate normalization issues
- Add support for additional visualization types (box plots, violin plots)
- Create interactive HTML versions for supplementary material