# Visualization Best Practices

## Overview
This guide consolidates best practices for creating publication-ready visualizations for academic papers, particularly for two-column formats common in computer vision and medical imaging conferences.

## Key Principles

### 1. Readability First
- **Font sizes must be readable** when figures are reduced to column width
- **Test your figures** at actual print size (typically 3.5 inches for single column)
- **High contrast** between text and background
- **Avoid clutter** - every element should have a purpose

### 2. Consistency Across Figures
- **Same color scheme** throughout the paper
- **Consistent font sizes** across all figures
- **Uniform style** for axes, grids, and labels
- **Standardized model ordering** in all comparisons

### 3. Information Hierarchy
- **Most important data** gets visual emphasis
- **Group related items** visually (color, position)
- **Separate distinct categories** (e.g., APIs vs open-source)
- **Progressive disclosure** in complex visualizations

## Color Usage Guidelines

### Color Semantics
Define consistent meaning for colors:
```python
COLOR_SCHEME = {
    'commercial_api': '#1f77b4',      # Blue - proprietary/commercial
    'open_source': '#2ca02c',         # Green - open/community
    'baseline': '#7f7f7f',            # Gray - reference/baseline
    'best_performance': '#ff7f0e',    # Orange - highlight best
    'error': '#d62728',               # Red - errors/failures
}
```

### Colorblind-Friendly Palettes
Use patterns or shapes in addition to color:
```python
# Add markers for line plots
markers = ['o', 's', '^', 'D', 'v', '<', '>']

# Add hatching for bar charts
hatches = ['', '/', '\\', '|', '-', '+', 'x']
```

## Table Formatting

### Effective Tables
```latex
\begin{table*}[t]
\centering
\caption{Clear, descriptive caption explaining what the table shows}
\label{tab:meaningful_label}
\begin{tabular}{l*{5}{r}}  % Left-align text, right-align numbers
\toprule
\textbf{Model} & \textbf{Metric 1} & \textbf{Metric 2} & \textbf{Metric 3} \\
\midrule
% Data rows
\bottomrule
\end{tabular}
\end{table*}
```

### Highlighting Best Results
```latex
% Bold the best value in each column
GPT-4.1 & \textbf{89.3} & 76.2 & 83.7 \\
Claude  & 87.1 & \textbf{78.5} & \textbf{85.2} \\
```

## Figure Composition

### Multi-Panel Figures
When combining multiple plots:

1. **Shared axes** where appropriate
2. **Common scale** for fair comparison
3. **Clear labels** (a), (b), (c) for each panel
4. **Single caption** explaining all panels

Example structure:
```python
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Overall Title', fontsize=16)

# Share y-axis for same metric
axes[0, 0].sharey(axes[0, 1])

# Individual panel titles
axes[0, 0].set_title('(a) Metric 1')
axes[0, 1].set_title('(b) Metric 2')
```

## Data Presentation

### Choosing the Right Chart Type

| Data Type | Recommended Visualization |
|-----------|---------------------------|
| Comparisons across categories | Bar chart |
| Trends over time | Line plot |
| Part-of-whole | Stacked bar or pie (rarely) |
| Distributions | Box plot or violin plot |
| Correlations | Scatter plot |
| Multiple metrics | Radar/spider chart or parallel coordinates |

### Statistical Significance
When showing comparisons:
- Include error bars (std dev or confidence intervals)
- Mark significant differences with asterisks
- Report p-values in caption or table

```python
# Add error bars
ax.bar(x, means, yerr=stds, capsize=5, 
       error_kw={'linewidth': 1, 'ecolor': 'gray'})

# Add significance markers
if p_value < 0.001:
    ax.annotate('***', xy=(x_pos, y_pos))
elif p_value < 0.01:
    ax.annotate('**', xy=(x_pos, y_pos))
elif p_value < 0.05:
    ax.annotate('*', xy=(x_pos, y_pos))
```

## Export Guidelines

### Resolution and Format
```python
# For paper submission
plt.savefig('figure.pdf', dpi=300, bbox_inches='tight')

# For presentations
plt.savefig('figure.png', dpi=150, bbox_inches='tight', 
            facecolor='white', edgecolor='none')

# For web/documentation
plt.savefig('figure.svg', bbox_inches='tight')
```

### File Organization
```
images/
├── main_figures/          # Figures in main paper
│   ├── figure1.pdf
│   └── figure2.pdf
├── supplementary/         # Additional figures
│   ├── supp_figure1.pdf
│   └── supp_figure2.pdf
└── source_data/          # Data used to generate figures
    ├── figure1_data.json
    └── figure2_data.json
```

## Accessibility Considerations

### Text Alternatives
- Provide detailed captions
- Include data tables in supplementary material
- Describe trends in text

### Visual Accessibility
- Minimum font size: 8pt when printed
- Line thickness: minimum 0.5pt
- Sufficient contrast ratios
- Avoid relying solely on color

## Common Pitfalls to Avoid

### Don't
- ❌ Use 3D effects unnecessarily
- ❌ Include chartjunk (decorative elements)
- ❌ Use too many colors (max 6-7 distinct)
- ❌ Forget to label axes
- ❌ Use inconsistent scales for comparison
- ❌ Include redundant legends
- ❌ Use pie charts for more than 3-4 categories
- ❌ Truncate y-axis to exaggerate differences

### Do
- ✅ Start y-axis at 0 for bar charts
- ✅ Use consistent color coding
- ✅ Include units in axis labels
- ✅ Provide context (baselines, previous work)
- ✅ Test figures at publication size
- ✅ Use vector formats (PDF, SVG)
- ✅ Include confidence intervals
- ✅ Make data available for reproducibility

## Review Checklist

Before submitting figures:

### Content
- [ ] Data is accurate and up-to-date
- [ ] All axes are labeled with units
- [ ] Legend explains all symbols/colors
- [ ] Caption provides context
- [ ] Statistical significance is indicated

### Format
- [ ] Font sizes are readable at publication size
- [ ] Colors work in grayscale
- [ ] File format is vector (PDF)
- [ ] Resolution is 300 DPI minimum
- [ ] No compression artifacts

### Consistency
- [ ] Same style across all figures
- [ ] Consistent color scheme
- [ ] Uniform model ordering
- [ ] Matching notation with text

### Accessibility
- [ ] Colorblind-friendly palette
- [ ] Sufficient contrast
- [ ] Clear visual hierarchy
- [ ] Alternative text in caption

## Examples of Good Practice

### Example 1: Model Comparison Bar Chart
```python
# Clear comparison with visual grouping
models = ['GPT-4.1', 'Gemini', 'Claude', 'LLaVA', 'Qwen', 'Pixtral']
commercial = [85, 83, 82, 0, 0, 0]
opensource = [0, 0, 0, 75, 77, 73]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width/2, commercial, width, label='Commercial', 
               color='#1f77b4')
bars2 = ax.bar(x + width/2, opensource, width, label='Open Source',
               color='#2ca02c')

# Visual separator
ax.axvline(x=2.5, color='gray', linestyle='--', alpha=0.5)

# Value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{height:.0f}%',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom')
```

### Example 2: Effective Table
```latex
\begin{table*}[t]
\centering
\caption{Performance comparison across evaluation methods. Best results per metric are \textbf{bold}.}
\label{tab:results}
\begin{tabular}{l@{\hspace{1em}}rrr@{\hspace{1em}}rrr}
\toprule
& \multicolumn{3}{c}{Zero-shot} & \multicolumn{3}{c}{Few-shot} \\
\cmidrule(lr){2-4} \cmidrule(lr){5-7}
Model & Acc. & F1 & Time & Acc. & F1 & Time \\
\midrule
GPT-4.1 & \textbf{89.3} & 87.2 & 1.2s & 91.5 & \textbf{90.1} & 1.5s \\
Claude  & 87.5 & \textbf{88.1} & 0.9s & \textbf{92.0} & 89.7 & 1.1s \\
\bottomrule
\end{tabular}
\end{table*}
```

## Resources and Tools

### Visualization Libraries
- **matplotlib**: Standard Python plotting
- **seaborn**: Statistical visualizations
- **plotly**: Interactive plots
- **tikz/pgfplots**: Native LaTeX plots

### Color Tools
- [ColorBrewer](https://colorbrewer2.org/): Color schemes for maps and charts
- [Accessible Colors](https://accessible-colors.com/): Check color contrast
- [Sim Daltonism](https://michelf.ca/projects/sim-daltonism/): Colorblind simulator

### LaTeX Packages
- **booktabs**: Professional tables
- **pgfplots**: Native LaTeX plotting
- **tikz**: Programmatic graphics
- **graphicx**: Include external graphics

## Final Thoughts

Good visualizations tell a story. They should:
1. Support your narrative
2. Highlight key findings
3. Be self-contained (understandable without reading the text)
4. Follow established conventions in your field
5. Be reproducible from provided data/code

Remember: If you have to explain what a figure shows, it needs improvement.