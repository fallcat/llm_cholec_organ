# Figure Generation Guidelines

## Font Size Requirements

### Matplotlib Configuration
All figures must use larger font sizes for better readability when displayed side-by-side:

```python
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set non-interactive backend
mpl.use('Agg')

# Configure font sizes (30-50% larger than defaults)
plt.rcParams['font.size'] = 14          # Base size (was 11)
plt.rcParams['axes.labelsize'] = 16     # Axis labels (was 12)
plt.rcParams['axes.titlesize'] = 18     # Subplot titles (was 12)
plt.rcParams['xtick.labelsize'] = 14    # X-axis tick labels (was 10)
plt.rcParams['ytick.labelsize'] = 14    # Y-axis tick labels (was 10)
plt.rcParams['legend.fontsize'] = 14    # Legend text (was 10)
plt.rcParams['figure.titlesize'] = 20   # Figure title (was 14)
```

### Value Labels on Bars
When adding value labels to bar charts:
```python
# Use fontsize=12 for value labels (was 9)
ax.annotate(f'{value:.1f}%',
           xy=(bar.get_x() + bar.get_width() / 2, height),
           xytext=(0, 3),
           textcoords="offset points",
           ha='center', va='bottom',
           fontsize=12)
```

## Model Ordering Convention

### Standard Order
Models should always be displayed in this specific order:

1. **Commercial APIs** (first group):
   - GPT-4.1
   - Gemini-2.0-Flash
   - Claude-Sonnet-4

2. **Open-Source Models** (second group):
   - LLaVA-v1.6
   - Qwen2.5-VL
   - Pixtral-12B

### Implementation
```python
def get_model_order():
    """Get the preferred order for models (APIs first, then open-source)."""
    return [
        "gpt-4.1",
        "gemini-2.0-flash", 
        "claude-sonnet-4-20250514",
        "llava-hf/llava-v1.6-mistral-7b-hf",
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "mistralai/Pixtral-12B-2409"
    ]
```

### Visual Separation
Add a vertical dashed line to separate APIs from open-source models:
```python
if len(models) > 3:
    ax.axvline(x=2.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
```

## Color Coding Standards

### Model Type Colors
Use consistent colors to distinguish model types:
```python
# Commercial APIs
API_COLOR = '#1f77b4'  # Blue

# Open-source models  
OPENSOURCE_COLOR = '#2ca02c'  # Green

# Evaluation types (for grouped bars)
COLORS = {
    'zero_shot': '#1f77b4',           # Blue
    'fewshot_standard': '#ff7f0e',    # Orange
    'fewshot_hard_negatives': '#2ca02c' # Green
}
```

## Figure Size Guidelines

### Standard Sizes
Use these figure sizes for consistency:

```python
# Individual model performance charts
MODEL_PERFORMANCE_SIZE = (10, 6)

# Metric comparison across models
METRIC_COMPARISON_SIZE = (14, 8)

# Overall comparison (2x2 grid)
OVERALL_COMPARISON_SIZE = (16, 10)
```

### Adjusting for Content
Scale figures based on number of items:
```python
# Dynamic width based on number of models
width = max(12, len(models) * 2)
fig, ax = plt.subplots(figsize=(width, 8))
```

## Grid and Layout

### Always Add Grids
Improve readability with horizontal grid lines:
```python
ax.yaxis.grid(True, linestyle='--', alpha=0.3)
ax.set_axisbelow(True)  # Place grid behind bars
```

### Tight Layout
Always use tight layout to prevent label cutoff:
```python
plt.tight_layout()
```

## File Export Settings

### DPI and Format
Save all figures as high-resolution PDFs:
```python
plt.savefig(output_path, 
            dpi=300,              # High resolution
            bbox_inches='tight',  # Prevent label cutoff
            format='pdf')         # Vector format
```

### File Naming Convention
Use descriptive, consistent names:
```
model_name_performance.pdf       # Individual model charts
comparison_metric_evaltype.pdf   # Comparison charts
overall_comparison.pdf           # Combined overview
```

## Model Name Formatting

### Display Names
Convert technical model names to readable format:
```python
MODEL_DISPLAY_NAMES = {
    "llava-hf/llava-v1.6-mistral-7b-hf": "LLaVA-v1.6",
    "mistralai/Pixtral-12B-2409": "Pixtral-12B",
    "Qwen/Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL",
    "claude-sonnet-4-20250514": "Claude-Sonnet-4",
    "gemini-2.0-flash": "Gemini-2.0",
    "gpt-4.1": "GPT-4.1"
}
```

### Safe Filenames
Clean model names for filesystem compatibility:
```python
safe_name = model_name.replace("/", "_").replace(" ", "_").replace(".", "_")
```

## Bar Chart Best Practices

### Grouped Bars
For comparing multiple evaluation types:
```python
x = np.arange(len(metrics))
width = 0.25  # Bar width

for i, eval_type in enumerate(eval_types):
    offset = (i - 1) * width
    ax.bar(x + offset, data[i], width, label=eval_type)
```

### Value Labels
Always show values on bars for precision:
```python
for bar in bars:
    height = bar.get_height()
    if height > 0:  # Only label non-zero bars
        ax.annotate(f'{height:.1f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom')
```

## Axis Configuration

### Y-axis Range
Set appropriate ranges for percentage data:
```python
ax.set_ylim(0, 105)  # 0-105% for percentage metrics
```

### X-axis Labels
Rotate long labels for readability:
```python
ax.set_xticklabels(labels, rotation=45, ha='right')
```

## Quality Checklist

Before generating figures, ensure:

- [ ] Font sizes increased by 30-50% from defaults
- [ ] Models ordered: APIs first, then open-source
- [ ] Color coding: Blue for APIs, Green for open-source
- [ ] Vertical separator line between model groups
- [ ] Grid lines added for readability
- [ ] Tight layout applied
- [ ] 300 DPI for all exports
- [ ] PDF format for vector graphics
- [ ] Value labels on all bars
- [ ] Model names properly formatted
- [ ] Safe filenames generated
- [ ] Appropriate figure sizes used