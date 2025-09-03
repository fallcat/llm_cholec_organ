# Figure Generation Updates

## Changes Made to Improve Figure Quality

### 1. Increased Font Sizes (bar_charts.py)
All font sizes have been increased for better readability when figures are displayed side-by-side:

- Base font size: 11 → 14
- Axes labels: 12 → 16
- Axes titles: 12 → 18
- Tick labels: 10 → 14
- Legend: 10 → 14
- Figure title: 14 → 20
- Value labels on bars: 9 → 12

### 2. Model Ordering
Models are now displayed in a consistent order:
1. **Commercial APIs (Blue bars):**
   - GPT-4.1
   - Gemini-2.0-Flash
   - Claude-Sonnet-4
   
2. **Open-Source VLMs (Green bars):**
   - LLaVA-v1.6
   - Qwen2.5-VL
   - Pixtral-12B

A vertical dashed line separates APIs from open-source models for clarity.

### 3. Figure Sizes
Increased figure sizes for better visibility:
- Metric comparison bars: (12, 6) → (14, 8)
- Overall comparison: (14, 8) → (16, 10)

### 4. Color Coding
- Commercial APIs: Blue (#1f77b4)
- Open-source models: Green (#2ca02c)
- Separator line: Gray dashed line at x=2.5

### 5. Data Loading Fix
Modified `generate_paper_figures.py` to load directly from individual metrics_summary files instead of relying on the potentially incomplete metrics_comparison.json.

## To Generate Figures

1. **Install matplotlib if missing:**
```bash
pip install matplotlib
```

2. **Run the generation script:**
```bash
python src/endopoint/vis/generate_paper_figures.py results/pointing_original /shared_data0/weiqiuy/llm_cholec_organ_paper
```

This will create:
- LaTeX tables in `/shared_data0/weiqiuy/llm_cholec_organ_paper/figures/`
- PDF figures in `/shared_data0/weiqiuy/llm_cholec_organ_paper/images/pointing/`

## Expected Output Files

### Individual Model Performance Charts
- `gpt-4_1_performance.pdf`
- `gemini-2_0-flash_performance.pdf`
- `claude-sonnet-4-20250514_performance.pdf`
- `llava-hf_llava-v1_6-mistral-7b-hf_performance.pdf`
- `Qwen_Qwen2_5-VL-7B-Instruct_performance.pdf`
- `mistralai_Pixtral-12B-2409_performance.pdf`

### Comparison Charts
- `overall_comparison.pdf` - All models and metrics in 2x2 grid
- `comparison_presence_accuracy_*.pdf` - Per-metric comparisons
- `comparison_hit_at_point_given_present_*.pdf`
- `comparison_gated_accuracy_*.pdf`
- `comparison_f1_score_*.pdf`

## Key Features

1. **Larger, clearer text** - All text elements are now ~30-50% larger
2. **Consistent model ordering** - APIs first, then open-source
3. **Visual grouping** - Color coding and separator line distinguish model types
4. **Better spacing** - Increased figure dimensions provide more room for labels

## Troubleshooting

If figures are not generated for all models:
1. Check that metrics_summary files exist in all evaluation directories
2. Verify the model folder structure matches expected patterns
3. Ensure matplotlib is installed: `pip install matplotlib`

If text is still too small:
- Further increase font sizes in `bar_charts.py` lines 16-22
- Increase figure sizes in function parameters