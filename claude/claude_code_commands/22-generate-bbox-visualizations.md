# 07 - Generate Bounding Box Visualizations

## Date: 2025-09-09

## Objective
Create visualization figures showing ground truth and predicted bounding boxes for organ detection results, specifically for three target organs: Gastrointestinal Tract, Gallbladder, and L-hook Electrocautery.

## Context
The user needed to generate figures similar to a LaTeX table showing per-organ visualizations, but using actual bounding boxes from the evaluation results instead of masks. The visualizations should show both ground truth (green boxes) and predictions (red boxes) with IoU scores.

## Implementation

### 1. Created Visualization Script
**File**: `notebooks_py/generate_bbox_organ_figures.py`

Key features:
- Loads actual CholecSeg8k images using the local dataset adapter
- Uses global indices for consistency with evaluation results
- Processes multiple examples for comparison
- Organizes outputs in folders named `{dataset_name}_{global_index}`

### 2. Script Components

#### Data Loading
- Uses `CholecSeg8kLocalAdapter` to load images from `/shared_data0/weiqiuy/datasets/cholecseg8k`
- Handles global index mapping correctly (indices in result files ARE global indices)

#### Bounding Box Extraction
- Reads from `bboxes_true` and `bboxes_pred` in result JSON files
- Format: `[x_min, y_min, x_max, y_max]`

#### Visualization
- **Green boxes**: Ground truth bounding boxes
- **Red boxes**: Model predictions
- **Line thickness**: 4x increased for visibility (12px for main boxes, 8px for GT overlay)
- **IoU display**: Shows IoU score in corner when both GT and prediction exist

#### Organization
- Creates folder for each example: `cholecseg8k_840/`, `cholecseg8k_4081/`
- Saves all organ-model combinations in the same folder
- Generates LaTeX table for each example

### 3. Example Selection

#### Example 840
- Has all three target organs in ground truth
- GPT predicts 2/3 organs (Gallbladder, L-hook Electrocautery)
- Good for showing complete ground truth coverage

#### Example 4081  
- Has best IoU (0.75) for Gastrointestinal Tract with GPT
- Shows excellent prediction performance
- Note: Doesn't have L-hook Electrocautery in ground truth

### 4. Key Issues Resolved

#### Index Mapping
- Initial confusion about index mapping (test split vs global indices)
- Confirmed: Result files use global indices directly
- No conversion needed - use indices as-is with `get_example_by_global_index()`

#### Missing Ground Truth
- Initial example (645) didn't have Gastrointestinal Tract
- Found better examples through systematic search
- Gastrointestinal Tract is relatively rare in the dataset

#### Data Structure
- Result files have both `bboxes_true`/`bboxes_pred` (for direct bbox access)
- And `organs` array (for IoU and presence information)
- Script uses `bboxes_true`/`bboxes_pred` for actual coordinates

## Output Structure

```
notebooks/images/bbox_examples/
├── cholecseg8k_840/
│   ├── GT_Gastrointestinal_Tract.png
│   ├── GT_Gallbladder.png
│   ├── GT_L_hook_Electrocautery.png
│   ├── GPT_Gastrointestinal_Tract.png
│   ├── GPT_Gallbladder.png
│   ├── GPT_L_hook_Electrocautery.png
│   ├── Gemini_Gastrointestinal_Tract.png
│   ├── ... (all models × all organs)
│   └── bbox_table_840.tex
└── cholecseg8k_4081/
    └── ... (similar structure)
```

## Models Included
- GPT-4.1
- Gemini-2.0-Flash  
- Claude-Sonnet-4
- Llava-v1.6-Mistral-7B
- Pixtral-12B
- Qwen2.5-VL-7B
- PeskaVLP (CLIP-based, no localization)
- RASO (CLIP-based, no localization)
- CholeNet
- GoNoGoNet

## Usage

```bash
python3 notebooks_py/generate_bbox_organ_figures.py
```

Note: Requires numpy, PIL, matplotlib. The script will load the CholecSeg8k dataset locally and generate all visualizations for both examples.

## Future Improvements
- Add command-line arguments for example selection
- Support for other datasets (CholecOrgans, CholecGoNoGo)
- Automatic example discovery based on criteria (e.g., "find best IoU for each organ")
- Add confidence scores or prediction probabilities if available