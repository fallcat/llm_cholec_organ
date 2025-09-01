# Pointing Experiments Setup

## Overview
The pointing experiments evaluate Vision-Language Models (VLMs) on their ability to:
1. **Detect** the presence/absence of surgical organs and tools in laparoscopic images
2. **Localize** detected organs by providing pixel coordinates (pointing)

The experiments test three conditions:
- **Zero-shot**: No examples provided
- **Few-shot standard**: With positive and negative examples
- **Few-shot hard negatives**: With challenging near-miss negative examples

## Experiment Pipeline

### Step 1: Data Preparation (`notebooks_py/prepare_fewshot_examples.py`)

#### 1.1 Build Presence Matrix
- Load all 8,080 training samples from CholecSeg8k dataset
- For each sample, extract presence/absence of 12 organ classes
- Create presence matrix Y [8080 × 12] where Y[i,j] = 1 if organ j is present in sample i
- Cache to `data_info/cholecseg8k/presence_matrix_train_8080.npz`

#### 1.2 Select Balanced Test Set
- Use greedy algorithm to select 100 balanced test samples
- Ensures good coverage of all organ classes
- Saves indices to `balanced_indices_train_100.json`

#### 1.3 Create Few-Shot Plans
For each organ class, select:

**Standard Plan** (`fewshot_plan_train_pos1_neg1_seed43_excl100.json`):
- 1 positive example (organ present)
- 1 negative example (organ absent)

**Hard Negatives Plan** (`fewshot_plan_train_pos1_neg1_nearmiss1_seed45_excl100.json`):
- 1 positive example
- 1 easy negative example
- 1 hard negative "near-miss" example (visually similar context but organ absent)

Examples are selected with:
- Minimum spacing between samples (diversity)
- Exclusion of test samples (no data leakage)
- Random seed for reproducibility

### Step 2: Model Evaluation (`notebooks_py/eval_pointing.py`)

#### 2.1 Initialization
```python
# Load dataset
dataset = load_dataset("minwoosun/CholecSeg8k")

# Load test indices (100 balanced samples)
test_indices = load_balanced_indices("balanced_indices_train_100.json")

# Optional: Select subset using linspace for quick testing
if num_samples < 100:
    indices = np.linspace(0, 99, num_samples, dtype=int)
    test_indices = [test_indices[i] for i in indices]

# Load few-shot plans
fewshot_plans = {
    "standard": load_fewshot_plan("fewshot_plan_train_pos1_neg1_seed43_excl100.json"),
    "hard_negatives": load_fewshot_plan("fewshot_plan_train_pos1_neg1_nearmiss1_seed45_excl100.json")
}
```

#### 2.2 Evaluation Loop
For each model (GPT-4o-mini, Claude-3.5-Sonnet, Gemini-2.0-Flash):

##### A. Zero-Shot Evaluation
For each test sample:
1. Load image and ground truth labels
2. For each of 12 organs:
   - Query model with system prompt + user prompt + image
   - System prompt defines JSON format and coordinate system (768×768 canvas)
   - User prompt asks about specific organ
   - Parse JSON response: `{"name": "organ", "present": 0/1, "point_canvas": [x,y] or null}`
3. Compare predictions with ground truth

##### B. Few-Shot Evaluation (Standard & Hard Negatives)
For each test sample:
1. Load image and ground truth labels
2. For each of 12 organs:
   - Prepare few-shot examples from plan:
     * Load example images
     * Format expected responses
   - Build combined query tuple:
     ```python
     prompt_parts = [
         "Here are some examples:\n",
         "Example 1: Organ: Liver", example1_image,
         "Response: {\"name\":\"Liver\",\"present\":1,\"point_canvas\":[384,400]}",
         "Example 2: Organ: Liver", example2_image,  
         "Response: {\"name\":\"Liver\",\"present\":0,\"point_canvas\":null}",
         "Now for the actual query: Organ: Liver", current_image
     ]
     ```
   - Send to model as single tuple
   - Parse JSON response
3. Compare predictions with ground truth

### Step 3: Core Components

#### 3.1 Prompt Engineering (`src/endopoint/prompts/builders.py`)

**System Prompt**:
```
You are a surgical vision validator looking at ONE image on a fixed canvas.
Return STRICT JSON only: {"name":"<organ>", "present":0|1, "point_canvas":[x,y] or null}
- Coordinates: origin=(0,0) is top-left of the CANVAS, x∈[0,767], y∈[0,767], integers only.
- present=1 ONLY if any visible part of the named structure is in view.
- If present=1, point_canvas MUST be inside the structure; else use null.
- No extra text or markdown.
```

**User Prompt**:
```
Organ: "{organ_name}". 
Return exactly: {"name":"{organ_name}", "present":0|1, "point_canvas":[x,y] or null}
```

#### 3.2 Response Parsing (`src/endopoint/eval/parser.py`)
1. Try JSON parsing first
2. Fallback to regex extraction for coordinates
3. Validate coordinate bounds (0-767 for 768×768 canvas)
4. Return structured result with present/absent and optional point

#### 3.3 Model Adapters (`src/endopoint/models/`)
- Support for OpenAI, Anthropic, and Google models
- Handle multi-image inputs in single tuple
- Disk-based caching with SHA256 hash keys
- Retry logic for API failures

### Step 4: Metrics Calculation

For each organ and overall:
- **Accuracy**: (TP + TN) / Total
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1 Score**: 2 × Precision × Recall / (Precision + Recall)
- **Pointing Accuracy**: % of correct localizations when present=1

### Step 5: Results Output

Results saved to `results/pointing_YYYYMMDD_HHMMSS/`:
- `raw_results.pkl`: Complete evaluation data
- `summary.csv`: Performance metrics table
- `report.md`: Detailed markdown report with:
  - Overall accuracies
  - Per-organ metrics
  - Top/bottom performing organs
  - Comparison across conditions

## Organ Classes Evaluated
1. Abdominal Wall
2. Liver
3. Gastrointestinal Tract
4. Fat
5. Grasper
6. Connective Tissue
7. Blood
8. Cystic Duct
9. L-hook Electrocautery
10. Gallbladder
11. Hepatic Vein
12. Liver Ligament

## Key Design Decisions

### Canvas Coordinate System
- Fixed 768×768 pixel canvas
- Origin (0,0) at top-left
- Integer coordinates only
- Ensures consistency across different image sizes

### Few-Shot Strategy
- Examples shown with their images (not just text)
- Positive examples demonstrate correct pointing
- Negative examples show null responses
- Hard negatives test discrimination ability

### Caching Strategy
- Cache key includes all prompt parts and images
- Enables fast re-runs without API costs
- Can be disabled with `use_cache=False` for testing

### Evaluation Efficiency
- Batch processing per model
- Parallel API calls where supported
- Linspace sampling for quick subset testing
- Progress bars with tqdm

## Running Experiments

### Full Evaluation (100 samples, all models, all conditions):
```python
python3 eval_pointing.py
```

### Quick Test (5 samples, one model):
```python
main(num_samples=5, models=['gpt-4o-mini'], use_cache=False)
```

### Custom Configuration:
```python
main(
    num_samples=20,  # Use 20 evenly-spaced samples
    models=['gpt-4o-mini', 'claude-3-5-sonnet-20241022'],
    use_cache=True
)
```

## Expected Outcomes
- Zero-shot: Baseline performance using only task instructions
- Few-shot standard: Improved performance with examples
- Few-shot hard negatives: Test robustness with challenging examples
- Model comparison: Relative strengths across different VLMs