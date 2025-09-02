# Few-Shot Learning Implementation

## Summary

Added comprehensive few-shot learning capabilities for organ detection and localization in laparoscopic surgery images. This builds upon the initial refactoring to enable models to learn from small sets of carefully selected examples.

## Key Changes

### 1. New Module: `src/few_shot_selection.py`
- **Purpose**: Implements algorithms for intelligent example selection in few-shot learning scenarios
- **Key Functions**:
  - `sample_point_in_mask()`: Samples points within organ boundaries for positive examples
  - `sample_nearmiss_point()`: Generates challenging near-miss hard negatives just outside organ boundaries
  - `select_diverse_examples()`: Ensures temporal and spatial diversity in selected examples
  - `prepare_few_shot_data()`: Main orchestrator for creating balanced few-shot datasets

### 2. Near-Miss Hard Negative Strategy (`experiment_setup/near_miss.md`)
- **Concept**: Points that are close to but clearly outside organ boundaries
- **Distance Bands**: 10-30px (primary), 40-100px (fallback)
- **Selection Criterion**: Visual similarity to target organ based on RGB color distance
- **Quality Control**: Minimum 10px from boundary to avoid ambiguity

### 3. Converted Notebooks to Python Scripts (`notebooks_py/`)
- **`prepare_fewshot_examples.py`**: Generates and visualizes few-shot learning datasets
- **`comprehensive_evaluation.py`**: Full evaluation pipeline with multiple models
- **`simple_evaluation.py`**: Quick evaluation for development
- **`simple_eval_v2.py`**: Enhanced simple evaluation with better metrics

### 4. Key Algorithms Implemented

#### Balanced Example Selection
```python
# Greedy algorithm ensures all organ types are represented
# Prioritizes rare organs to avoid class imbalance
# Maintains minimum temporal spacing (100+ frames) for diversity
```

#### Near-Miss Point Generation
```python
# 1. Compute distance transform from organ boundary
# 2. Create bands at specific distances (10, 15, 20, 25, 30px)
# 3. Find point in band with color most similar to organ
# 4. Verify point is truly outside organ mask
```

#### Three-Tier Negative Strategy
- **Easy Negatives**: Random points clearly outside organs
- **Hard Negatives**: Near-miss points just outside boundaries  
- **Contrastive Learning**: Mix of both types for robust training

## Experimental Setup

### Few-Shot Configurations
- **1-shot**: Single example per organ class
- **3-shot**: Three diverse examples per class
- **5-shot**: Five examples with maximum diversity
- **10-shot**: Extended set for comparison with traditional learning

### Example Selection Criteria
1. **Organ visibility**: Clear, unoccluded views
2. **Temporal diversity**: Examples from different video segments
3. **Visual variety**: Different lighting, angles, surgical phases
4. **Boundary clarity**: Well-defined organ edges for pointing tasks

## Results Structure

### Cached Data (`cache/`)
- Few-shot example indices
- Selected positive/negative points
- Model predictions with different prompt strategies

### Visualizations (`vis/`)
- Near-miss point demonstrations
- Few-shot example galleries
- Model performance comparisons

## Usage

### Generate Few-Shot Examples
```bash
python notebooks_py/prepare_fewshot_examples.py
```

### Run Evaluation
```bash
python notebooks_py/comprehensive_evaluation.py
```

### Visualize Results
Check `vis/` directory for:
- `nearmiss_examples.png`: Near-miss hard negative visualizations
- `fewshot_gallery_*.png`: Selected few-shot examples per organ
- Performance comparison plots

## Technical Innovations

1. **Color-Based Near-Miss Selection**: Uses RGB similarity to find visually confusing negatives
2. **Distance Band Strategy**: Progressive search from 10px to 100px ensures quality
3. **Diversity Enforcement**: Minimum 100-frame spacing prevents redundant examples
4. **Fallback Mechanisms**: Graceful degradation when ideal examples unavailable

## Future Improvements

- [ ] Implement active learning for iterative example selection
- [ ] Add semantic similarity beyond RGB color matching
- [ ] Explore curriculum learning with progressive difficulty
- [ ] Integrate with endopoint package architecture
- [ ] Add unit tests for selection algorithms