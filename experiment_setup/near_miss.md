# Near-Miss Hard Negative Selection for Organ Pointing Tasks

## Overview
Near-miss hard negatives are points that are **close to but clearly outside** an organ boundary. They test whether models can precisely identify organ boundaries rather than just detecting general organ regions.

## Key Design Principles

### 1. Minimum Distance Requirement
- **10 pixels minimum** from organ boundary
- Ensures no ambiguity about whether point is on or off the organ
- Creates a "forbidden zone" around organ edges where no near-miss points can be placed
- Typical organ boundaries in surgical images have 2-5 pixel transition zones due to blur

### 2. Distance Bands Strategy
**Primary bands**: 10, 15, 20, 25, 30 pixels from boundary
- Start with closest band (10px) for maximum challenge
- Progressively expand outward if no suitable point found
- 5-pixel increments provide gradual expansion

**Extended fallback bands**: 40, 50, 60, 80, 100 pixels
- Used only when primary bands fail
- Maintains "near-miss" concept (within ~7% of image width)
- Beyond 100 pixels becomes "clearly wrong" rather than "near-miss"

### 3. Color Similarity Criterion
The primary selection criterion is **visual similarity** to the target organ:

```python
# Calculate mean RGB color of organ
organ_mean_color = average(all_pixels_in_organ)

# Find point in band with most similar color
for each point in band:
    color_distance = L2_norm(point_color - organ_mean_color)
select point with minimum color_distance
```

**Rationale**: Creates realistic hard negatives that:
- Look like they could be part of the organ (similar color/texture)
- Test boundary detection, not just color detection
- Mimic real surgical confusion scenarios

### 4. Verification Steps
Each selected point undergoes multiple checks:
1. Confirm point is outside organ mask
2. Verify minimum distance using distance transform
3. Ensure adequate spacing from other selected points (100+ frames apart for video data)

## Visual Example

```
        Organ (e.g., Liver)
    ████████████████████████
  ██████████████████████████████
██████████████████████████████████
██████████ORGAN█PIXELS████████████  
██████████████████████████████████
  ██████████████████████████████
    ████████████████████████
    
Distance zones:
├─ 0-9px:   ❌ FORBIDDEN (too close, ambiguous)
├─ 10-30px: ✅ PRIMARY (ideal near-miss zone)
├─ 31-100px: 🔄 FALLBACK (if primary fails)
└─ >100px:  ❌ TOO FAR (not "near-miss")
```

## Example Selection Scenarios

### Good Near-Miss Points
- **Liver boundary**: Fatty tissue 15px away with similar yellowish color
- **Gallbladder**: Adjacent tissue 20px away with similar texture
- **Surgical instrument**: Nearby reflection 12px away with metallic appearance

### Poor Near-Miss Points (Avoided)
- Points < 10px from boundary (ambiguous)
- Black background pixels (too different visually)
- Points > 100px away (not challenging)

## Fallback Strategy

When no color-similar point exists in primary bands:

1. **Try extended bands** (40-100px) with relaxed color requirements
2. **Random selection within band** if color matching completely fails
3. **Return None** if no valid point exists within 100px
   - Prevents data quality degradation
   - Calling code can skip this example

## Context for Surgical Images

### Scale Considerations
- Typical organ diameter: 50-300 pixels
- 10-30 pixel distance ≈ 3-10% of organ size
- Captures the "almost but not quite" zone where surgeons might hesitate

### Why These Parameters Work
- **10px minimum**: Clear separation from natural boundary blur
- **30px preferred maximum**: Still visually proximate to organ
- **100px absolute maximum**: ~7% of 1280px width, maintains locality

## Implementation Benefits

### Training Advantages
1. **Precise boundary learning**: Models must identify exact organ edges
2. **Color-invariant detection**: Can't rely solely on color similarity
3. **Realistic challenges**: Mimics actual surgical decision points

### Quality Assurance
- Deterministic selection with seed control
- Minimum distance guarantee eliminates ambiguity
- Diversity enforcement (100+ frame spacing) prevents redundancy
- None return for impossible cases maintains data integrity

## Usage in Few-Shot Learning

Near-miss points serve as hard negatives in few-shot pointing tasks:
- **Positive examples**: Points correctly on organs
- **Easy negatives**: Random points clearly outside organs  
- **Near-miss hard negatives**: Challenging points just outside boundaries

This three-tier approach creates a curriculum from easy to hard, improving model robustness for precise organ localization in surgical procedures.