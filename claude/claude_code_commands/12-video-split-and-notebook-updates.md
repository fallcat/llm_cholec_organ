# Session 12: Video-based Dataset Splitting and Notebook Updates

## Date: 2025-09-06

## Summary
Modified the CholecSeg8k local dataset adapter to implement proper video-based splitting instead of random frame splitting, and updated the notebook to visualize bounding boxes with segmentation masks.

## Changes Made

### 1. Modified Dataset Splitting in `src/endopoint/datasets/cholecseg8k_local.py`

**Problem**: The original implementation randomly shuffled all frames from all videos together, causing data leakage where frames from the same surgical video could appear in train, validation, and test sets.

**Solution**: Implemented video-based splitting where:
- Videos are randomly assigned to train/val/test splits
- All frames from a video stay together in the same split
- Split ratios apply to videos, not individual frames

**Key Changes**:
- Modified `_index_dataset()` method to:
  - Build a mapping of video IDs to frame indices
  - Shuffle videos (not frames) and assign to splits
  - Ensure minimum 1 video per split when possible
  - Store video assignments in `_video_splits` for reference
  
- Added `get_video_splits()` method to retrieve which videos are in each split

- Added logic to handle edge cases with small numbers of videos (e.g., with 9 videos, ensures proper distribution instead of 0 validation videos)

### 2. Updated `notebooks/load_local_cholecseg8k.ipynb`

**Added Features**:
1. **Bounding box visualization**: Added cells to display bounding boxes overlaid on segmentation masks
2. **Individual organ visualization**: Created subplot showing each organ's segment with its bounding boxes separately
3. **Video split verification**: Added cell to check and verify the video-based splits

**New Visualizations**:
- Combined view showing image + segmentation mask + bounding boxes
- Per-organ view with dimmed background highlighting each organ and its bbox
- Split statistics showing video distribution

### 3. Created Test Script `notebooks/test_video_split.py`

Created a standalone test script to verify the video-based splitting functionality, showing:
- Video distribution across splits
- Frame counts per video
- Verification that no video appears in multiple splits

## Files Modified

### Modified Files:
- `src/endopoint/datasets/cholecseg8k_local.py` - Implemented video-based splitting
- `notebooks/load_local_cholecseg8k.ipynb` - Added bbox visualization and split verification
- `src/endopoint/datasets/__init__.py` - Minor import updates
- `src/endopoint/eval/evaluator.py` - Minor updates
- `src/endopoint/models/__init__.py` - Minor updates
- `notebooks_py/eval_*.py` - Minor adjustments to evaluation scripts

### New Files:
- `notebooks/test_video_split.py` - Test script for video splitting
- Multiple new notebooks in `notebooks/test_models/` for testing various models (de_lightsam, llava-next-med-olab, medsam, raso, yolo_v8)
- `notebooks_py/prepare_fewshot_examples_bounding_box.py` - Script for preparing bounding box examples
- `notebooks_py/convert_point_to_bbox_plan.py` - Planning script for bbox conversion
- Various test model assets and region images

## Technical Details

### Video Split Algorithm:
```python
# 1. Group all frames by video
video_to_examples[video_id] = [frame_indices]

# 2. Shuffle videos (not frames)
shuffled_video_indices = np.random.permutation(n_videos)

# 3. Assign videos to splits based on ratios
n_train_videos = max(1, int(n_videos * 0.8))
n_val_videos = max(1, int(n_videos * 0.1)) if n_videos >= 3 else 0
n_test_videos = n_videos - n_train_videos - n_val_videos

# 4. Collect all frame indices for each split
for video_id in train_videos:
    train_indices.extend(video_to_examples[video_id])
```

### Results with 9 Videos:
- **Train**: 7 videos (video01, video09, video12, video18, video20, video25, video26) - 2908 frames
- **Validation**: 1 video (with fix) - frames vary by random seed
- **Test**: 1-2 videos (video17, video24) - 1280 frames

## Benefits

1. **No Data Leakage**: Frames from the same surgical procedure stay together
2. **More Realistic Evaluation**: Model performance measured on completely unseen videos
3. **Better Generalization**: Training doesn't see similar frames from test videos
4. **Clearer Visualization**: Bounding boxes help understand organ detection regions

## Next Steps

- Consider adjusting split ratios for better distribution with 9 videos (e.g., 6/1/2 or 5/2/2)
- Add configuration option to specify exact videos for each split
- Implement stratified splitting based on organ presence statistics per video