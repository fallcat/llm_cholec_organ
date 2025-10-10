#!/usr/bin/env python3
"""
Bounding Box Evaluation with Balanced/Unbalanced Split Support

This script evaluates models on bounding box detection with support for:
- Balanced (pre-computed) or unbalanced (video-aware random) sampling
- Multi-seed evaluation with aggregation
- Automatic caching of unbalanced indices for reproducibility

USAGE:
    # Balanced evaluation (default)
    python3 eval_bbox_balanced_unbalanced.py
    
    # Unbalanced with multiple seeds
    python3 eval_bbox_balanced_unbalanced.py --split unbalanced --seeds "42,7,2025,518,1337"
    
    # With specific model and samples
    python3 eval_bbox_balanced_unbalanced.py --split unbalanced --model gpt-4.1 --num-samples 50
"""

import os
import sys
import json
import math
import argparse
from pathlib import Path
from datetime import datetime
import time
import numpy as np
from typing import Dict, List, Tuple, Optional

# Add src to path
sys.path.insert(0, '/shared_data0/weiqiuy/llm_cholec_organ/src')

# Load API keys
api_keys_file = Path("/shared_data0/weiqiuy/llm_cholec_organ/API_KEYS2.json")
if api_keys_file.exists():
    with open(api_keys_file, "r") as f:
        api_keys = json.load(f)
    
    os.environ['OPENAI_API_KEY'] = api_keys.get('OPENAI_API_KEY', '')
    os.environ['ANTHROPIC_API_KEY'] = api_keys.get('ANTHROPIC_API_KEY', '')
    os.environ['GOOGLE_API_KEY'] = api_keys.get('GOOGLE_API_KEY', '')

from endopoint.datasets.cholecseg8k_local import CholecSeg8kLocalAdapter
from endopoint.eval.bbox_evaluator import BoundingBoxEvaluator


def set_global_seed(seed: int):
    """Set random seed for all libraries."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed) if using torch

import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

# Accept e.g. "video12", "video_12", "VID12", "VID_12", case-insensitive
_VIDEO_PATTERNS = [
    re.compile(r"(video[_-]?\d+)", re.IGNORECASE),
    re.compile(r"(vid[_-]?\d+)", re.IGNORECASE),
]

def _find_path_like_value(d: Any) -> Optional[str]:
    """
    Search a nested dict/list for a string that looks like a filesystem path.
    Heuristics: contains '/' or '\\' and has an image-like extension or folder-ish segments.
    """
    IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    def looks_like_path(s: str) -> bool:
        if not isinstance(s, str):
            return False
        if "/" in s or "\\" in s:
            p = s.lower()
            return p.endswith(IMAGE_EXTS) or "frame" in p or "video" in p or "images" in p
        return False

    if isinstance(d, dict):
        # try common keys first
        for k in ["image_path", "img_path", "path", "image_file", "frame_path"]:
            if k in d and isinstance(d[k], str) and looks_like_path(d[k]):
                return d[k]
        # then scan all values
        for v in d.values():
            res = _find_path_like_value(v)
            if res: return res
    elif isinstance(d, (list, tuple)):
        for v in d:
            res = _find_path_like_value(v)
            if res: return res
    elif isinstance(d, str) and looks_like_path(d):
        return d
    return None

def _extract_video_id_from_anywhere(ex: dict, img_path: Optional[str]) -> Optional[str]:
    # 1) Explicit fields
    for key in ["video_id", "video", "vid"]:
        v = ex.get(key)
        if isinstance(v, str) and len(v) > 0:
            return v

    # 2) Look in nested meta/info
    for key in ["meta", "info", "annotation"]:
        sub = ex.get(key)
        if isinstance(sub, dict):
            v = sub.get("video_id") or sub.get("video") or sub.get("vid")
            if isinstance(v, str) and len(v) > 0:
                return v

    # 3) Parse from path segments
    if img_path:
        parts = Path(img_path).parts
        for seg in parts:
            for pat in _VIDEO_PATTERNS:
                m = pat.search(seg)
                if m:
                    return m.group(1)
        # fallback: search full path
        for pat in _VIDEO_PATTERNS:
            m = pat.search(img_path)
            if m:
                return m.group(1)
    return None


def build_video_frame_map(adapter, test_only=True) -> Dict[str, List[int]]:
    """Build mapping of video IDs to frame indices."""
    print("Building video-frame mapping...")
    vid2idxs = {}
    
    if test_only:
        # Define test videos directly (from the dataset split)
        test_videos = ['video17', 'video24', 'video37']
        print(f"Using test videos only: {test_videos}")
        
        # Get total number of examples
        total = 0
        for split in ['train', 'validation', 'test']:
            split_size = adapter.total(split)
            print(f"  {split}: {split_size} frames")
            total += split_size
        
        print(f"Total dataset size: {total} frames")
        
        # Iterate through all examples to find those in test videos
        test_count = 0
        videos_seen = set()
        # import pdb; pdb.set_trace()
        
        # Sample first few to debug
        for i in range(min(5, total)):
            example = adapter.get_example_by_global_index(i)
            if 'image_path' in example:
                print(f"  Sample {i} path: {example['image_path']}")
        
        for global_idx in range(total):
            # Get example by global index
            example = adapter.get_example_by_global_index(global_idx)
            
            # Extract video ID from the example
            if 'image_path' in example:
                path_parts = Path(example['image_path']).parts
                video_id = None
                for part in path_parts:
                    if part.startswith('video'):
                        video_id = part.split('_')[0]  # Get just "videoXX"
                        if video_id not in videos_seen:
                            videos_seen.add(video_id)
                            print(f"  Found video: {video_id}")
                        break
                
                if video_id and video_id in test_videos:
                    vid2idxs.setdefault(video_id, []).append(global_idx)
                    test_count += 1
        
        print(f"All videos found: {sorted(videos_seen)}")
        total = test_count
    else:
        # Get total number of examples from all splits
        total = 0
        for split in ['train', 'validation', 'test']:
            total += adapter.total(split)
        
        # Iterate through all examples
        for global_idx in range(total):
            # Get example by global index
            example = adapter.get_example_by_global_index(global_idx)
            
            # Extract video ID from the example
            if 'image_path' in example:
                path_parts = Path(example['image_path']).parts
                video_id = None
                for part in path_parts:
                    if part.startswith('video'):
                        video_id = part.split('_')[0]  # Get just "videoXX"
                        break
                
                if video_id:
                    vid2idxs.setdefault(video_id, []).append(global_idx)
    
    # Sort indices for each video
    for v in vid2idxs:
        vid2idxs[v] = sorted(vid2idxs[v])
    
    print(f"Found {len(vid2idxs)} videos with {total} total frames")
    return vid2idxs


# def build_video_frame_map(adapter, use_only_test: bool = True, debug_print: int = 6) -> Dict[str, List[int]]:
#     """
#     Build {video_id: [global_idx,...]} robustly.

#     Strategy:
#       A) Build a map of path_string -> global_index by scanning ALL GLOBAL indices.
#       B) If 'test' split is available and use_only_test=True, iterate the test split
#          (adapter.get_example('test', i)), extract its path, look up global_index,
#          and assign to video buckets. If 'test' split is unavailable, fall back to
#          using ALL GLOBAL indices.

#     Returns:
#       vid2idxs: dict(video_id -> sorted list of GLOBAL indices)
#     """
#     # ---------- A) Build path -> global index ----------
#     # Determine how many global examples exist by summing known splits, else keep incrementing until failure
#     total_global = 0
#     for split in ("train", "validation", "test"):
#         try:
#             total_global += int(adapter.total(split))
#         except Exception:
#             pass
#     if total_global == 0:
#         # fallback: attempt a large bound with try/except
#         # but we already saw 8080 in your logs, so this probably isn't needed
#         total_global = 200000

#     path2global: Dict[str, int] = {}
#     print(f"[build_video_frame_map] scanning ALL GLOBAL indices (0..{total_global-1}) to map path→global...")
#     gcount = 0
#     for gidx in range(total_global):
#         try:
#             ex = adapter.get_example_by_global_index(gidx)
#         except Exception:
#             break  # stop when out of range
#         img_path = _find_path_like_value(ex)
#         if img_path:
#             key = str(Path(img_path))  # normalize
#             path2global[key] = gidx
#         gcount += 1
#         if gcount <= debug_print and img_path:
#             print(f"  global[{gidx}] path: {img_path}")
#     print(f"[build_video_frame_map] mapped {len(path2global)} paths to global indices")

#     # ---------- B) Build video→indices using test split if available ----------
#     def split_size(split: str) -> int:
#         try:
#             return int(adapter.total(split))
#         except Exception:
#             return 0

#     n_test = split_size("test")
#     vid2idxs: Dict[str, List[int]] = {}

#     if use_only_test and n_test > 0:
#         print(f"[build_video_frame_map] iterating TEST split ({n_test} examples)...")
#         for i in range(n_test):
#             ex = adapter.get_example("test", i)
#             img_path = _find_path_like_value(ex)
#             if not img_path:
#                 continue
#             video_id = _extract_video_id_from_anywhere(ex, img_path)
#             # Look up the global index for this test example by its path
#             gidx = path2global.get(str(Path(img_path)))
#             if gidx is None:
#                 # path mismatch due to different normalization; try as-is, lower, etc.
#                 alt = str(Path(img_path)).lower()
#                 matched = None
#                 for k, v in path2global.items():
#                     if k.lower() == alt:
#                         matched = v
#                         break
#                 gidx = matched
#             if gidx is None:
#                 continue
#             if video_id is None:
#                 video_id = "unknown"
#             vid2idxs.setdefault(video_id, []).append(gidx)
#     else:
#         # Fallback: use all GLOBAL indices (unrestricted)
#         print("[build_video_frame_map] using ALL GLOBAL indices (no test-only split detected)")
#         # We still try to attach video ids for distribution; if we cannot find any, put into 'unknown'
#         for gidx in range(len(path2global)):  # cheaper than re-walking get_example_by_global_index
#             try:
#                 ex = adapter.get_example_by_global_index(gidx)
#             except Exception:
#                 break
#             img_path = _find_path_like_value(ex)
#             video_id = _extract_video_id_from_anywhere(ex, img_path)
#             if video_id is None:
#                 video_id = "unknown"
#             vid2idxs.setdefault(video_id, []).append(gidx)

#     # Sort
#     for v in vid2idxs:
#         vid2idxs[v] = sorted(vid2idxs[v])

#     total_kept = sum(len(v) for v in vid2idxs.values())
#     print(f"[build_video_frame_map] videos: {len(vid2idxs)}; kept indices: {total_kept}")
#     if total_kept == 0:
#         print("WARNING: Still 0 indices. Dumping a few example keys for diagnosis...")
#         try:
#             ex0 = adapter.get_example_by_global_index(0)
#             print("example[0] keys:", list(ex0.keys()))
#             print("example[0] value types:", {k: type(v).__name__ for k, v in ex0.items()})
#         except Exception as e:
#             print("Could not fetch example[0]:", e)
#     return vid2idxs

def sample_unbalanced_frames(
    adapter,
    N: int = 200,
    seed: int = 42
) -> Tuple[List[int], Dict[str, int]]:
    """
    Sample N frames uniformly from all available frames.
    """
    rng = np.random.default_rng(seed)
    all_test_indices = adapter.get_test_indices()
    all_frames = rng.choice(all_test_indices, size=N, replace=False)
    return all_frames.tolist(), {}



def sample_unbalanced_frames_video_aware(
    vid2idxs: Dict[str, List[int]],
    N: int = 200,
    seed: int = 42,
    exclude_videos: List[str] = None,
    max_frames_per_video: int = None,
    min_frame_gap: int = 5
) -> Tuple[List[int], Dict[str, int]]:
    """
    Sample N frames using video-aware random sampling.
    
    Args:
        vid2idxs: Mapping of video IDs to frame indices
        N: Number of frames to sample
        seed: Random seed
        exclude_videos: Videos to exclude from sampling
        max_frames_per_video: Maximum frames per video (default: N/3)
        min_frame_gap: Minimum gap between sampled frames in same video
    
    Returns:
        Tuple of (sampled indices, video distribution)
    """
    if exclude_videos is None:
        exclude_videos = []
    rng = np.random.default_rng(seed)

    # Flatten all eligible frames
    all_frames = []
    frame_to_vid = {}
    for v, idxs in vid2idxs.items():
        if v in exclude_videos:
            continue
        for f in idxs:
            all_frames.append(f)
            frame_to_vid[f] = v

    if len(all_frames) < N:
        print(f"Warning: Only {len(all_frames)} total frames available, sampling all of them")
        N = len(all_frames)

    print(f"Sampling {N} frames uniformly from {len(all_frames)} available frames (seed={seed})...")

    # Random sample without replacement
    sampled = rng.choice(all_frames, size=N, replace=False)
    sampled = np.sort(sampled)

    # Count per-video distribution
    dist = {}
    for f in sampled:
        v = frame_to_vid[f]
        dist[v] = dist.get(v, 0) + 1

    print(f"Sampled from {len(dist)} videos: {sorted(dist.items())}")
    
    return sampled.tolist(), dist


def get_or_make_unbalanced_indices(
    adapter,
    N: int,
    seed: int,
    exclude_videos: List[str],
    save_dir: Path
) -> List[int]:
    """Get or create unbalanced indices for a given seed."""
    save_dir.mkdir(parents=True, exist_ok=True)
    out_file = save_dir / f"unbalanced_test_indices_seed{seed}.json"
    
    if out_file.exists():
        print(f"Loading existing unbalanced indices from {out_file}")
        with open(out_file, 'r') as f:
            data = json.load(f)
        return data["indices"]
    
    print(f"Generating new unbalanced indices with seed {seed}")
    # vid2idxs = build_video_frame_map(adapter)
    # indices, dist = sample_unbalanced_frames_video_aware(
    #     vid2idxs, N=N, seed=seed, exclude_videos=exclude_videos
    # )
    # vid2idxs = build_video_frame_map(adapter) #, use_only_test=True)
    # indices, dist = sample_unbalanced_frames_video_aware(
    #     vid2idxs, N=N, seed=seed, exclude_videos=exclude_videos
    #     )

    indices, dist = sample_unbalanced_frames(
        adapter, N=N, seed=seed
        )


    print(f"Sampled {len(indices)} frames from {len(dist)} videos")
    print(f"First 10 indices: {indices[:10]}")
    print(f"Video distribution: {dist}")
    
    payload = {
        "seed": seed,
        "num_samples": N,
        "indices": indices,
        "video_distribution": dist,
        "num_videos": len(dist),
        "timestamp": datetime.now().isoformat()
    }

    print(f"Payload: {payload}")
    
    with open(out_file, 'w') as f:
        json.dump(payload, f, indent=2)
    
    print(f"Saved unbalanced indices to {out_file}")
    return indices


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Bounding box evaluation with balanced/unbalanced splits")
    
    # Split configuration
    parser.add_argument("--split", choices=["balanced", "unbalanced"], default="balanced",
                        help="Sampling strategy: balanced (pre-computed) or unbalanced (random)")
    parser.add_argument("--num-samples", type=int, default=200,
                        help="Number of samples to evaluate")
    parser.add_argument("--seeds", type=str, default="42,7,2025,518,1337",
                        help="Comma-separated random seeds for unbalanced sampling")
    parser.add_argument("--exclude-videos", type=str, default="",
                        help="Comma-separated video IDs to exclude from sampling")
    
    # Model configuration
    parser.add_argument("--model", type=str, default=None,
                        help="Model to evaluate (overrides EVAL_MODEL env var)")
    parser.add_argument("--detection-mode", choices=["combined", "separate"], default="combined",
                        help="Detection mode")
    parser.add_argument("--use-fewshot", action="store_true",
                        help="Use few-shot examples")
    
    # Output configuration
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: auto-generated)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip evaluation if results already exist")
    parser.add_argument("--use-cache", action="store_true", default=True,
                        help="Use cached LLM responses")
    
    return parser.parse_args()


def check_results_exist(output_dir: Path, model: str, detection_mode: str, use_fewshot: bool) -> bool:
    """Check if results already exist for this configuration."""
    subdir = f"{detection_mode}_{'fewshot' if use_fewshot else 'zeroshot'}"
    model_dir = output_dir / subdir / model
    summary_file = model_dir / f"summary_{model}.json"
    
    if summary_file.exists():
        print(f"Results already exist at {summary_file}")
        try:
            with open(summary_file, 'r') as f:
                data = json.load(f)
            metrics = data.get('metrics', {})
            print(f"  Presence Accuracy: {metrics.get('presence_accuracy', 'N/A'):.1%}")
            print(f"  Created: {data.get('timestamp', 'Unknown')}")
            return True
        except:
            return False
    return False


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Parse configuration
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    exclude_videos = [v.strip() for v in args.exclude_videos.split(",") if v.strip()]
    
    # Model configuration (CLI overrides env var)
    MODEL = args.model or os.environ.get('EVAL_MODEL', 'gpt-4.1')
    DETECTION_MODE = args.detection_mode
    USE_FEWSHOT = args.use_fewshot
    USE_CACHE = args.use_cache
    NUM_SAMPLES = args.num_samples
    DATASET_NAME = "cholecseg8k_local"
    
    print("=" * 80)
    print("BOUNDING BOX EVALUATION - BALANCED/UNBALANCED")
    print("=" * 80)
    print(f"Model: {MODEL}")
    print(f"Split: {args.split}")
    print(f"Seeds: {seeds}")
    print(f"Samples: {NUM_SAMPLES}")
    print(f"Detection: {DETECTION_MODE}")
    print(f"Few-shot: {USE_FEWSHOT}")
    print(f"Cache: {USE_CACHE}")
    if exclude_videos:
        print(f"Excluded videos: {exclude_videos}")
    print()
    
    # Load dataset
    data_dir = "/shared_data0/weiqiuy/datasets/cholecseg8k"
    dataset_adapter = CholecSeg8kLocalAdapter(data_dir=data_dir)
    
    # Get image dimensions
    example = dataset_adapter.get_example('train', 0)
    img_width, img_height = example['image'].size
    print(f"Image dimensions: {img_width}x{img_height}")
    
    # Process each seed
    for seed in seeds:
        print("\n" + "=" * 80)
        print(f"SEED {seed}")
        print("=" * 80)
        
        # Get test indices based on split mode
        if args.split == "balanced":
            # Load pre-computed balanced indices
            indices_dir = Path(f"/shared_data0/weiqiuy/llm_cholec_organ/data_info/{DATASET_NAME}_balanced_200")
            with open(indices_dir / "balanced_test_indices_advanced_200.json", 'r') as f:
                test_data = json.load(f)
            test_indices = test_data['indices'][:NUM_SAMPLES]
            split_suffix = f"balanced{NUM_SAMPLES}_seed{seed}"
        else:
            # Generate or load unbalanced indices
            save_dir = Path(f"/shared_data0/weiqiuy/llm_cholec_organ/data_info/{DATASET_NAME}_unbalanced_{NUM_SAMPLES}")
            test_indices = get_or_make_unbalanced_indices(
                dataset_adapter, NUM_SAMPLES, seed, exclude_videos, save_dir
            )
            split_suffix = f"unbalanced{NUM_SAMPLES}_seed{seed}"
        
        print(f"Selected {len(test_indices)} test indices")
        print(f"First 10 indices: {test_indices[:10]}")
        
        # Create output directory
        if args.output_dir:
            output_dir = Path(args.output_dir) / f"bbox_{DATASET_NAME}_{split_suffix}"
        else:
            output_dir = Path(f"/shared_data0/weiqiuy/llm_cholec_organ/results/bbox_{DATASET_NAME}_{split_suffix}")
        
        # Check if results exist (if skip-existing flag is set)
        if args.skip_existing:
            if check_results_exist(output_dir, MODEL, DETECTION_MODE, USE_FEWSHOT):
                print("Skipping evaluation (--skip-existing flag is set)")
                continue
        
        # Load few-shot plan if needed
        fewshot_plan = None
        fewshot_examples = None
        
        if USE_FEWSHOT:
            indices_dir = Path(f"/shared_data0/weiqiuy/llm_cholec_organ/data_info/{DATASET_NAME}_balanced_200")
            if DETECTION_MODE == 'combined':
                with open(indices_dir / "fewshot_plan_bbox_combined_greedy.json", 'r') as f:
                    combined_plan = json.load(f)
                fewshot_examples = combined_plan.get('examples', [])
                print(f"Loaded {len(fewshot_examples)} combined few-shot examples")
            else:
                with open(indices_dir / "fewshot_plan_bbox_200.json", 'r') as f:
                    fewshot_plan = json.load(f)
                print(f"Loaded separate few-shot plan")
        
        # Initialize evaluator
        evaluator = BoundingBoxEvaluator(
            models=[MODEL],
            dataset=None,
            dataset_adapter=dataset_adapter,
            canvas_width=img_width,
            canvas_height=img_height,
            output_dir=output_dir,
            use_cache=USE_CACHE,
            min_pixels=50
        )
        
        print(f"Output directory: {evaluator.output_dir}")
        
        # Run evaluation
        start_time = time.time()
        
        try:
            results = evaluator.evaluate_model(
                model_name=MODEL,
                test_indices=test_indices,
                detection_mode=DETECTION_MODE,
                use_fewshot=USE_FEWSHOT,
                fewshot_plan=fewshot_plan if USE_FEWSHOT and DETECTION_MODE == 'separate' else None,
                fewshot_examples=fewshot_examples if USE_FEWSHOT and DETECTION_MODE == 'combined' else None,
                split='test'
            )
            
            elapsed = time.time() - start_time
            
            # Store results with seed info
            results_summary = {
                "model": MODEL,
                "split": args.split,
                "seed": seed,
                "num_samples": NUM_SAMPLES,
                "detection_mode": DETECTION_MODE,
                "use_fewshot": USE_FEWSHOT,
                "timestamp": datetime.now().isoformat(),
                "metrics": {
                    "presence_accuracy": results['metrics']['presence_accuracy'],
                    "mean_iou_bbox_to_bbox": results['metrics'].get('mean_iou_bbox_to_bbox', 0),
                    "mean_iou_bbox_to_mask": results['metrics'].get('mean_iou_bbox_to_mask', 0),
                    "elapsed_seconds": elapsed
                }
            }
            
            print(f"✓ Presence Accuracy: {results['metrics']['presence_accuracy']:.1%}")
            print(f"  Bbox-to-Bbox IoU: {results['metrics'].get('mean_iou_bbox_to_bbox', 0):.3f}")
            print(f"  Bbox-to-Mask IoU: {results['metrics'].get('mean_iou_bbox_to_mask', 0):.3f}")
            print(f"  Time: {elapsed:.1f}s")
            
        except Exception as e:
            print(f"✗ Error: {str(e)}")
            results_summary = {
                "model": MODEL,
                "split": args.split,
                "seed": seed,
                "num_samples": NUM_SAMPLES,
                "detection_mode": DETECTION_MODE,
                "use_fewshot": USE_FEWSHOT,
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "elapsed_seconds": time.time() - start_time
            }
        
        # Save summary
        summary_filename = f"summary_{DETECTION_MODE}_{'fewshot' if USE_FEWSHOT else 'zeroshot'}.json"
        summary_file = evaluator.output_dir / summary_filename
        with open(summary_file, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"Saved summary to: {summary_file}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Model: {MODEL}")
    print(f"Split: {args.split}")
    print(f"Seeds evaluated: {seeds}")
    print(f"Samples per seed: {NUM_SAMPLES}")
    
    # TODO: Add aggregation across seeds if multiple seeds were run
    if len(seeds) > 1:
        print("\nNote: Multi-seed aggregation will be implemented in a separate script")


if __name__ == "__main__":
    main()