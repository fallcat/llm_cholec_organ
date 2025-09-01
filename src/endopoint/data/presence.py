"""Presence matrix computation with caching."""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from ..datasets import DatasetAdapter
from ..utils.io import load_json, save_json
from ..utils.logging import get_logger

logger = get_logger(__name__)


def compute_presence_matrix_cached(
    adapter: DatasetAdapter,
    split: str,
    indices: Sequence[int],
    *,
    min_pixels: int = 1,
    cache_dir: str | Path = "cache",
    force_recompute: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Path, Dict]:
    """Compute presence matrix with caching support.
    
    Args:
        adapter: Dataset adapter
        split: Dataset split
        indices: List of indices to process
        min_pixels: Minimum pixels for presence
        cache_dir: Directory for cache files
        force_recompute: Force recomputation
        
    Returns:
        Y: [N, K] uint8 presence matrix
        counts_per_image: [N] int array of present organs per image
        cache_path: Path to cache file
        meta: Metadata dictionary
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Build cache key
    K = len(adapter.label_ids)
    cache_key_parts = [
        adapter.dataset_tag,
        adapter.version,
        split,
        str(min_pixels),
        str(K),
        hashlib.sha256(str(sorted(indices)).encode()).hexdigest()[:12],
    ]
    cache_key = "_".join(cache_key_parts)
    
    cache_path = cache_dir / f"presence_{cache_key}.npz"
    meta_path = cache_dir / f"presence_{cache_key}.meta.json"
    
    # Check cache
    if not force_recompute and cache_path.exists() and meta_path.exists():
        try:
            # Load and validate
            data = np.load(cache_path)
            meta = load_json(meta_path)
            
            # Validate indices
            if meta.get("indices_hash") == hashlib.sha256(str(sorted(indices)).encode()).hexdigest():
                logger.info(f"Loaded presence matrix from cache: {cache_path}")
                return data["Y"], data["counts_per_image"], cache_path, meta
            else:
                logger.warning("Cache indices mismatch, recomputing")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}, recomputing")
    
    # Compute presence matrix
    logger.info(f"Computing presence matrix for {len(indices)} examples")
    
    N = len(indices)
    Y = np.zeros((N, K), dtype=np.uint8)
    counts_per_image = np.zeros(N, dtype=np.int32)
    
    for i, idx in enumerate(tqdm(indices, desc="Computing presence")):
        example = adapter.get_example(split, idx)
        img_t, lab_t = adapter.example_to_tensors(example)
        presence = adapter.labels_to_presence_vector(lab_t, min_pixels=min_pixels)
        
        Y[i] = presence.numpy()
        counts_per_image[i] = presence.sum().item()
    
    # Save cache
    meta = {
        "dataset_tag": adapter.dataset_tag,
        "version": adapter.version,
        "split": split,
        "min_pixels": min_pixels,
        "K": K,
        "N": N,
        "indices_hash": hashlib.sha256(str(sorted(indices)).encode()).hexdigest(),
        "label_ids": list(adapter.label_ids),
        "id2label": adapter.id2label,
    }
    
    np.savez_compressed(cache_path, Y=Y, counts_per_image=counts_per_image)
    save_json(meta_path, meta)
    
    logger.info(f"Saved presence matrix to cache: {cache_path}")
    
    return Y, counts_per_image, cache_path, meta