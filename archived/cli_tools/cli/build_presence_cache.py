"""CLI for building presence cache."""

import click
import yaml

from endopoint.datasets import build_dataset
from endopoint.data import compute_presence_matrix_cached
from endopoint.utils.logging import setup_logger

logger = setup_logger()


@click.command()
@click.option("--dataset", required=True, help="Dataset name (e.g., cholecseg8k)")
@click.option("--dataset-config", type=click.Path(exists=True), help="Path to dataset config YAML")
@click.option("--split", default="train", help="Dataset split")
@click.option("--start-idx", default=0, type=int, help="Start index")
@click.option("--end-idx", type=int, help="End index (default: all)")
@click.option("--min-pixels", default=1, type=int, help="Minimum pixels for presence")
@click.option("--cache-dir", default="cache", help="Cache directory")
@click.option("--force", is_flag=True, help="Force recompute")
def main(dataset, dataset_config, split, start_idx, end_idx, min_pixels, cache_dir, force):
    """Build presence cache for a dataset."""
    
    # Load dataset config
    if dataset_config:
        logger.info(f"Loading dataset config from {dataset_config}")
        with open(dataset_config) as f:
            config = yaml.safe_load(f)
    else:
        # Use default config path
        config_path = f"configs/dataset_{dataset}.yaml"
        logger.info(f"Loading default config from {config_path}")
        with open(config_path) as f:
            config = yaml.safe_load(f)
    
    # Build dataset adapter
    logger.info(f"Building dataset adapter for {dataset}")
    adapter = build_dataset(config["name"], **{k: v for k, v in config.items() if k != "name"})
    
    # Determine indices
    total = adapter.total(split)
    if end_idx is None:
        end_idx = total
    
    indices = list(range(start_idx, min(end_idx, total)))
    logger.info(f"Processing {len(indices)} examples from {split} split")
    
    # Compute presence matrix
    Y, counts, cache_path, meta = compute_presence_matrix_cached(
        adapter,
        split,
        indices,
        min_pixels=min_pixels,
        cache_dir=cache_dir,
        force_recompute=force,
    )
    
    # Print statistics
    logger.info(f"Presence matrix shape: {Y.shape}")
    logger.info(f"Mean organs per image: {counts.mean():.2f}")
    logger.info(f"Cache saved to: {cache_path}")
    
    # Print per-organ statistics
    organ_presence = Y.sum(axis=0)
    logger.info("\nPer-organ presence counts:")
    for i, label_id in enumerate(adapter.label_ids):
        label_name = adapter.id2label[label_id]
        count = organ_presence[i]
        pct = 100 * count / len(indices)
        logger.info(f"  {label_name}: {count}/{len(indices)} ({pct:.1f}%)")


if __name__ == "__main__":
    main()