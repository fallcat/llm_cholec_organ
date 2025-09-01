"""Base dataset adapter protocol and registry."""

from typing import Any, Callable, Dict, Optional, Protocol, Sequence, Tuple

import torch
from PIL import Image


class DatasetAdapter(Protocol):
    """Protocol for dataset adapters.
    
    All dataset implementations must follow this interface to ensure
    compatibility with the endopoint system.
    """
    
    # Identity & schema
    @property
    def dataset_tag(self) -> str:
        """Unique identifier for the dataset."""
        ...
    
    @property
    def version(self) -> str:
        """Dataset version (e.g., 'v1' or HF revision)."""
        ...
    
    @property
    def id2label(self) -> Dict[int, str]:
        """Mapping from class ID to label name."""
        ...
    
    @property
    def label_ids(self) -> Sequence[int]:
        """List of valid label IDs (excludes background)."""
        ...
    
    @property
    def ignore_index(self) -> int:
        """Label ID to ignore in loss computation (-1 if none)."""
        ...
    
    @property
    def recommended_canvas(self) -> Tuple[int, int]:
        """Recommended canvas size (width, height) for processing."""
        ...
    
    # Data access
    def total(self, split: str) -> int:
        """Get total number of examples in split.
        
        Args:
            split: Dataset split name (e.g., 'train', 'val', 'test')
            
        Returns:
            Number of examples in split
        """
        ...
    
    def get_example(self, split: str, index: int) -> Any:
        """Get example from dataset.
        
        Args:
            split: Dataset split name
            index: Example index
            
        Returns:
            Dataset-specific example object
        """
        ...
    
    def example_to_tensors(self, example: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert example to image and label tensors.
        
        Args:
            example: Dataset-specific example object
            
        Returns:
            img_t: FloatTensor [C,H,W] in [0,1]
            lab_t: LongTensor [H,W] with class IDs
        """
        ...
    
    # Semantics
    def labels_to_presence_vector(
        self, lab_t: torch.Tensor, min_pixels: int = 1
    ) -> torch.Tensor:
        """Convert label tensor to presence vector.
        
        Args:
            lab_t: Label tensor [H,W]
            min_pixels: Minimum pixels for presence
            
        Returns:
            LongTensor [K] over self.label_ids with {0,1}
        """
        ...
    
    # Optional utilities
    def sample_point_in_mask(
        self, 
        lab_t: torch.Tensor, 
        class_id: int, 
        strategy: str = "centroid"
    ) -> Optional[Tuple[int, int]]:
        """Sample a point within the mask for a given class.
        
        Args:
            lab_t: Label tensor [H,W]
            class_id: Class ID to sample from
            strategy: Sampling strategy ('centroid', 'random', etc.)
            
        Returns:
            (x, y) coordinates or None if class not present
        """
        ...


# Dataset registry
DATASET_REGISTRY: Dict[str, Callable[..., DatasetAdapter]] = {}


def register_dataset(name: str) -> Callable:
    """Decorator to register a dataset adapter factory.
    
    Args:
        name: Dataset name for registry
        
    Returns:
        Decorator function
    """
    def decorator(fn: Callable[..., DatasetAdapter]) -> Callable[..., DatasetAdapter]:
        DATASET_REGISTRY[name] = fn
        return fn
    return decorator


def build_dataset(name: str, **cfg: Any) -> DatasetAdapter:
    """Build dataset adapter from registry.
    
    Args:
        name: Dataset name in registry
        **cfg: Configuration parameters for dataset
        
    Returns:
        Dataset adapter instance
        
    Raises:
        KeyError: If dataset name not found in registry
    """
    if name not in DATASET_REGISTRY:
        available = list(DATASET_REGISTRY.keys())
        raise KeyError(
            f"Dataset '{name}' not found in registry. "
            f"Available datasets: {available}"
        )
    
    return DATASET_REGISTRY[name](**cfg)