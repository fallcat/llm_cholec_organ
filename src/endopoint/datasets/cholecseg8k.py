"""CholecSeg8k dataset adapter."""

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from torchvision.transforms import functional as tvtf

from .base import register_dataset


# Class mappings
ID2LABEL: Dict[int, str] = {
    0: "Black Background",
    1: "Abdominal Wall",
    2: "Liver",
    3: "Gastrointestinal Tract",
    4: "Fat",
    5: "Grasper",
    6: "Connective Tissue",
    7: "Blood",
    8: "Cystic Duct",
    9: "L-hook Electrocautery",
    10: "Gallbladder",
    11: "Hepatic Vein",
    12: "Liver Ligament",
}

LABEL2ID: Dict[str, int] = {v: k for k, v in ID2LABEL.items()}
LABEL_IDS: Sequence[int] = [k for k in sorted(ID2LABEL) if k != 0]  # 1..12

# Color → class-id mapping for color mask
COLOR_CLASS_MAPPING: Dict[Tuple[int, int, int], int] = {
    (127, 127, 127): 0,
    (210, 140, 140): 1,
    (255, 114, 114): 2,
    (231, 70, 156): 3,
    (186, 183, 75): 4,
    (170, 255, 0): 5,
    (255, 85, 0): 6,
    (255, 0, 0): 7,
    (255, 255, 0): 8,
    (169, 255, 184): 9,
    (255, 160, 165): 10,
    (0, 50, 128): 11,
    (111, 74, 0): 12,
}


class CholecSeg8kAdapter:
    """Dataset adapter for CholecSeg8k."""
    
    def __init__(
        self,
        hf_name: str = "minwoosun/CholecSeg8k",
        hf_revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        """Initialize CholecSeg8k adapter.
        
        Args:
            hf_name: HuggingFace dataset name
            hf_revision: Dataset revision
            cache_dir: Cache directory for dataset
        """
        self.hf_name = hf_name
        self.hf_revision = hf_revision or "main"
        self._dataset = None
        self._cache_dir = cache_dir
    
    def _load_dataset(self) -> None:
        """Lazy load the dataset."""
        if self._dataset is None:
            self._dataset = load_dataset(
                self.hf_name,
                revision=self.hf_revision,
                cache_dir=self._cache_dir,
            )
    
    # Identity & schema
    @property
    def dataset_tag(self) -> str:
        return "cholecseg8k"
    
    @property
    def version(self) -> str:
        return self.hf_revision
    
    @property
    def id2label(self) -> Dict[int, str]:
        return ID2LABEL.copy()
    
    @property
    def label_ids(self) -> Sequence[int]:
        return LABEL_IDS
    
    @property
    def label2id(self) -> Dict[str, int]:
        return LABEL2ID.copy()
    
    @property
    def ignore_index(self) -> int:
        return -1
    
    @property
    def recommended_canvas(self) -> Tuple[int, int]:
        return (768, 768)
    
    # Data access
    def total(self, split: str) -> int:
        self._load_dataset()
        return len(self._dataset[split])
    
    def get_example(self, split: str, index: int) -> Any:
        self._load_dataset()
        return self._dataset[split][index]
    
    def example_to_tensors(self, example: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert example to tensors.
        
        Returns:
            img_t: FloatTensor [3,H,W] in [0,1]
            lab_t: LongTensor [H,W] with class IDs
        """
        img = example['image']        # PIL Image
        mask = example['color_mask']  # PIL Image (color-coded)
        
        # Image -> torch [3,H,W], float32 in [0,1]
        img_t = tvtf.to_tensor(img).contiguous()
        
        # Color mask -> labels -> torch [H,W], int64
        lab_t = self._color_mask_to_labels(mask)
        
        # Safety: spatial dims must match
        _, H, W = img_t.shape
        if lab_t.shape != (H, W):
            raise ValueError(f"Shape mismatch: image [{H},{W}] vs labels {tuple(lab_t.shape)}")
        
        return img_t, lab_t
    
    def _color_mask_to_labels(self, pil_img: Image.Image) -> torch.Tensor:
        """Convert color mask to label tensor.
        
        Args:
            pil_img: PIL Image with color-coded masks
            
        Returns:
            Label tensor [H,W] with class IDs
        """
        alpha = None
        if pil_img.mode == 'RGBA':
            arr = np.array(pil_img)            # H x W x 4
            alpha = arr[..., 3]
            arr = arr[..., :3]
        elif pil_img.mode == 'P':
            arr = np.array(pil_img.convert('RGB'))
        elif pil_img.mode != 'RGB':
            arr = np.array(pil_img.convert('RGB'))
        else:
            arr = np.array(pil_img)
        
        h, w = arr.shape[:2]
        labels = np.full((h, w), self.ignore_index, dtype=np.int64)
        
        # Assign each mapped color
        for (r, g, b), cls in COLOR_CLASS_MAPPING.items():
            m = (arr[..., 0] == r) & (arr[..., 1] == g) & (arr[..., 2] == b)
            labels[m] = cls
        
        if alpha is not None:
            labels[alpha == 0] = self.ignore_index
        
        return torch.from_numpy(labels).contiguous()
    
    # Semantics
    def labels_to_presence_vector(
        self, lab_t: torch.Tensor, min_pixels: int = 1
    ) -> torch.Tensor:
        """Convert label tensor to presence vector.
        
        Args:
            lab_t: Label tensor [H,W]
            min_pixels: Minimum pixels for presence
            
        Returns:
            LongTensor [K=12] with {0,1}
        """
        if isinstance(lab_t, np.ndarray):
            lab_t = torch.from_numpy(lab_t)
        lab_t = lab_t.to(torch.long)
        
        valid = lab_t != self.ignore_index
        flat = lab_t[valid].view(-1)
        
        num_classes = max(ID2LABEL.keys()) + 1  # 13
        counts = torch.zeros(num_classes, dtype=torch.long)
        if flat.numel() > 0:
            counts = torch.bincount(flat, minlength=num_classes)
        
        y = (counts[self.label_ids] >= min_pixels).to(torch.long)  # [12]
        return y
    
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
            strategy: Sampling strategy ('centroid' or 'random')
            
        Returns:
            (x, y) coordinates or None if class not present
        """
        mask = (lab_t == class_id)
        if not mask.any():
            return None
        
        if strategy == "centroid":
            # Compute centroid
            y_coords, x_coords = torch.where(mask)
            x_center = int(x_coords.float().mean().round().item())
            y_center = int(y_coords.float().mean().round().item())
            return (x_center, y_center)
        
        elif strategy == "random":
            # Random sampling
            y_coords, x_coords = torch.where(mask)
            idx = torch.randint(len(x_coords), (1,)).item()
            return (int(x_coords[idx].item()), int(y_coords[idx].item()))
        
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")


@register_dataset("cholecseg8k")
def build_cholecseg8k(**cfg: Any) -> CholecSeg8kAdapter:
    """Build CholecSeg8k dataset adapter.
    
    Args:
        **cfg: Configuration parameters
        
    Returns:
        CholecSeg8k adapter instance
    """
    return CholecSeg8kAdapter(**cfg)