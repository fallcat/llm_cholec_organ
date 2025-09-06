"""CholecSeg8k dataset adapter for local files."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as tvtf
from scipy import ndimage

from .base import register_dataset


# Class mappings (same as original)
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


class CholecSeg8kLocalAdapter:
    """Dataset adapter for locally stored CholecSeg8k."""
    
    def __init__(
        self,
        data_dir: str = "/shared_data0/weiqiuy/datasets/cholecseg8k",
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
        seed: int = 42,
    ):
        """Initialize CholecSeg8k local adapter.
        
        Args:
            data_dir: Root directory containing video folders
            train_split: Fraction of data for training
            val_split: Fraction of data for validation 
            test_split: Fraction of data for testing
            seed: Random seed for splitting
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise ValueError(f"Data directory {data_dir} does not exist")
        
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed
        
        self._examples = None
        self._splits = None
        self._index_dataset()
    
    def _index_dataset(self) -> None:
        """Index all examples in the dataset with video-based splitting."""
        examples = []
        video_to_examples = {}  # Map video_id to list of example indices
        
        # Find all video directories
        video_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir() and d.name.startswith("video")])
        
        example_idx = 0
        for video_dir in video_dirs:
            video_id = video_dir.name
            video_to_examples[video_id] = []
            
            # Find all frame directories within each video
            frame_dirs = sorted([d for d in video_dir.iterdir() if d.is_dir()])
            
            for frame_dir in frame_dirs:
                # Find all frame files in this directory
                frame_files = sorted(frame_dir.glob("frame_*_endo.png"))
                
                for frame_file in frame_files:
                    # Extract frame number from filename
                    frame_name = frame_file.stem  # e.g., "frame_100_endo"
                    frame_num = frame_name.split("_")[1]
                    
                    # Construct paths for image and masks
                    image_path = frame_file
                    color_mask_path = frame_dir / f"frame_{frame_num}_endo_color_mask.png"
                    mask_path = frame_dir / f"frame_{frame_num}_endo_mask.png"
                    watershed_mask_path = frame_dir / f"frame_{frame_num}_endo_watershed_mask.png"
                    
                    # Only add if color mask exists (required for labels)
                    if color_mask_path.exists():
                        examples.append({
                            "image_path": str(image_path),
                            "color_mask_path": str(color_mask_path),
                            "mask_path": str(mask_path) if mask_path.exists() else None,
                            "watershed_mask_path": str(watershed_mask_path) if watershed_mask_path.exists() else None,
                            "video_id": video_id,
                            "frame_id": f"{video_dir.name}_{frame_dir.name}_{frame_num}",
                        })
                        video_to_examples[video_id].append(example_idx)
                        example_idx += 1
        
        self._examples = examples
        
        # Create video-based train/val/test splits
        np.random.seed(self.seed)
        
        # Get list of videos and shuffle them
        video_ids = sorted(video_to_examples.keys())
        n_videos = len(video_ids)
        shuffled_video_indices = np.random.permutation(n_videos)
        
        # Calculate number of videos for each split
        # Ensure at least 1 video per split if we have enough videos
        n_train_videos = max(1, int(n_videos * self.train_split))
        n_val_videos = max(1, int(n_videos * self.val_split)) if n_videos >= 3 else 0
        n_test_videos = n_videos - n_train_videos - n_val_videos
        
        # Ensure test set has at least 1 video if possible
        if n_test_videos == 0 and n_videos >= 3:
            # Adjust train set to ensure test has at least 1 video
            n_train_videos = n_videos - 2  # 1 for val, 1 for test
            n_val_videos = 1
            n_test_videos = 1
        
        # Assign videos to splits
        train_videos = [video_ids[i] for i in shuffled_video_indices[:n_train_videos]]
        val_videos = [video_ids[i] for i in shuffled_video_indices[n_train_videos:n_train_videos + n_val_videos]]
        test_videos = [video_ids[i] for i in shuffled_video_indices[n_train_videos + n_val_videos:]]
        
        # Collect example indices for each split
        train_indices = []
        val_indices = []
        test_indices = []
        
        for video_id in train_videos:
            train_indices.extend(video_to_examples[video_id])
        for video_id in val_videos:
            val_indices.extend(video_to_examples[video_id])
        for video_id in test_videos:
            test_indices.extend(video_to_examples[video_id])
        
        self._splits = {
            "train": train_indices,
            "validation": val_indices,
            "test": test_indices,
        }
        
        # Store video assignments for reference
        self._video_splits = {
            "train": train_videos,
            "validation": val_videos,
            "test": test_videos,
        }
        
        # Print split information
        print(f"Video-based split created:")
        print(f"  Total videos: {n_videos}")
        print(f"  Train: {len(train_videos)} videos, {len(train_indices)} frames")
        print(f"  Validation: {len(val_videos)} videos, {len(val_indices)} frames")
        print(f"  Test: {len(test_videos)} videos, {len(test_indices)} frames")
        print(f"  Train videos: {', '.join(train_videos)}")
        print(f"  Val videos: {', '.join(val_videos)}")
        print(f"  Test videos: {', '.join(test_videos)}")
    
    # Identity & schema
    @property
    def dataset_tag(self) -> str:
        return "cholecseg8k_local"
    
    @property
    def version(self) -> str:
        return "local"
    
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
        if split not in self._splits:
            raise ValueError(f"Unknown split: {split}")
        return len(self._splits[split])
    
    def get_video_splits(self) -> Dict[str, List[str]]:
        """Get the video IDs assigned to each split."""
        return self._video_splits.copy()
    
    def get_frame_counts_by_video(self, split: str) -> Dict[str, int]:
        """Get frame counts for each video in a split without loading images.
        
        Args:
            split: Split name ('train', 'validation', or 'test')
            
        Returns:
            Dictionary mapping video_id to frame count
        """
        if split not in self._splits:
            raise ValueError(f"Unknown split: {split}")
        
        frame_counts = {}
        for idx in self._splits[split]:
            video_id = self._examples[idx]['video_id']
            frame_counts[video_id] = frame_counts.get(video_id, 0) + 1
        
        return frame_counts
    
    def get_example(self, split: str, index: int) -> Any:
        if split not in self._splits:
            raise ValueError(f"Unknown split: {split}")
        
        example_idx = self._splits[split][index]
        example_data = self._examples[example_idx]
        
        # Load images
        image = Image.open(example_data["image_path"]).convert("RGB")
        color_mask = Image.open(example_data["color_mask_path"])
        
        # Return in same format as HuggingFace version
        return {
            "image": image,
            "color_mask": color_mask,
            "mask_path": example_data["mask_path"],
            "watershed_mask_path": example_data["watershed_mask_path"],
            "video_id": example_data["video_id"],
            "frame_id": example_data["frame_id"],
        }
    
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
    
    def get_bounding_boxes(
        self,
        lab_t: torch.Tensor,
        class_id: Optional[int] = None,
        min_pixels: int = 10
    ) -> Dict[int, List[Tuple[int, int, int, int]]]:
        """Get bounding boxes for segmented regions.
        
        For each organ class, finds all connected components and returns
        a bounding box for each component.
        
        Args:
            lab_t: Label tensor [H,W]
            class_id: Optional specific class to get boxes for. If None, gets boxes for all classes.
            min_pixels: Minimum pixels for a component to be included
            
        Returns:
            Dictionary mapping class_id to list of bounding boxes.
            Each bounding box is (x_min, y_min, x_max, y_max).
        """
        if isinstance(lab_t, torch.Tensor):
            lab_np = lab_t.numpy()
        else:
            lab_np = lab_t
        
        result = {}
        
        # Determine which classes to process
        if class_id is not None:
            class_ids = [class_id]
        else:
            class_ids = self.label_ids
        
        for cid in class_ids:
            # Get binary mask for this class
            mask = (lab_np == cid).astype(np.uint8)
            
            if not mask.any():
                continue
            
            # Find connected components
            labeled_array, num_features = ndimage.label(mask)
            
            boxes = []
            for component_id in range(1, num_features + 1):
                # Get component mask
                component_mask = (labeled_array == component_id)
                
                # Check if component is large enough
                if component_mask.sum() < min_pixels:
                    continue
                
                # Find bounding box
                y_coords, x_coords = np.where(component_mask)
                
                if len(x_coords) > 0:
                    x_min = int(x_coords.min())
                    x_max = int(x_coords.max())
                    y_min = int(y_coords.min())
                    y_max = int(y_coords.max())
                    
                    boxes.append((x_min, y_min, x_max, y_max))
            
            if boxes:
                result[cid] = boxes
        
        return result
    
    def get_bounding_boxes_with_labels(
        self,
        lab_t: torch.Tensor,
        min_pixels: int = 10
    ) -> List[Dict[str, Any]]:
        """Get bounding boxes with organ labels.
        
        Args:
            lab_t: Label tensor [H,W]
            min_pixels: Minimum pixels for a component to be included
            
        Returns:
            List of dictionaries, each containing:
            - 'class_id': int
            - 'class_name': str
            - 'bbox': (x_min, y_min, x_max, y_max)
            - 'area': int (number of pixels)
        """
        if isinstance(lab_t, torch.Tensor):
            lab_np = lab_t.numpy()
        else:
            lab_np = lab_t
        
        results = []
        
        for cid in self.label_ids:
            # Get binary mask for this class
            mask = (lab_np == cid).astype(np.uint8)
            
            if not mask.any():
                continue
            
            # Find connected components
            labeled_array, num_features = ndimage.label(mask)
            
            for component_id in range(1, num_features + 1):
                # Get component mask
                component_mask = (labeled_array == component_id)
                area = int(component_mask.sum())
                
                # Check if component is large enough
                if area < min_pixels:
                    continue
                
                # Find bounding box
                y_coords, x_coords = np.where(component_mask)
                
                if len(x_coords) > 0:
                    x_min = int(x_coords.min())
                    x_max = int(x_coords.max())
                    y_min = int(y_coords.min())
                    y_max = int(y_coords.max())
                    
                    results.append({
                        'class_id': cid,
                        'class_name': self.id2label[cid],
                        'bbox': (x_min, y_min, x_max, y_max),
                        'area': area
                    })
        
        return results


@register_dataset("cholecseg8k_local")
def build_cholecseg8k_local(**cfg: Any) -> CholecSeg8kLocalAdapter:
    """Build CholecSeg8k local dataset adapter.
    
    Args:
        **cfg: Configuration parameters
        
    Returns:
        CholecSeg8k local adapter instance
    """
    return CholecSeg8kLocalAdapter(**cfg)