"""CholecOrgans dataset adapter for local files."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import glob

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as tvtf
from scipy import ndimage
from tqdm import tqdm

from .base import register_dataset


# Class mappings for organs (matches the original notebook)
ID2LABEL: Dict[int, str] = {
    0: "Background",
    1: "Liver",
    2: "Gallbladder",
    3: "Hepatocystic Triangle"
}

LABEL2ID: Dict[str, int] = {v: k for k, v in ID2LABEL.items()}
LABEL_IDS: Sequence[int] = [1, 2, 3]  # Excluding background

# Video globs for splitting (from original notebook)
VIDEO_GLOBS_PUBLIC = (
    [f"cholec80_video{i:02d}_*" for i in range(1, 81)] +
    [f"M2CCAI2016_video{i}_*" for i in range(81, 122)]
)

VIDEO_GLOBS_PRIVATE = (
    [f"AdnanSet_LC_{i}_*" for i in range(1, 165)] +
    [f"AminSet_LC_{i}_*" for i in range(1, 11)] +
    ["HokkaidoSet_LC_1_*", "HokkaidoSet_LC_2_*"] +
    [f"UTSWSet_Case_{i}_*" for i in range(1, 13)] +
    ["WashUSet_LC_01_*"]
)

VIDEO_GLOBS_DICT = {
    'public': VIDEO_GLOBS_PUBLIC,
    'private': VIDEO_GLOBS_PRIVATE
}


class CholecOrgansAdapter:
    """Dataset adapter for CholecOrgans dataset with organ labels."""
    
    def __init__(
        self,
        data_dir: str = "/shared_data0/weiqiuy/real_drs/data/abdomen_exlib",
        images_dir: str = "images",
        organ_labels_dir: str = "organ_labels",
        video_globs: str = 'public',
        train_ratio: float = 0.8,
        gen_seed: int = 56,
        train_val_seed: int = 0,
        image_height: int = 384,
        image_width: int = 640,
    ):
        """Initialize CholecOrgans adapter.
        
        Args:
            data_dir: Root directory containing images and labels
            images_dir: Subdirectory name for images
            organ_labels_dir: Subdirectory name for organ labels
            video_globs: Video source ('public', 'private', or list)
            train_ratio: Fraction of videos for training
            gen_seed: Random seed for train/test splitting (default: 56)
            train_val_seed: Random seed for train/val splitting (default: 0)
            image_height: Target image height
            image_width: Target image width
        """
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / images_dir
        self.organ_labels_dir = self.data_dir / organ_labels_dir
        
        if not self.images_dir.exists():
            raise ValueError(f"Images directory {self.images_dir} does not exist")
        if not self.organ_labels_dir.exists():
            raise ValueError(f"Organ labels directory {self.organ_labels_dir} does not exist")
        
        self.train_ratio = train_ratio
        self.gen_seed = gen_seed
        self.train_val_seed = train_val_seed
        self.image_height = image_height
        self.image_width = image_width
        
        # Get video globs
        if isinstance(video_globs, str):
            if video_globs in VIDEO_GLOBS_DICT:
                self.video_globs = VIDEO_GLOBS_DICT[video_globs]
            else:
                raise ValueError(f'video_globs {video_globs} not in {VIDEO_GLOBS_DICT.keys()}')
        elif isinstance(video_globs, list):
            self.video_globs = video_globs
        else:
            raise TypeError(f'video_globs must be str or list, got {type(video_globs)}')
        
        self._examples = None
        self._splits = None
        self._index_dataset()
    
    def _index_dataset(self) -> None:
        """Index all examples with video-based splitting matching original notebook."""
        # Use torch generator for consistency with original code
        gen = torch.Generator()
        gen.manual_seed(self.gen_seed)
        
        # Split videos into train/test
        num_all = len(self.video_globs)
        num_train = int(num_all * self.train_ratio)
        perm = torch.randperm(num_all, generator=gen)
        
        train_video_indices = perm[:num_train]
        test_video_indices = perm[num_train:]
        
        # Collect image files for each split
        train_files = []
        test_files = []
        
        print("Indexing training videos...")
        for i in tqdm(train_video_indices.tolist(), desc="  Train videos"):
            pattern = str(self.images_dir / self.video_globs[i])
            files = glob.glob(pattern)
            train_files.extend([os.path.basename(f) for f in files])
        
        print("Indexing test videos...")
        for i in tqdm(test_video_indices.tolist(), desc="  Test videos"):
            pattern = str(self.images_dir / self.video_globs[i])
            files = glob.glob(pattern)
            test_files.extend([os.path.basename(f) for f in files])
        
        train_files = sorted(train_files)
        test_files = sorted(test_files)
        
        # Build examples list
        all_files = train_files + test_files
        self._examples = []
        
        print("Building examples list...")
        for filename in tqdm(all_files, desc="  Checking files"):
            image_path = self.images_dir / filename
            label_path = self.organ_labels_dir / filename
            
            if image_path.exists() and label_path.exists():
                self._examples.append({
                    'filename': filename,
                    'image_path': str(image_path),
                    'label_path': str(label_path),
                })
        
        # Create index mappings for splits
        train_indices = []
        test_indices = []
        
        for idx, example in enumerate(self._examples):
            if example['filename'] in train_files:
                train_indices.append(idx)
            elif example['filename'] in test_files:
                test_indices.append(idx)
        
        # No validation split in original - use small portion of train
        # Match original behavior: train/val split with separate seed
        gen.manual_seed(self.train_val_seed)
        
        num_train_examples = len(train_indices)
        num_train_split = int(num_train_examples * 0.9)  # 90% train, 10% val
        perm = torch.randperm(num_train_examples, generator=gen)
        
        train_indices_shuffled = [train_indices[i] for i in perm]
        final_train = train_indices_shuffled[:num_train_split]
        final_val = train_indices_shuffled[num_train_split:]
        
        self._splits = {
            "train": final_train,
            "validation": final_val,
            "test": test_indices,
        }
        
        # Print split information
        print(f"CholecOrgans dataset indexed:")
        print(f"  Total examples: {len(self._examples)}")
        print(f"  Train: {len(self._splits['train'])} examples")
        print(f"  Validation: {len(self._splits['validation'])} examples")
        print(f"  Test: {len(self._splits['test'])} examples")
    
    # Identity & schema
    @property
    def dataset_tag(self) -> str:
        return "cholec_organs"
    
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
        return (self.image_height, self.image_width)
    
    # Data access
    def total(self, split: str) -> int:
        if split not in self._splits:
            raise ValueError(f"Unknown split: {split}")
        return len(self._splits[split])
    
    def get_example(self, split: str, index: int) -> Any:
        if split not in self._splits:
            raise ValueError(f"Unknown split: {split}")
        
        example_idx = self._splits[split][index]
        example_data = self._examples[example_idx]
        
        # Load images
        image = Image.open(example_data["image_path"]).convert("RGB")
        label = Image.open(example_data["label_path"]).convert("L")
        
        # Resize to target dimensions
        image = image.resize((self.image_width, self.image_height), Image.BILINEAR)
        label = label.resize((self.image_width, self.image_height), Image.NEAREST)
        
        return {
            "image": image,
            "organ_label": label,
            "filename": example_data["filename"],
        }
    
    def example_to_tensors(self, example: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert example to tensors.
        
        Returns:
            img_t: FloatTensor [3,H,W] in [0,1]
            lab_t: LongTensor [H,W] with class IDs
        """
        img = example['image']
        label = example['organ_label']
        
        # Image -> torch [3,H,W], float32 in [0,1]
        img_t = tvtf.to_tensor(img).contiguous()
        
        # Label -> torch [H,W], int64
        # Convert PIL Image to numpy, then to tensor
        label_np = np.array(label, dtype=np.int64)
        # The label values are already 0-3 (background, liver, gallbladder, hepatocystic)
        lab_t = torch.from_numpy(label_np).contiguous()
        
        # Safety check
        _, H, W = img_t.shape
        if lab_t.shape != (H, W):
            raise ValueError(f"Shape mismatch: image [{H},{W}] vs labels {tuple(lab_t.shape)}")
        
        return img_t, lab_t
    
    # Semantics
    def labels_to_presence_vector(
        self, lab_t: torch.Tensor, min_pixels: int = 1
    ) -> torch.Tensor:
        """Convert label tensor to presence vector.
        
        Args:
            lab_t: Label tensor [H,W]
            min_pixels: Minimum pixels for presence
            
        Returns:
            LongTensor [K=3] with {0,1} (excluding background)
        """
        if isinstance(lab_t, np.ndarray):
            lab_t = torch.from_numpy(lab_t)
        lab_t = lab_t.to(torch.long)
        
        flat = lab_t.view(-1)
        
        # Count pixels for each class
        num_classes = max(ID2LABEL.keys()) + 1  # 4
        counts = torch.zeros(num_classes, dtype=torch.long)
        if flat.numel() > 0:
            counts = torch.bincount(flat, minlength=num_classes)
        
        # Return presence for non-background classes
        y = (counts[self.label_ids] >= min_pixels).to(torch.long)  # [3]
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
        
        Args:
            lab_t: Label tensor [H,W]
            class_id: Optional specific class to get boxes for
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


@register_dataset("cholec_organs")
def build_cholec_organs(**cfg: Any) -> CholecOrgansAdapter:
    """Build CholecOrgans dataset adapter.
    
    Args:
        **cfg: Configuration parameters
        
    Returns:
        CholecOrgans adapter instance
    """
    return CholecOrgansAdapter(**cfg)