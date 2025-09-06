"""Few-shot example builders for different tasks."""

from typing import List, Dict, Tuple, Optional, Any, Protocol
import numpy as np
import torch
import random
from scipy import ndimage
from abc import ABC, abstractmethod


class DatasetProtocol(Protocol):
    """Protocol for dataset adapters."""
    
    @property
    def label_ids(self) -> List[int]:
        """Get list of label IDs."""
        ...
    
    @property
    def id2label(self) -> Dict[int, str]:
        """Get ID to label mapping."""
        ...
    
    def total(self, split: str) -> int:
        """Get total number of examples in split."""
        ...
    
    def get_example(self, split: str, index: int) -> Any:
        """Get example at index."""
        ...
    
    def example_to_tensors(self, example: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert example to tensors."""
        ...
    
    def sample_point_in_mask(self, lab_t: torch.Tensor, class_id: int, 
                            strategy: str = "centroid") -> Optional[Tuple[int, int]]:
        """Sample point in mask."""
        ...
    
    def get_bounding_boxes(self, lab_t: torch.Tensor, class_id: Optional[int] = None,
                          min_pixels: int = 10) -> Dict[int, List[Tuple[int, int, int, int]]]:
        """Get bounding boxes."""
        ...


class FewShotExampleBuilder(ABC):
    """Abstract base class for few-shot example builders."""
    
    def __init__(
        self,
        dataset: DatasetProtocol,
        n_pos_examples: int = 1,
        n_neg_absent: int = 1,
        n_neg_wrong: int = 1,
        min_pixels: int = 50,
        seed: int = 42
    ):
        """Initialize few-shot example builder.
        
        Args:
            dataset: Dataset adapter
            n_pos_examples: Number of positive examples per class
            n_neg_absent: Number of negative examples where class is absent
            n_neg_wrong: Number of negative examples with wrong answer
            min_pixels: Minimum pixels for presence
            seed: Random seed
        """
        self.dataset = dataset
        self.n_pos_examples = n_pos_examples
        self.n_neg_absent = n_neg_absent
        self.n_neg_wrong = n_neg_wrong
        self.min_pixels = min_pixels
        self.seed = seed
        
        # Get dataset info
        self.label_ids = dataset.label_ids
        self.id2label = dataset.id2label
    
    @abstractmethod
    def build_plan(
        self,
        Y: np.ndarray,
        excluded_indices: List[int],
        split: str = "train"
    ) -> Dict[str, Any]:
        """Build few-shot example plan.
        
        Args:
            Y: Presence matrix [N, K]
            excluded_indices: Indices to exclude (e.g., test set)
            split: Dataset split to use
            
        Returns:
            Dictionary with few-shot plan
        """
        pass
    
    def find_adjacent_organs(
        self,
        lab_t: torch.Tensor,
        target_class_id: int,
        distance_threshold: int = 50
    ) -> List[int]:
        """Find organs adjacent to target organ.
        
        Args:
            lab_t: Label tensor [H, W]
            target_class_id: Target class ID
            distance_threshold: Distance for adjacency
            
        Returns:
            List of adjacent class IDs
        """
        # Create mask for target organ
        target_mask = (lab_t == target_class_id).cpu().numpy()
        
        if not target_mask.any():
            return []
        
        # Dilate target mask to find nearby regions
        dilated = ndimage.binary_dilation(target_mask, iterations=distance_threshold)
        
        # Find what other organs are in the dilated region
        adjacent_organs = []
        for class_id in self.label_ids:
            if class_id == target_class_id:
                continue
            organ_mask = (lab_t == class_id).cpu().numpy()
            if organ_mask.any() and (organ_mask & dilated).any():
                adjacent_organs.append(class_id)
        
        return adjacent_organs


class BoundingBoxFewShotBuilder(FewShotExampleBuilder):
    """Few-shot example builder for bounding box tasks."""
    
    def __init__(self, dataset: DatasetProtocol, min_bbox_size: int = 20, **kwargs):
        """Initialize bounding box few-shot builder.
        
        Args:
            dataset: Dataset adapter
            min_bbox_size: Minimum bbox width/height
            **kwargs: Additional arguments for base class
        """
        super().__init__(dataset, **kwargs)
        self.min_bbox_size = min_bbox_size
    
    def sample_shifted_bbox(
        self,
        true_bbox: Tuple[int, int, int, int],
        image_shape: Tuple[int, int],
        shift_factor: float = 0.5
    ) -> Tuple[int, int, int, int]:
        """Create a shifted/wrong bounding box.
        
        Args:
            true_bbox: True bounding box (x1, y1, x2, y2)
            image_shape: Image shape (H, W)
            shift_factor: Shift amount as fraction of bbox size
            
        Returns:
            Shifted bounding box
        """
        x1, y1, x2, y2 = true_bbox
        width = x2 - x1
        height = y2 - y1
        H, W = image_shape
        
        # Random shift direction
        direction = random.choice(['left', 'right', 'up', 'down'])
        
        if direction == 'left':
            shift_x = -int(width * shift_factor)
            shift_y = random.randint(-height//4, height//4)
        elif direction == 'right':
            shift_x = int(width * shift_factor)
            shift_y = random.randint(-height//4, height//4)
        elif direction == 'up':
            shift_x = random.randint(-width//4, width//4)
            shift_y = -int(height * shift_factor)
        else:  # down
            shift_x = random.randint(-width//4, width//4)
            shift_y = int(height * shift_factor)
        
        # Apply shift and clip to image bounds
        new_x1 = max(0, min(W - width, x1 + shift_x))
        new_y1 = max(0, min(H - height, y1 + shift_y))
        new_x2 = min(W - 1, new_x1 + width)
        new_y2 = min(H - 1, new_y1 + height)
        
        return (int(new_x1), int(new_y1), int(new_x2), int(new_y2))
    
    def build_plan(
        self,
        Y: np.ndarray,
        excluded_indices: List[int],
        split: str = "train"
    ) -> Dict[str, Any]:
        """Build bounding box few-shot plan."""
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        n_total = self.dataset.total(split)
        n_in_matrix = Y.shape[0]  # Actual samples in presence matrix
        plan = {'split': split, 'task': 'bounding_box', 'plan': {}}
        
        # Exclude test samples from selection pool
        # Only consider indices that are within the presence matrix bounds
        excluded = set(excluded_indices)
        available_indices = [i for i in range(min(n_total, n_in_matrix)) if i not in excluded]
        
        for class_idx, class_id in enumerate(self.label_ids):
            organ_name = self.id2label[class_id]
            
            # Find positive and negative samples (only within matrix bounds)
            pos_indices = [i for i in available_indices if i < n_in_matrix and Y[i, class_idx] == 1]
            neg_indices = [i for i in available_indices if i < n_in_matrix and Y[i, class_idx] == 0]
            
            organ_info = {
                'name': organ_name,
                'class_id': int(class_id),
                'pos_available': len(pos_indices),
                'neg_available': len(neg_indices),
                'positives': [],
                'negatives_absent': [],
                'negatives_wrong_bbox': []
            }
            
            # (1) Positive examples: correct bounding boxes
            if len(pos_indices) > 0:
                selected_pos = np.random.choice(
                    pos_indices,
                    min(self.n_pos_examples, len(pos_indices)),
                    replace=False
                )
                for idx in selected_pos:
                    example = self.dataset.get_example(split, idx)
                    img_t, lab_t = self.dataset.example_to_tensors(example)
                    
                    # Get bounding boxes for this organ
                    bboxes = self.dataset.get_bounding_boxes(
                        lab_t, class_id, min_pixels=self.min_pixels
                    )
                    if class_id in bboxes and len(bboxes[class_id]) > 0:
                        organ_info['positives'].append({
                            'idx': int(idx),
                            'bboxes': bboxes[class_id],
                            'frame_id': example.get('frame_id', f"{split}_{idx}"),
                            'num_regions': len(bboxes[class_id])
                        })
            
            # (2) Negative examples: organ is absent
            if len(neg_indices) > 0:
                selected_neg = np.random.choice(
                    neg_indices,
                    min(self.n_neg_absent, len(neg_indices)),
                    replace=False
                )
                for idx in selected_neg:
                    example = self.dataset.get_example(split, idx)
                    organ_info['negatives_absent'].append({
                        'idx': int(idx),
                        'bboxes': None,
                        'frame_id': example.get('frame_id', f"{split}_{idx}")
                    })
            
            # (3) Negative examples: wrong bbox
            if len(pos_indices) > 0:
                remaining_pos = [i for i in pos_indices if i not in selected_pos]
                if len(remaining_pos) > 0:
                    n_wrong_needed = self.n_neg_wrong
                    if len(neg_indices) == 0:
                        n_wrong_needed += 1
                    
                    selected_wrong = np.random.choice(
                        remaining_pos,
                        min(n_wrong_needed, len(remaining_pos)),
                        replace=False
                    )
                    for idx in selected_wrong:
                        example = self.dataset.get_example(split, idx)
                        img_t, lab_t = self.dataset.example_to_tensors(example)
                        
                        # Get true bbox
                        true_bboxes = self.dataset.get_bounding_boxes(
                            lab_t, class_id, min_pixels=self.min_pixels
                        )
                        
                        if class_id not in true_bboxes or len(true_bboxes[class_id]) == 0:
                            continue
                        
                        # Find adjacent organs
                        adjacent = self.find_adjacent_organs(lab_t, class_id)
                        
                        wrong_bboxes = None
                        wrong_type = 'unknown'
                        
                        if adjacent:
                            # Use bbox of adjacent organ
                            adj_class = random.choice(adjacent)
                            adj_bboxes = self.dataset.get_bounding_boxes(
                                lab_t, adj_class, min_pixels=self.min_pixels
                            )
                            if adj_class in adj_bboxes and len(adj_bboxes[adj_class]) > 0:
                                wrong_bboxes = adj_bboxes[adj_class]
                                wrong_type = f'adjacent_{self.id2label[adj_class]}'
                        
                        if wrong_bboxes is None:
                            # Create shifted bbox
                            true_bbox = true_bboxes[class_id][0]
                            shifted_bbox = self.sample_shifted_bbox(true_bbox, lab_t.shape)
                            wrong_bboxes = [shifted_bbox]
                            wrong_type = 'shifted'
                        
                        organ_info['negatives_wrong_bbox'].append({
                            'idx': int(idx),
                            'bboxes': wrong_bboxes,
                            'frame_id': example.get('frame_id', f"{split}_{idx}"),
                            'wrong_type': wrong_type
                        })
            
            plan['plan'][str(class_id)] = organ_info
        
        # Add summary
        plan['summary'] = {
            'total_excluded': len(excluded_indices),
            'total_pool_size': len(available_indices),
            'task_type': 'bounding_box_detection',
            'description': 'Few-shot examples for bounding box prediction task'
        }
        
        return plan


class PointingFewShotBuilder(FewShotExampleBuilder):
    """Few-shot example builder for pointing tasks."""
    
    def build_plan(
        self,
        Y: np.ndarray,
        excluded_indices: List[int],
        split: str = "train"
    ) -> Dict[str, Any]:
        """Build pointing few-shot plan."""
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        n_total = self.dataset.total(split)
        n_in_matrix = Y.shape[0]  # Actual samples in presence matrix
        plan = {'split': split, 'task': 'pointing', 'plan': {}}
        
        # Exclude test samples from selection pool
        # Only consider indices that are within the presence matrix bounds
        excluded = set(excluded_indices)
        available_indices = [i for i in range(min(n_total, n_in_matrix)) if i not in excluded]
        
        for class_idx, class_id in enumerate(self.label_ids):
            organ_name = self.id2label[class_id]
            
            # Find positive and negative samples (only within matrix bounds)
            pos_indices = [i for i in available_indices if i < n_in_matrix and Y[i, class_idx] == 1]
            neg_indices = [i for i in available_indices if i < n_in_matrix and Y[i, class_idx] == 0]
            
            organ_info = {
                'name': organ_name,
                'class_id': int(class_id),
                'pos_available': len(pos_indices),
                'neg_available': len(neg_indices),
                'positives': [],
                'negatives_absent': [],
                'negatives_wrong_point': []
            }
            
            # Select positive examples with correct points
            if len(pos_indices) > 0:
                selected_pos = np.random.choice(
                    pos_indices,
                    min(self.n_pos_examples, len(pos_indices)),
                    replace=False
                )
                for idx in selected_pos:
                    example = self.dataset.get_example(split, idx)
                    img_t, lab_t = self.dataset.example_to_tensors(example)
                    point = self.dataset.sample_point_in_mask(
                        lab_t, class_id, strategy='centroid'
                    )
                    if point:
                        organ_info['positives'].append({
                            'idx': int(idx),
                            'point': point,
                            'frame_id': example.get('frame_id', f"{split}_{idx}")
                        })
            
            # Select negative examples where organ is absent
            if len(neg_indices) > 0:
                selected_neg = np.random.choice(
                    neg_indices,
                    min(self.n_neg_absent, len(neg_indices)),
                    replace=False
                )
                for idx in selected_neg:
                    example = self.dataset.get_example(split, idx)
                    organ_info['negatives_absent'].append({
                        'idx': int(idx),
                        'point': None,
                        'frame_id': example.get('frame_id', f"{split}_{idx}")
                    })
            
            # Select negative examples with wrong points
            if len(pos_indices) > 0:
                remaining_pos = [i for i in pos_indices if i not in selected_pos]
                if len(remaining_pos) > 0:
                    selected_wrong = np.random.choice(
                        remaining_pos,
                        min(self.n_neg_wrong, len(remaining_pos)),
                        replace=False
                    )
                    for idx in selected_wrong:
                        example = self.dataset.get_example(split, idx)
                        img_t, lab_t = self.dataset.example_to_tensors(example)
                        
                        # Find adjacent organs
                        adjacent = self.find_adjacent_organs(lab_t, class_id)
                        
                        wrong_point = None
                        if adjacent:
                            # Sample point from adjacent organ
                            adj_class = random.choice(adjacent)
                            wrong_point = self.dataset.sample_point_in_mask(
                                lab_t, adj_class, strategy='centroid'
                            )
                        
                        if wrong_point is None:
                            # Create near-miss point by shifting
                            true_point = self.dataset.sample_point_in_mask(
                                lab_t, class_id, strategy='centroid'
                            )
                            if true_point:
                                H, W = lab_t.shape
                                shift_dist = random.randint(30, 100)
                                angle = random.uniform(0, 2 * np.pi)
                                shift_x = int(shift_dist * np.cos(angle))
                                shift_y = int(shift_dist * np.sin(angle))
                                wrong_x = max(0, min(W-1, true_point[0] + shift_x))
                                wrong_y = max(0, min(H-1, true_point[1] + shift_y))
                                wrong_point = (wrong_x, wrong_y)
                        
                        if wrong_point:
                            organ_info['negatives_wrong_point'].append({
                                'idx': int(idx),
                                'point': wrong_point,
                                'frame_id': example.get('frame_id', f"{split}_{idx}"),
                                'note': 'wrong_location'
                            })
            
            plan['plan'][str(class_id)] = organ_info
        
        # Add summary
        plan['summary'] = {
            'total_excluded': len(excluded_indices),
            'total_pool_size': len(available_indices),
            'task_type': 'pointing',
            'description': 'Few-shot examples for pointing task'
        }
        
        return plan