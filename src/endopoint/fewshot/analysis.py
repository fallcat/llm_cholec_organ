"""Analysis utilities for few-shot selection and dataset balance."""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from pathlib import Path


class DatasetBalanceAnalyzer:
    """Analyzer for dataset balance and selection quality."""
    
    def __init__(self, dataset, presence_matrix: np.ndarray):
        """Initialize analyzer.
        
        Args:
            dataset: Dataset adapter
            presence_matrix: Presence matrix [N, K]
        """
        self.dataset = dataset
        self.Y = presence_matrix
        self.n_samples, self.n_classes = presence_matrix.shape
        
        # Compute base statistics
        self.class_counts = self.Y.sum(axis=0)
        self.class_percentages = (self.class_counts / self.n_samples) * 100
    
    def get_class_distribution(self) -> Dict[str, Any]:
        """Get class distribution statistics.
        
        Returns:
            Dictionary with class distribution info
        """
        distribution = {}
        for i, cid in enumerate(self.dataset.label_ids):
            name = self.dataset.id2label[cid]
            distribution[name] = {
                'class_id': int(cid),
                'count': int(self.class_counts[i]),
                'percentage': float(self.class_percentages[i])
            }
        return distribution
    
    def identify_rare_classes(self, threshold: float = 20.0) -> List[str]:
        """Identify rare classes below threshold.
        
        Args:
            threshold: Percentage threshold for rare classes
            
        Returns:
            List of rare class names
        """
        rare_classes = []
        for i, cid in enumerate(self.dataset.label_ids):
            if self.class_percentages[i] < threshold:
                rare_classes.append(self.dataset.id2label[cid])
        return rare_classes
    
    def compare_distributions(
        self,
        selected_indices: List[int]
    ) -> Dict[str, Any]:
        """Compare original vs selected distributions.
        
        Args:
            selected_indices: Indices of selected samples
            
        Returns:
            Dictionary with comparison metrics
        """
        # Get selected distribution
        Y_selected = self.Y[selected_indices]
        selected_counts = Y_selected.sum(axis=0)
        selected_percentages = (selected_counts / len(selected_indices)) * 100
        
        # Build comparison
        comparison = {
            'n_selected': len(selected_indices),
            'classes': {}
        }
        
        for i, cid in enumerate(self.dataset.label_ids):
            name = self.dataset.id2label[cid]
            orig_pct = self.class_percentages[i]
            sel_count = int(selected_counts[i])
            sel_pct = float(selected_percentages[i])
            improvement = sel_pct - orig_pct
            
            # Determine change type
            if improvement > 5:
                change = "boosted"
            elif improvement < -5:
                change = "reduced"
            else:
                change = "similar"
            
            comparison['classes'][name] = {
                'original_count': int(self.class_counts[i]),
                'original_pct': float(orig_pct),
                'selected_count': sel_count,
                'selected_pct': sel_pct,
                'improvement': float(improvement),
                'change_type': change
            }
        
        # Calculate balance metrics
        orig_std = np.std(self.class_percentages)
        selected_std = np.std(selected_percentages)
        balance_improvement = ((orig_std - selected_std) / orig_std) * 100 if orig_std > 0 else 0
        
        comparison['metrics'] = {
            'original_stddev': float(orig_std),
            'selected_stddev': float(selected_std),
            'balance_improvement_pct': float(balance_improvement)
        }
        
        return comparison
    
    def print_distribution_report(self, selected_indices: Optional[List[int]] = None):
        """Print formatted distribution report.
        
        Args:
            selected_indices: Optional selected indices for comparison
        """
        print(f"\n📊 Class Distribution Report")
        print("="*60)
        
        # Original distribution
        print(f"\nOriginal Distribution ({self.n_samples} samples):")
        for i, cid in enumerate(self.dataset.label_ids):
            name = self.dataset.id2label[cid]
            count = self.class_counts[i]
            pct = self.class_percentages[i]
            print(f"  {name:30} {count:6d} samples ({pct:5.1f}%)")
        
        # Identify rare classes
        rare_classes = self.identify_rare_classes()
        if rare_classes:
            print(f"\n🔴 Rare classes (<20%): {rare_classes}")
        
        # Selected distribution comparison
        if selected_indices is not None:
            comparison = self.compare_distributions(selected_indices)
            
            print(f"\n📊 Selected Distribution ({comparison['n_selected']} samples):")
            for name, stats in comparison['classes'].items():
                marker = {
                    'boosted': '⬆️',
                    'reduced': '⬇️', 
                    'similar': '➡️'
                }[stats['change_type']]
                
                print(f"  {name:30} {stats['selected_count']:3d} samples "
                      f"({stats['selected_pct']:5.1f}%) {marker} "
                      f"(was {stats['original_pct']:5.1f}%)")
            
            # Balance metrics
            metrics = comparison['metrics']
            print(f"\n📈 Balance Metrics:")
            print(f"  Original StdDev: {metrics['original_stddev']:.2f}%")
            print(f"  Selected StdDev: {metrics['selected_stddev']:.2f}%")
            print(f"  Balance Improvement: {metrics['balance_improvement_pct']:.1f}%")


def auto_configure_selection_params(
    n_classes: int,
    n_test_samples: int
) -> Dict[str, Any]:
    """Auto-configure selection parameters based on dataset.
    
    Args:
        n_classes: Number of classes in dataset
        n_test_samples: Target number of test samples
        
    Returns:
        Dictionary with configured parameters
    """
    params = {
        'rare_top_k': min(4, max(1, n_classes // 3)),
        'min_quota_rare': int(0.30 * n_test_samples),  # 30% of test samples
        'max_cap_frac': 0.70
    }
    return params


def summarize_fewshot_plan(plan: Dict[str, Any]) -> Dict[str, int]:
    """Summarize few-shot plan statistics.
    
    Args:
        plan: Few-shot plan dictionary
        
    Returns:
        Dictionary with counts
    """
    if not plan or 'plan' not in plan:
        return {}
    
    total_pos = 0
    total_neg_absent = 0
    total_neg_wrong = 0
    multi_region = 0
    
    for info in plan['plan'].values():
        total_pos += len(info.get('positives', []))
        total_neg_absent += len(info.get('negatives_absent', []))
        
        # Handle both pointing and bbox variants
        if 'negatives_wrong_point' in info:
            total_neg_wrong += len(info['negatives_wrong_point'])
        elif 'negatives_wrong_bbox' in info:
            total_neg_wrong += len(info['negatives_wrong_bbox'])
            
        # Count multi-region examples (bbox only)
        for pos in info.get('positives', []):
            if pos.get('num_regions', 1) > 1:
                multi_region += 1
    
    return {
        'total_positive': total_pos,
        'total_negative_absent': total_neg_absent,
        'total_negative_wrong': total_neg_wrong,
        'total_multi_region': multi_region,
        'total_examples': total_pos + total_neg_absent + total_neg_wrong
    }