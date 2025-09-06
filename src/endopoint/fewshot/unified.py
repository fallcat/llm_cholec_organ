"""Unified few-shot example selection for all datasets."""

from typing import Dict, List, Tuple, Optional, Any, Union
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

from .balanced_selection import select_balanced_with_caps, select_balanced_simple
from .example_builder import BoundingBoxFewShotBuilder, PointingFewShotBuilder
from .analysis import (
    DatasetBalanceAnalyzer, 
    auto_configure_selection_params,
    summarize_fewshot_plan
)


class UnifiedFewShotSelector:
    """Unified few-shot selector for any dataset that follows the protocol.
    
    This class works with any dataset that implements:
    - label_ids: List of class IDs
    - id2label: Dictionary mapping ID to label name
    - total(split): Get number of examples in split
    - get_example(split, index): Get example at index
    - example_to_tensors(example): Convert to tensors
    - labels_to_presence_vector(label_tensor): Get presence vector
    - sample_point_in_mask(...): Sample point for pointing task
    - get_bounding_boxes(...): Get bounding boxes
    """
    
    def __init__(
        self,
        dataset,
        output_dir: Optional[Union[str, Path]] = None,
        n_test_samples: int = 100,
        n_pos_examples: int = 1,
        n_neg_absent: int = 1,
        n_neg_wrong: int = 1,
        min_pixels: int = 50,
        seed: int = 42,
        cache_enabled: bool = True
    ):
        """Initialize unified few-shot selector.
        
        Args:
            dataset: Any dataset adapter with required methods
            output_dir: Directory to save results (auto-generated if None)
            n_test_samples: Number of test samples to select
            n_pos_examples: Positive examples per class
            n_neg_absent: Negative examples where class is absent
            n_neg_wrong: Negative examples with wrong answer
            min_pixels: Minimum pixels for presence
            seed: Random seed
            cache_enabled: Whether to use caching for results
        """
        self.dataset = dataset
        self.n_test_samples = n_test_samples
        self.n_pos_examples = n_pos_examples
        self.n_neg_absent = n_neg_absent
        self.n_neg_wrong = n_neg_wrong
        self.min_pixels = min_pixels
        self.seed = seed
        self.cache_enabled = cache_enabled
        
        # Validate dataset has required methods
        self._validate_dataset()
        
        # Set output directory
        if output_dir is None:
            # Try to get dataset tag, fallback to class name
            dataset_tag = getattr(dataset, 'dataset_tag', dataset.__class__.__name__.lower())
            base_dir = Path.home() / ".cache" / "endopoint" / "fewshot"
            output_dir = base_dir / dataset_tag
        self.output_dir = Path(output_dir)
        if self.cache_enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache for computed data
        self._presence_matrix = None
        self._balanced_indices = None
        self._analyzer = None
    
    def _validate_dataset(self):
        """Validate that dataset has required methods."""
        required_attrs = [
            'label_ids', 'id2label', 'total', 'get_example',
            'example_to_tensors', 'labels_to_presence_vector'
        ]
        
        for attr in required_attrs:
            if not hasattr(self.dataset, attr):
                raise AttributeError(
                    f"Dataset {self.dataset.__class__.__name__} missing required "
                    f"attribute/method: {attr}"
                )
        
        # Check optional methods for specific tasks
        self.supports_pointing = hasattr(self.dataset, 'sample_point_in_mask')
        self.supports_bbox = hasattr(self.dataset, 'get_bounding_boxes')
        
        if not self.supports_pointing and not self.supports_bbox:
            print("Warning: Dataset doesn't support pointing or bbox tasks")
    
    def compute_presence_matrix(self, split: str = "train", max_samples: Optional[int] = None) -> np.ndarray:
        """Compute presence matrix for all examples in split.
        
        Args:
            split: Dataset split to use
            max_samples: Maximum number of samples to process (None for all)
            
        Returns:
            Presence matrix [N, K] where K is number of classes
        """
        # Check cache
        cache_key = f"presence_matrix_{split}"
        if max_samples is not None:
            cache_key += f"_{max_samples}"
        
        if self._presence_matrix is not None and not max_samples:
            return self._presence_matrix
        
        n_total = self.dataset.total(split)
        n_examples = min(n_total, max_samples) if max_samples else n_total
        n_classes = len(self.dataset.label_ids)
        
        # Check for cached file
        if self.cache_enabled:
            cache_file = self.output_dir / f"{cache_key}.npy"
            if cache_file.exists():
                try:
                    Y = np.load(cache_file)
                    print(f"Loaded cached presence matrix from {cache_file.name}")
                    if not max_samples:
                        self._presence_matrix = Y
                    return Y
                except Exception as e:
                    print(f"Failed to load cache: {e}")
        
        print(f"Computing presence matrix for {n_examples}/{n_total} {split} examples...")
        Y = np.zeros((n_examples, n_classes), dtype=np.uint8)
        
        # Use tqdm for progress
        for idx in tqdm(range(n_examples), desc="Processing examples"):
            example = self.dataset.get_example(split, idx)
            img_t, lab_t = self.dataset.example_to_tensors(example)
            
            # Handle different return types (numpy array or torch tensor)
            presence = self.dataset.labels_to_presence_vector(lab_t, min_pixels=self.min_pixels)
            if hasattr(presence, 'cpu'):
                presence = presence.cpu().numpy()
            elif not isinstance(presence, np.ndarray):
                presence = np.array(presence)
            
            Y[idx] = presence
        
        # Cache result
        if self.cache_enabled:
            cache_file = self.output_dir / f"{cache_key}.npy"
            np.save(cache_file, Y)
            print(f"Cached presence matrix to {cache_file.name}")
        
        if not max_samples:
            self._presence_matrix = Y
            # Create analyzer for this presence matrix
            self._analyzer = DatasetBalanceAnalyzer(self.dataset, Y)
        return Y
    
    def get_analyzer(self) -> DatasetBalanceAnalyzer:
        """Get or create dataset balance analyzer.
        
        Returns:
            DatasetBalanceAnalyzer instance
        """
        if self._analyzer is None:
            if self._presence_matrix is None:
                raise ValueError("Must compute presence matrix first")
            self._analyzer = DatasetBalanceAnalyzer(self.dataset, self._presence_matrix)
        return self._analyzer
    
    def select_balanced_test_set(
        self,
        Y: Optional[np.ndarray] = None,
        split: str = "train",
        use_advanced: bool = True,
        rare_top_k: Optional[int] = None,
        min_quota_rare: Optional[int] = None,
        max_cap_frac: float = 0.70,
        extra_min_quota: Optional[Dict[int, int]] = None
    ) -> Tuple[List[int], Dict[str, Any]]:
        """Select balanced test set with flexible configuration.
        
        Args:
            Y: Presence matrix (computed if not provided)
            split: Dataset split to use for computing Y
            use_advanced: Use advanced selection with caps
            rare_top_k: Number of rare classes to boost (auto if None)
            min_quota_rare: Minimum samples for rare classes (auto if None)
            max_cap_frac: Maximum fraction for abundant classes
            extra_min_quota: Additional minimum quotas per class
            
        Returns:
            Tuple of (selected_indices, selection_info)
        """
        if Y is None:
            Y = self.compute_presence_matrix(split)
        
        # Auto-configure parameters based on dataset
        n_classes = len(self.dataset.label_ids)
        auto_params = auto_configure_selection_params(n_classes, self.n_test_samples)
        
        if rare_top_k is None:
            rare_top_k = auto_params['rare_top_k']
        if min_quota_rare is None:
            min_quota_rare = auto_params['min_quota_rare']
        
        # Ensure we don't select more than available
        actual_n_select = min(self.n_test_samples, Y.shape[0])
        
        # Check for cached results if caching enabled
        if self.cache_enabled:
            cache_suffix = "advanced" if use_advanced else "simple"
            cache_file = self.output_dir / f"balanced_test_indices_{cache_suffix}_{actual_n_select}.json"
            
            if cache_file.exists():
                try:
                    with open(cache_file, 'r') as f:
                        data = json.load(f)
                        print(f"Loaded cached balanced indices from {cache_file.name}")
                        self._balanced_indices = data['indices']
                        return data['indices'], data.get('info', {})
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Cache file corrupted ({e}), regenerating...")
        
        # Select balanced samples
        pool_indices = list(range(Y.shape[0]))
        
        if use_advanced:
            print(f"Using advanced balanced selection (rare_top_k={rare_top_k}, min_quota={min_quota_rare})...")
            selected_indices, info = select_balanced_with_caps(
                Y=Y,
                pool_indices=pool_indices,
                n_select=actual_n_select,
                rare_top_k=rare_top_k,
                min_quota_rare=min_quota_rare,
                max_cap_frac=max_cap_frac,
                extra_min_quota=extra_min_quota,
                seed=self.seed
            )
        else:
            print("Using simple balanced selection...")
            selected_indices = select_balanced_simple(
                Y=Y,
                pool_indices=pool_indices,
                n_select=actual_n_select,
                seed=self.seed
            )
            info = {'method': 'simple', 'n_selected': len(selected_indices)}
        
        # Save to cache if enabled
        if self.cache_enabled:
            cache_file = self.output_dir / f"balanced_test_indices_{cache_suffix}_{actual_n_select}.json"
            with open(cache_file, 'w') as f:
                json.dump({
                    'indices': selected_indices,
                    'info': info,
                    'seed': self.seed,
                    'dataset_tag': getattr(self.dataset, 'dataset_tag', 'unknown')
                }, f, indent=2)
            print(f"Saved balanced indices to {cache_file.name}")
        
        self._balanced_indices = selected_indices
        return selected_indices, info
    
    def build_pointing_examples(
        self,
        Y: Optional[np.ndarray] = None,
        excluded_indices: Optional[List[int]] = None,
        split: str = "train"
    ) -> Optional[Dict[str, Any]]:
        """Build few-shot examples for pointing task.
        
        Args:
            Y: Presence matrix (computed if not provided)
            excluded_indices: Indices to exclude (uses balanced test set if not provided)
            split: Dataset split to use
            
        Returns:
            Few-shot plan dictionary or None if not supported
        """
        if not self.supports_pointing:
            print("Warning: Dataset doesn't support pointing task (missing sample_point_in_mask method)")
            return None
        
        if Y is None:
            Y = self.compute_presence_matrix(split)
        
        if excluded_indices is None:
            if self._balanced_indices is None:
                excluded_indices, _ = self.select_balanced_test_set(Y, split)
            else:
                excluded_indices = self._balanced_indices
        
        # Check cache
        if self.cache_enabled:
            cache_file = self.output_dir / f"fewshot_plan_pointing_{len(excluded_indices)}.json"
            if cache_file.exists():
                try:
                    with open(cache_file, 'r') as f:
                        plan = json.load(f)
                        print(f"Loaded cached pointing plan from {cache_file.name}")
                        return plan
                except json.JSONDecodeError:
                    print(f"Cache file corrupted, regenerating...")
        
        # Build plan
        print("Building pointing few-shot examples...")
        builder = PointingFewShotBuilder(
            dataset=self.dataset,
            n_pos_examples=self.n_pos_examples,
            n_neg_absent=self.n_neg_absent,
            n_neg_wrong=self.n_neg_wrong,
            min_pixels=self.min_pixels,
            seed=self.seed + 1
        )
        
        plan = builder.build_plan(Y, excluded_indices, split)
        
        # Save to cache
        if self.cache_enabled:
            cache_file = self.output_dir / f"fewshot_plan_pointing_{len(excluded_indices)}.json"
            with open(cache_file, 'w') as f:
                json.dump(plan, f, indent=2)
            print(f"Saved pointing plan to {cache_file.name}")
        
        return plan
    
    def build_bbox_examples(
        self,
        Y: Optional[np.ndarray] = None,
        excluded_indices: Optional[List[int]] = None,
        split: str = "train",
        min_bbox_size: int = 20
    ) -> Optional[Dict[str, Any]]:
        """Build few-shot examples for bounding box task.
        
        Args:
            Y: Presence matrix (computed if not provided)
            excluded_indices: Indices to exclude (uses balanced test set if not provided)
            split: Dataset split to use
            min_bbox_size: Minimum bbox size
            
        Returns:
            Few-shot plan dictionary or None if not supported
        """
        if not self.supports_bbox:
            print("Warning: Dataset doesn't support bounding box task (missing get_bounding_boxes method)")
            return None
        
        if Y is None:
            Y = self.compute_presence_matrix(split)
        
        if excluded_indices is None:
            if self._balanced_indices is None:
                excluded_indices, _ = self.select_balanced_test_set(Y, split)
            else:
                excluded_indices = self._balanced_indices
        
        # Check cache
        if self.cache_enabled:
            cache_file = self.output_dir / f"fewshot_plan_bbox_{len(excluded_indices)}.json"
            if cache_file.exists():
                try:
                    with open(cache_file, 'r') as f:
                        plan = json.load(f)
                        print(f"Loaded cached bbox plan from {cache_file.name}")
                        return plan
                except json.JSONDecodeError:
                    print(f"Cache file corrupted, regenerating...")
        
        # Build plan
        print("Building bounding box few-shot examples...")
        builder = BoundingBoxFewShotBuilder(
            dataset=self.dataset,
            n_pos_examples=self.n_pos_examples,
            n_neg_absent=self.n_neg_absent,
            n_neg_wrong=self.n_neg_wrong,
            min_pixels=self.min_pixels,
            min_bbox_size=min_bbox_size,
            seed=self.seed + 2
        )
        
        plan = builder.build_plan(Y, excluded_indices, split)
        
        # Save to cache
        if self.cache_enabled:
            cache_file = self.output_dir / f"fewshot_plan_bbox_{len(excluded_indices)}.json"
            with open(cache_file, 'w') as f:
                json.dump(plan, f, indent=2)
            print(f"Saved bbox plan to {cache_file.name}")
        
        return plan
    
    def print_summary(
        self, 
        pointing_plan: Optional[Dict] = None, 
        bbox_plan: Optional[Dict] = None,
        selection_info: Optional[Dict] = None
    ):
        """Print summary of few-shot examples.
        
        Args:
            pointing_plan: Pointing task plan
            bbox_plan: Bounding box task plan
            selection_info: Selection info from balanced test set
        """
        print("\n" + "="*60)
        print("📊 Few-Shot Example Summary")
        print("="*60)
        
        # Dataset info
        dataset_tag = getattr(self.dataset, 'dataset_tag', self.dataset.__class__.__name__)
        print(f"Dataset: {dataset_tag}")
        print(f"Number of classes: {len(self.dataset.label_ids)}")
        if self.cache_enabled:
            print(f"Cache directory: {self.output_dir}")
        print(f"Test samples: {self.n_test_samples}")
        
        # Selection info
        if selection_info:
            if 'rare_order_cols' in selection_info:
                print(f"\nBalanced Selection Details:")
                print(f"  Rare classes boosted: {selection_info['rare_order_cols']}")
                print(f"  Cap fraction: {selection_info.get('cap_frac', 0.7):.1%}")
        
        # Task summaries
        if pointing_plan:
            print("\n📍 Pointing Task:")
            total_pos = total_neg_absent = total_neg_wrong = 0
            for class_id_str, info in pointing_plan['plan'].items():
                name = info['name']
                n_pos = len(info['positives'])
                n_neg_absent = len(info['negatives_absent'])
                n_neg_wrong = len(info.get('negatives_wrong_point', []))
                total_pos += n_pos
                total_neg_absent += n_neg_absent
                total_neg_wrong += n_neg_wrong
                print(f"  {name:25} pos={n_pos}, neg_absent={n_neg_absent}, neg_wrong={n_neg_wrong}")
            print(f"  {'TOTAL':25} pos={total_pos}, neg_absent={total_neg_absent}, neg_wrong={total_neg_wrong}")
        elif self.supports_pointing:
            print("\n📍 Pointing Task: Not yet generated (run build_pointing_examples)")
        else:
            print("\n📍 Pointing Task: Not supported by dataset")
        
        if bbox_plan:
            print("\n📦 Bounding Box Task:")
            total_pos = total_neg_absent = total_neg_wrong = total_multi = 0
            for class_id_str, info in bbox_plan['plan'].items():
                name = info['name']
                n_pos = len(info['positives'])
                n_neg_absent = len(info['negatives_absent'])
                n_neg_wrong = len(info.get('negatives_wrong_bbox', []))
                total_pos += n_pos
                total_neg_absent += n_neg_absent
                total_neg_wrong += n_neg_wrong
                
                # Count multi-region examples
                multi_region = sum(1 for p in info['positives'] if p.get('num_regions', 1) > 1)
                total_multi += multi_region
                
                print(f"  {name:25} pos={n_pos}, neg_absent={n_neg_absent}, neg_wrong={n_neg_wrong}")
                if multi_region > 0:
                    print(f"    └─ {multi_region} positive examples have multiple regions")
            print(f"  {'TOTAL':25} pos={total_pos}, neg_absent={total_neg_absent}, neg_wrong={total_neg_wrong}")
            if total_multi > 0:
                print(f"    └─ {total_multi} total examples with multiple regions")
        elif self.supports_bbox:
            print("\n📦 Bounding Box Task: Not yet generated (run build_bbox_examples)")
        else:
            print("\n📦 Bounding Box Task: Not supported by dataset")
    
    def build_all_examples(
        self,
        split: str = "train",
        max_samples: Optional[int] = None,
        use_advanced_selection: bool = True
    ) -> Dict[str, Any]:
        """Build all few-shot examples in one go.
        
        Args:
            split: Dataset split to use
            max_samples: Maximum samples to process (for large datasets)
            use_advanced_selection: Use advanced balanced selection
            
        Returns:
            Dictionary with all results
        """
        results = {}
        
        # Compute presence matrix
        print("\n🔄 Step 1: Computing presence matrix...")
        Y = self.compute_presence_matrix(split, max_samples)
        results['presence_matrix_shape'] = Y.shape
        
        # Select balanced test set
        print("\n🔄 Step 2: Selecting balanced test set...")
        test_indices, selection_info = self.select_balanced_test_set(
            Y=Y,
            split=split,
            use_advanced=use_advanced_selection
        )
        results['test_indices'] = test_indices
        results['selection_info'] = selection_info
        
        # Build pointing examples if supported
        if self.supports_pointing:
            print("\n🔄 Step 3: Building pointing examples...")
            pointing_plan = self.build_pointing_examples(Y, test_indices, split)
            results['pointing_plan'] = pointing_plan
        else:
            print("\n⏭️ Step 3: Skipping pointing (not supported)")
            results['pointing_plan'] = None
        
        # Build bbox examples if supported
        if self.supports_bbox:
            print("\n🔄 Step 4: Building bounding box examples...")
            bbox_plan = self.build_bbox_examples(Y, test_indices, split)
            results['bbox_plan'] = bbox_plan
        else:
            print("\n⏭️ Step 4: Skipping bounding boxes (not supported)")
            results['bbox_plan'] = None
        
        # Print summary
        print("\n" + "="*60)
        self.print_summary(
            results.get('pointing_plan'),
            results.get('bbox_plan'),
            selection_info
        )
        
        return results
    
    def run_balanced_selection_pipeline(
        self,
        split: str = "train",
        visualize: bool = False,
        save_summary: bool = True
    ) -> Dict[str, Any]:
        """Run complete balanced selection pipeline with analysis.
        
        This method combines all the steps with comprehensive analysis,
        replacing the repetitive code in notebooks.
        
        Args:
            split: Dataset split to use
            visualize: Whether to print visualizations
            save_summary: Whether to save summary to file
            
        Returns:
            Dictionary with complete results and analysis
        """
        results = {}
        
        # Step 1: Compute presence matrix
        print(f"\n🔄 Step 1: Computing presence matrix for {split} split...")
        Y = self.compute_presence_matrix(split)
        results['presence_matrix_shape'] = Y.shape
        
        # Get analyzer
        analyzer = self.get_analyzer()
        
        # Print distribution report
        if visualize:
            distribution = analyzer.get_class_distribution()
            rare_classes = analyzer.identify_rare_classes()
            
            print(f"\n📊 Class distribution in {split} set:")
            for name, stats in distribution.items():
                print(f"  {name:30} {stats['count']:6d} samples ({stats['percentage']:5.1f}%)")
            
            if rare_classes:
                print(f"\n🔴 Rare classes (<20%): {rare_classes}")
        
        # Step 2: Select balanced test set
        print(f"\n🔄 Step 2: Selecting {self.n_test_samples} balanced test samples...")
        
        # Show auto-configured parameters
        n_classes = len(self.dataset.label_ids)
        auto_params = auto_configure_selection_params(n_classes, self.n_test_samples)
        print(f"  Configuration:")
        print(f"    - Rare class boost: top {auto_params['rare_top_k']} rarest classes")
        print(f"    - Min quota for rare: {auto_params['min_quota_rare']} samples ({auto_params['min_quota_rare']/self.n_test_samples*100:.0f}%)")
        print(f"    - Cap for abundant: {auto_params['max_cap_frac']*100:.0f}% of ideal distribution")
        
        # Auto-configure and select
        test_indices, selection_info = self.select_balanced_test_set(
            Y=Y,
            split=split,
            use_advanced=True
        )
        
        results['test_indices'] = test_indices
        results['selection_info'] = selection_info
        
        # Analyze selection quality
        comparison = analyzer.compare_distributions(test_indices)
        results['balance_comparison'] = comparison
        
        if visualize:
            analyzer.print_distribution_report(test_indices)
        
        # Step 3: Build pointing examples
        if self.supports_pointing:
            print(f"\n🔄 Step 3: Building pointing examples...")
            pointing_plan = self.build_pointing_examples(Y, test_indices, split)
            results['pointing_plan'] = pointing_plan
            
            if pointing_plan and visualize:
                stats = summarize_fewshot_plan(pointing_plan)
                print(f"  Created {stats['total_examples']} pointing examples")
                print(f"    Positive: {stats['total_positive']}")
                print(f"    Negative (absent): {stats['total_negative_absent']}")
                print(f"    Negative (wrong): {stats['total_negative_wrong']}")
        else:
            print(f"\n⏭️ Step 3: Skipping pointing (not supported)")
            results['pointing_plan'] = None
        
        # Step 4: Build bbox examples
        if self.supports_bbox:
            print(f"\n🔄 Step 4: Building bounding box examples...")
            bbox_plan = self.build_bbox_examples(Y, test_indices, split)
            results['bbox_plan'] = bbox_plan
            
            if bbox_plan and visualize:
                stats = summarize_fewshot_plan(bbox_plan)
                print(f"  Created {stats['total_examples']} bbox examples")
                print(f"    Positive: {stats['total_positive']}")
                print(f"    Negative (absent): {stats['total_negative_absent']}")
                print(f"    Negative (wrong): {stats['total_negative_wrong']}")
                if stats['total_multi_region'] > 0:
                    print(f"    Multi-region: {stats['total_multi_region']}")
        else:
            print(f"\n⏭️ Step 4: Skipping bounding boxes (not supported)")
            results['bbox_plan'] = None
        
        # Save summary if requested
        if save_summary and self.cache_enabled:
            summary_file = self.output_dir / "pipeline_summary.json"
            
            # Prepare JSON-serializable summary
            summary = {
                'dataset_tag': getattr(self.dataset, 'dataset_tag', 'unknown'),
                'n_classes': len(self.dataset.label_ids),
                'n_train_total': int(Y.shape[0]),
                'n_test_selected': len(test_indices),
                'balance_metrics': comparison['metrics'],
                'class_distribution': {
                    'original': analyzer.class_percentages.tolist(),
                    'selected': comparison['classes']
                },
                'rare_classes': analyzer.identify_rare_classes(),
                'pointing_stats': summarize_fewshot_plan(results.get('pointing_plan')),
                'bbox_stats': summarize_fewshot_plan(results.get('bbox_plan'))
            }
            
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"\n💾 Pipeline summary saved to: {summary_file}")
        
        return results