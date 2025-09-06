#!/usr/bin/env python3
"""
Comprehensive test script for refactored balance selection modules.

Tests the new refactored balance selection modules and validates:
1. 200 balanced samples are generated for each dataset
2. DatasetBalanceAnalyzer functionality 
3. UnifiedFewShotSelector pipeline method works correctly
4. Comparison between old and new approaches

Usage:
    python test_balanced_selection.py [--dataset DATASET] [--quick] [--compare-old]
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
from tabulate import tabulate

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from endopoint.datasets import build_dataset
from endopoint.fewshot.unified import UnifiedFewShotSelector
from endopoint.fewshot.balanced_selection import select_balanced_with_caps, select_balanced_simple
from endopoint.fewshot.analysis import DatasetBalanceAnalyzer

# Import old approach for comparison
from few_shot_selection import select_balanced_indices as old_select_balanced_indices
from few_shot_selection import build_presence_matrix as old_build_presence_matrix


class BalanceSelectionTester:
    """Comprehensive tester for balance selection functionality."""
    
    def __init__(self, datasets_to_test: Optional[List[str]] = None, quick_mode: bool = False):
        """Initialize tester.
        
        Args:
            datasets_to_test: List of dataset names to test (None for all)
            quick_mode: If True, test with smaller sample sizes
        """
        self.datasets_to_test = datasets_to_test or ["cholecseg8k", "cholec_organs", "cholec_gonogo"]
        self.quick_mode = quick_mode
        self.n_test_samples = 50 if quick_mode else 200
        
        self.results = {}
        self.test_cache_dir = Path("test_cache") 
        self.test_cache_dir.mkdir(exist_ok=True)
        
    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp."""
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
    
    def test_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """Test balance selection for a single dataset."""
        self.log(f"Testing dataset: {dataset_name}")
        
        results = {
            "dataset_name": dataset_name,
            "n_test_samples": self.n_test_samples,
            "errors": [],
            "warnings": [],
            "metrics": {}
        }
        
        try:
            # Load dataset
            self.log(f"Loading {dataset_name} dataset...")
            dataset = build_dataset(dataset_name)
            
            # Test 1: UnifiedFewShotSelector initialization
            selector = self._test_unified_selector_init(dataset, results)
            if selector is None:
                return results
            
            # Test 2: Presence matrix computation
            presence_matrix = self._test_presence_matrix(selector, results)
            if presence_matrix is None:
                return results
            
            # Test 3: DatasetBalanceAnalyzer
            analyzer = self._test_balance_analyzer(selector, presence_matrix, results)
            
            # Test 4: Balanced selection (simple and advanced)
            self._test_balanced_selection(selector, presence_matrix, results)
            
            # Test 5: Pipeline method
            self._test_pipeline_method(selector, results)
            
            # Test 6: Few-shot example building (if supported)
            self._test_fewshot_building(selector, results)
            
            self.log(f"✅ Completed testing {dataset_name}")
            
        except Exception as e:
            error_msg = f"Fatal error testing {dataset_name}: {str(e)}"
            self.log(error_msg, "ERROR")
            results["errors"].append(error_msg)
            import traceback
            traceback.print_exc()
        
        return results
    
    def _test_unified_selector_init(self, dataset, results: Dict) -> Optional[UnifiedFewShotSelector]:
        """Test UnifiedFewShotSelector initialization."""
        self.log("Testing UnifiedFewShotSelector initialization...")
        
        try:
            # Test with custom output dir
            output_dir = self.test_cache_dir / dataset.__class__.__name__.lower()
            selector = UnifiedFewShotSelector(
                dataset=dataset,
                output_dir=output_dir,
                n_test_samples=self.n_test_samples,
                seed=42,
                cache_enabled=True
            )
            
            # Validate properties
            assert hasattr(selector, 'dataset')
            assert hasattr(selector, 'n_test_samples')
            assert selector.n_test_samples == self.n_test_samples
            assert hasattr(selector, 'supports_pointing')
            assert hasattr(selector, 'supports_bbox')
            
            self.log(f"✅ Selector initialized (pointing: {selector.supports_pointing}, bbox: {selector.supports_bbox})")
            
            results["metrics"]["supports_pointing"] = selector.supports_pointing
            results["metrics"]["supports_bbox"] = selector.supports_bbox
            results["metrics"]["n_classes"] = len(dataset.label_ids)
            
            return selector
            
        except Exception as e:
            error_msg = f"UnifiedFewShotSelector init failed: {str(e)}"
            self.log(error_msg, "ERROR")
            results["errors"].append(error_msg)
            return None
    
    def _test_presence_matrix(self, selector: UnifiedFewShotSelector, results: Dict) -> Optional[np.ndarray]:
        """Test presence matrix computation."""
        self.log("Testing presence matrix computation...")
        
        try:
            # Test with limited samples in quick mode
            max_samples = 500 if self.quick_mode else None
            
            Y = selector.compute_presence_matrix(split="train", max_samples=max_samples)
            
            # Validate matrix properties
            assert isinstance(Y, np.ndarray)
            assert Y.dtype == np.uint8
            assert Y.ndim == 2
            assert Y.shape[1] == len(selector.dataset.label_ids)
            assert np.all((Y == 0) | (Y == 1))  # Binary values only
            
            self.log(f"✅ Presence matrix computed: {Y.shape}")
            
            results["metrics"]["presence_matrix_shape"] = Y.shape
            results["metrics"]["total_positives"] = int(Y.sum())
            results["metrics"]["class_counts"] = Y.sum(axis=0).tolist()
            
            return Y
            
        except Exception as e:
            error_msg = f"Presence matrix computation failed: {str(e)}"
            self.log(error_msg, "ERROR")
            results["errors"].append(error_msg)
            return None
    
    def _test_balance_analyzer(self, selector: UnifiedFewShotSelector, Y: np.ndarray, results: Dict) -> Optional[DatasetBalanceAnalyzer]:
        """Test DatasetBalanceAnalyzer functionality."""
        self.log("Testing DatasetBalanceAnalyzer...")
        
        try:
            analyzer = DatasetBalanceAnalyzer(selector.dataset, Y)
            
            # Test distribution analysis
            distribution = analyzer.get_class_distribution()
            assert isinstance(distribution, dict)
            assert len(distribution) == len(selector.dataset.label_ids)
            
            # Test rare class identification
            rare_classes = analyzer.identify_rare_classes(threshold=20.0)
            assert isinstance(rare_classes, list)
            
            # Test with sample selection
            sample_indices = list(range(min(50, Y.shape[0])))
            comparison = analyzer.compare_distributions(sample_indices)
            
            assert isinstance(comparison, dict)
            assert "n_selected" in comparison
            assert "classes" in comparison
            assert "metrics" in comparison
            
            self.log(f"✅ Balance analyzer works (found {len(rare_classes)} rare classes)")
            
            results["metrics"]["rare_classes"] = rare_classes
            results["metrics"]["distribution_stddev"] = float(np.std([
                stats["percentage"] for stats in distribution.values()
            ]))
            
            return analyzer
            
        except Exception as e:
            error_msg = f"DatasetBalanceAnalyzer test failed: {str(e)}"
            self.log(error_msg, "ERROR")
            results["errors"].append(error_msg)
            return None
    
    def _test_balanced_selection(self, selector: UnifiedFewShotSelector, Y: np.ndarray, results: Dict):
        """Test balanced selection algorithms."""
        self.log("Testing balanced selection algorithms...")
        
        try:
            # Test simple selection
            selected_simple, info_simple = selector.select_balanced_test_set(
                Y=Y,
                use_advanced=False
            )
            
            assert isinstance(selected_simple, list)
            assert len(selected_simple) == min(self.n_test_samples, Y.shape[0])
            assert all(isinstance(idx, int) for idx in selected_simple)
            assert len(set(selected_simple)) == len(selected_simple)  # No duplicates
            
            # Test advanced selection
            selected_advanced, info_advanced = selector.select_balanced_test_set(
                Y=Y,
                use_advanced=True
            )
            
            assert isinstance(selected_advanced, list)
            assert len(selected_advanced) == min(self.n_test_samples, Y.shape[0])
            assert all(isinstance(idx, int) for idx in selected_advanced)
            assert len(set(selected_advanced)) == len(selected_advanced)  # No duplicates
            
            # Analyze selection quality
            Y_simple = Y[selected_simple] if selected_simple else np.zeros((0, Y.shape[1]))
            Y_advanced = Y[selected_advanced] if selected_advanced else np.zeros((0, Y.shape[1]))
            
            simple_balance = np.std(Y_simple.mean(axis=0)) if len(selected_simple) > 0 else float('inf')
            advanced_balance = np.std(Y_advanced.mean(axis=0)) if len(selected_advanced) > 0 else float('inf')
            
            self.log(f"✅ Balanced selection works (simple: {len(selected_simple)}, advanced: {len(selected_advanced)})")
            self.log(f"   Balance quality: simple={simple_balance:.3f}, advanced={advanced_balance:.3f}")
            
            results["metrics"]["simple_selection_count"] = len(selected_simple)
            results["metrics"]["advanced_selection_count"] = len(selected_advanced)
            results["metrics"]["simple_balance_stddev"] = float(simple_balance)
            results["metrics"]["advanced_balance_stddev"] = float(advanced_balance)
            
            # Check if advanced is better than simple (should be in most cases)
            if advanced_balance < simple_balance:
                self.log("✅ Advanced selection outperforms simple selection")
            else:
                warning_msg = "Advanced selection did not outperform simple selection"
                self.log(warning_msg, "WARN")
                results["warnings"].append(warning_msg)
                
        except Exception as e:
            error_msg = f"Balanced selection test failed: {str(e)}"
            self.log(error_msg, "ERROR")
            results["errors"].append(error_msg)
    
    def _test_pipeline_method(self, selector: UnifiedFewShotSelector, results: Dict):
        """Test the unified pipeline method."""
        self.log("Testing unified pipeline method...")
        
        try:
            max_samples = 500 if self.quick_mode else None
            
            pipeline_results = selector.run_balanced_selection_pipeline(
                split="train",
                visualize=False,  # Don't print verbose output during testing
                save_summary=True
            )
            
            # Validate pipeline results
            assert isinstance(pipeline_results, dict)
            assert "presence_matrix_shape" in pipeline_results
            assert "test_indices" in pipeline_results
            assert "selection_info" in pipeline_results
            assert "balance_comparison" in pipeline_results
            
            # Validate test indices
            test_indices = pipeline_results["test_indices"]
            assert isinstance(test_indices, list)
            assert len(test_indices) <= self.n_test_samples
            assert all(isinstance(idx, int) for idx in test_indices)
            
            self.log(f"✅ Pipeline method works (selected {len(test_indices)} samples)")
            
            results["metrics"]["pipeline_test_count"] = len(test_indices)
            results["metrics"]["pipeline_balance_improvement"] = pipeline_results["balance_comparison"]["metrics"]["balance_improvement_pct"]
            
        except Exception as e:
            error_msg = f"Pipeline method test failed: {str(e)}"
            self.log(error_msg, "ERROR")
            results["errors"].append(error_msg)
    
    def _test_fewshot_building(self, selector: UnifiedFewShotSelector, results: Dict):
        """Test few-shot example building."""
        self.log("Testing few-shot example building...")
        
        try:
            # Build limited presence matrix for testing
            max_samples = 200 if self.quick_mode else 1000
            Y = selector.compute_presence_matrix(split="train", max_samples=max_samples)
            
            # Get test indices
            test_indices, _ = selector.select_balanced_test_set(Y=Y, use_advanced=True)
            
            pointing_plan = None
            bbox_plan = None
            
            # Test pointing examples if supported
            if selector.supports_pointing:
                pointing_plan = selector.build_pointing_examples(Y, test_indices, split="train")
                assert pointing_plan is not None
                assert "plan" in pointing_plan
                self.log("✅ Pointing examples built successfully")
                
                results["metrics"]["pointing_plan_classes"] = len(pointing_plan["plan"])
            
            # Test bbox examples if supported  
            if selector.supports_bbox:
                bbox_plan = selector.build_bbox_examples(Y, test_indices, split="train")
                assert bbox_plan is not None
                assert "plan" in bbox_plan
                self.log("✅ Bounding box examples built successfully")
                
                results["metrics"]["bbox_plan_classes"] = len(bbox_plan["plan"])
            
            if not selector.supports_pointing and not selector.supports_bbox:
                self.log("ℹ️  Dataset doesn't support pointing or bbox tasks", "WARN")
                
        except Exception as e:
            error_msg = f"Few-shot building test failed: {str(e)}"
            self.log(error_msg, "ERROR")
            results["errors"].append(error_msg)
    
    def compare_with_old_approach(self, dataset_name: str) -> Dict[str, Any]:
        """Compare new refactored approach with old approach."""
        self.log(f"Comparing new vs old approach for {dataset_name}")
        
        comparison_results = {
            "dataset_name": dataset_name,
            "comparison": {},
            "errors": []
        }
        
        try:
            if dataset_name != "cholecseg8k":
                warning_msg = f"Old approach comparison only supported for cholecseg8k, skipping {dataset_name}"
                self.log(warning_msg, "WARN")
                comparison_results["comparison"]["status"] = "skipped"
                return comparison_results
            
            from datasets import load_dataset
            
            # Load dataset for old approach
            hf_dataset = load_dataset("minwoosun/CholecSeg8k")
            
            # Old approach
            self.log("Running old approach...")
            max_samples = 500 if self.quick_mode else None
            indices_to_use = list(range(min(max_samples or len(hf_dataset["train"]), len(hf_dataset["train"]))))
            
            Y_old, _ = old_build_presence_matrix(hf_dataset, "train", indices_to_use, min_pixels=1)
            old_selected = old_select_balanced_indices(Y_old, indices_to_use, n_select=self.n_test_samples, seed=42)
            
            # New approach
            self.log("Running new approach...")
            dataset = build_dataset("cholecseg8k")
            selector = UnifiedFewShotSelector(
                dataset=dataset,
                n_test_samples=self.n_test_samples,
                seed=42,
                cache_enabled=False  # Disable caching for fair comparison
            )
            
            Y_new = selector.compute_presence_matrix("train", max_samples=max_samples)
            new_selected, _ = selector.select_balanced_test_set(Y=Y_new, use_advanced=True)
            
            # Compare results
            Y_old_selected = Y_old[old_selected] if old_selected else np.zeros((0, Y_old.shape[1]))
            Y_new_selected = Y_new[new_selected] if new_selected else np.zeros((0, Y_new.shape[1]))
            
            old_balance = np.std(Y_old_selected.mean(axis=0)) if len(old_selected) > 0 else float('inf')
            new_balance = np.std(Y_new_selected.mean(axis=0)) if len(new_selected) > 0 else float('inf')
            
            comparison_results["comparison"] = {
                "status": "completed",
                "old_approach": {
                    "selected_count": len(old_selected),
                    "balance_stddev": float(old_balance),
                    "presence_matrix_shape": Y_old.shape
                },
                "new_approach": {
                    "selected_count": len(new_selected),
                    "balance_stddev": float(new_balance),
                    "presence_matrix_shape": Y_new.shape
                },
                "improvement": {
                    "balance_improvement": float(old_balance - new_balance),
                    "is_better": new_balance < old_balance
                }
            }
            
            self.log(f"✅ Comparison completed: old={old_balance:.3f}, new={new_balance:.3f}")
            
        except Exception as e:
            error_msg = f"Comparison failed: {str(e)}"
            self.log(error_msg, "ERROR")
            comparison_results["errors"].append(error_msg)
            comparison_results["comparison"]["status"] = "failed"
        
        return comparison_results
    
    def run_all_tests(self, compare_old: bool = False) -> Dict[str, Any]:
        """Run all tests."""
        self.log(f"Starting comprehensive balance selection tests (quick_mode={self.quick_mode})")
        self.log(f"Testing datasets: {', '.join(self.datasets_to_test)}")
        self.log(f"Target samples per dataset: {self.n_test_samples}")
        
        all_results = {
            "test_config": {
                "datasets": self.datasets_to_test,
                "n_test_samples": self.n_test_samples,
                "quick_mode": self.quick_mode,
                "compare_old": compare_old
            },
            "dataset_results": {},
            "comparisons": {},
            "summary": {}
        }
        
        # Test each dataset
        for dataset_name in self.datasets_to_test:
            try:
                self.log(f"\n{'='*60}")
                self.log(f"Testing {dataset_name}")
                self.log(f"{'='*60}")
                
                dataset_results = self.test_dataset(dataset_name)
                all_results["dataset_results"][dataset_name] = dataset_results
                
                # Old approach comparison if requested
                if compare_old:
                    self.log(f"\n{'-'*40}")
                    self.log(f"Comparing with old approach")
                    self.log(f"{'-'*40}")
                    comparison_results = self.compare_with_old_approach(dataset_name)
                    all_results["comparisons"][dataset_name] = comparison_results
                    
            except Exception as e:
                error_msg = f"Failed to test {dataset_name}: {str(e)}"
                self.log(error_msg, "ERROR")
                all_results["dataset_results"][dataset_name] = {
                    "dataset_name": dataset_name,
                    "errors": [error_msg],
                    "metrics": {}
                }
        
        # Generate summary
        all_results["summary"] = self._generate_summary(all_results)
        
        return all_results
    
    def _generate_summary(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate test summary."""
        summary = {
            "total_datasets": len(self.datasets_to_test),
            "successful_tests": 0,
            "failed_tests": 0,
            "total_errors": 0,
            "total_warnings": 0,
            "by_dataset": {}
        }
        
        for dataset_name, results in all_results["dataset_results"].items():
            errors = len(results.get("errors", []))
            warnings = len(results.get("warnings", []))
            
            if errors == 0:
                summary["successful_tests"] += 1
            else:
                summary["failed_tests"] += 1
                
            summary["total_errors"] += errors
            summary["total_warnings"] += warnings
            
            summary["by_dataset"][dataset_name] = {
                "success": errors == 0,
                "errors": errors,
                "warnings": warnings,
                "metrics_count": len(results.get("metrics", {}))
            }
        
        return summary
    
    def print_results(self, results: Dict[str, Any]):
        """Print formatted test results."""
        print(f"\n{'='*80}")
        print("BALANCE SELECTION TEST RESULTS")
        print(f"{'='*80}")
        
        # Test configuration
        config = results["test_config"]
        print(f"\nTest Configuration:")
        print(f"  Datasets: {', '.join(config['datasets'])}")
        print(f"  Target samples: {config['n_test_samples']}")
        print(f"  Quick mode: {config['quick_mode']}")
        print(f"  Compare old: {config['compare_old']}")
        
        # Summary
        summary = results["summary"]
        print(f"\nOverall Summary:")
        print(f"  Total datasets tested: {summary['total_datasets']}")
        print(f"  Successful: {summary['successful_tests']}")
        print(f"  Failed: {summary['failed_tests']}")
        print(f"  Total errors: {summary['total_errors']}")
        print(f"  Total warnings: {summary['total_warnings']}")
        
        # Dataset-by-dataset results
        print(f"\n{'='*60}")
        print("DATASET RESULTS")
        print(f"{'='*60}")
        
        for dataset_name, dataset_results in results["dataset_results"].items():
            status = "✅ PASS" if len(dataset_results.get("errors", [])) == 0 else "❌ FAIL"
            print(f"\n{dataset_name.upper()}: {status}")
            
            errors = dataset_results.get("errors", [])
            warnings = dataset_results.get("warnings", [])
            metrics = dataset_results.get("metrics", {})
            
            if errors:
                print("  Errors:")
                for error in errors:
                    print(f"    - {error}")
            
            if warnings:
                print("  Warnings:")
                for warning in warnings:
                    print(f"    - {warning}")
            
            if metrics:
                print("  Key Metrics:")
                # Display important metrics in a table
                important_metrics = [
                    ("Classes", metrics.get("n_classes", "N/A")),
                    ("Presence Matrix", f"{metrics.get('presence_matrix_shape', 'N/A')}"),
                    ("Advanced Balance StdDev", f"{metrics.get('advanced_balance_stddev', 'N/A'):.3f}" if isinstance(metrics.get('advanced_balance_stddev'), (int, float)) else "N/A"),
                    ("Pipeline Selected", metrics.get("pipeline_test_count", "N/A")),
                    ("Balance Improvement %", f"{metrics.get('pipeline_balance_improvement', 'N/A'):.1f}" if isinstance(metrics.get('pipeline_balance_improvement'), (int, float)) else "N/A"),
                    ("Supports Pointing", metrics.get("supports_pointing", "N/A")),
                    ("Supports BBox", metrics.get("supports_bbox", "N/A"))
                ]
                
                for name, value in important_metrics:
                    print(f"    {name:25}: {value}")
        
        # Comparison results
        if results.get("comparisons"):
            print(f"\n{'='*60}")
            print("OLD VS NEW COMPARISON")
            print(f"{'='*60}")
            
            for dataset_name, comparison in results["comparisons"].items():
                comp = comparison.get("comparison", {})
                status = comp.get("status", "unknown")
                
                if status == "completed":
                    old_balance = comp["old_approach"]["balance_stddev"]
                    new_balance = comp["new_approach"]["balance_stddev"]
                    is_better = comp["improvement"]["is_better"]
                    
                    result_status = "✅ IMPROVED" if is_better else "⚠️  SAME/WORSE"
                    print(f"\n{dataset_name.upper()}: {result_status}")
                    print(f"  Old approach balance: {old_balance:.3f}")
                    print(f"  New approach balance: {new_balance:.3f}")
                    print(f"  Improvement: {old_balance - new_balance:.3f}")
                    
                elif status == "skipped":
                    print(f"\n{dataset_name.upper()}: SKIPPED (not supported)")
                else:
                    print(f"\n{dataset_name.upper()}: FAILED")
                    for error in comparison.get("errors", []):
                        print(f"    - {error}")
        
        # Final verdict
        if summary["failed_tests"] == 0:
            print(f"\n🎉 ALL TESTS PASSED! ({summary['successful_tests']}/{summary['total_datasets']})")
            if summary["total_warnings"] > 0:
                print(f"   (with {summary['total_warnings']} warnings)")
        else:
            print(f"\n⚠️  SOME TESTS FAILED: {summary['failed_tests']}/{summary['total_datasets']} failed")
        
        print(f"\n{'='*80}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test refactored balance selection modules")
    parser.add_argument("--dataset", choices=["cholecseg8k", "cholec_organs", "cholec_gonogo"], 
                       help="Test specific dataset only")
    parser.add_argument("--quick", action="store_true", help="Quick mode with smaller samples")
    parser.add_argument("--compare-old", action="store_true", help="Compare with old approach")
    parser.add_argument("--save-results", help="Save results to JSON file")
    
    args = parser.parse_args()
    
    # Determine datasets to test
    if args.dataset:
        datasets_to_test = [args.dataset]
    else:
        datasets_to_test = ["cholecseg8k", "cholec_organs", "cholec_gonogo"]
    
    # Create tester and run tests
    tester = BalanceSelectionTester(datasets_to_test, quick_mode=args.quick)
    results = tester.run_all_tests(compare_old=args.compare_old)
    
    # Print results
    tester.print_results(results)
    
    # Save results if requested
    if args.save_results:
        output_path = Path(args.save_results)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {output_path}")
    
    # Exit with appropriate code
    if results["summary"]["failed_tests"] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()