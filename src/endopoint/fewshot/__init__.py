"""Few-shot example selection and building module."""

from .balanced_selection import (
    select_balanced_with_caps,
    select_balanced_simple
)

from .example_builder import (
    FewShotExampleBuilder,
    BoundingBoxFewShotBuilder,
    PointingFewShotBuilder
)

from .unified import UnifiedFewShotSelector

from .analysis import (
    DatasetBalanceAnalyzer,
    auto_configure_selection_params,
    summarize_fewshot_plan
)

__all__ = [
    'select_balanced_with_caps',
    'select_balanced_simple',
    'FewShotExampleBuilder',
    'BoundingBoxFewShotBuilder', 
    'PointingFewShotBuilder',
    'UnifiedFewShotSelector',
    'DatasetBalanceAnalyzer',
    'auto_configure_selection_params',
    'summarize_fewshot_plan'
]