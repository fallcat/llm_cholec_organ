"""Dataset adapters for endopoint."""

from .base import DatasetAdapter, build_dataset, register_dataset

# Import concrete adapters to register them
from . import cholecseg8k
from . import cholecseg8k_local
from . import cholec_organs
from . import cholec_gonogo

__all__ = ["DatasetAdapter", "build_dataset", "register_dataset"]