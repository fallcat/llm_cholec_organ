"""Dataset adapters for endopoint."""

from .base import DatasetAdapter, build_dataset, register_dataset

# Import concrete adapters to register them
from . import cholecseg8k, endoscape

__all__ = ["DatasetAdapter", "build_dataset", "register_dataset"]