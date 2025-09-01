"""Random number generation utilities."""

import random
from typing import Optional

import numpy as np
import torch


def seed_all(seed: int, deterministic: bool = True) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
        deterministic: Whether to use deterministic algorithms in PyTorch
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_rng_state() -> dict:
    """Get current RNG state from all sources.
    
    Returns:
        Dictionary containing RNG states
    """
    state = {
        "random": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    
    return state


def set_rng_state(state: dict) -> None:
    """Restore RNG state from dictionary.
    
    Args:
        state: Dictionary containing RNG states
    """
    random.setstate(state["random"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])