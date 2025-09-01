"""Shared utilities for model adapters."""

import base64
import hashlib
import io
import pickle
from pathlib import Path
from typing import Any, Optional, Union

import diskcache
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as tvtf


# Create a shared cache
CACHE_DIR = Path.home() / ".cache" / "endopoint" / "models"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
cache = diskcache.Cache(str(CACHE_DIR))


def is_image(x: Any) -> bool:
    """Check if the input is an image in a supported format."""
    return isinstance(x, (Image.Image, torch.Tensor, np.ndarray))


def image_to_base64(
    image: Union[torch.Tensor, np.ndarray, Image.Image],
    image_format: str = "PNG"
) -> str:
    """Convert an image to a base64-encoded string.
    
    Args:
        image: Input image
        image_format: The format to save the image in
        
    Returns:
        Base64-encoded image string
        
    Raises:
        ValueError: If conversion fails
    """
    try:
        # Convert to PIL image if needed
        if isinstance(image, torch.Tensor):
            mode = "RGB" if image.ndim == 3 and image.shape[0] == 3 else "L"
            image = tvtf.to_pil_image(image, mode=mode)
        elif isinstance(image, np.ndarray):
            mode = "RGB" if image.ndim == 3 and image.shape[2] == 3 else "L"
            image = tvtf.to_pil_image(image, mode=mode)
        
        assert isinstance(image, Image.Image), f"Image is not a PIL.Image.Image: {type(image)}"
        image.load()  # Force loading the image
        
        with io.BytesIO() as buffer:
            image.save(buffer, format=image_format)
            return base64.standard_b64encode(buffer.getvalue()).decode('utf-8')
    
    except Exception as e:
        raise ValueError(f"Failed to convert image to base64: {str(e)}")


def get_cache_key(model_name: str, prompt: Any, system_prompt: Optional[str] = None) -> str:
    """Convert a (system_prompt, prompt) into a stable hash string."""
    sys_part = system_prompt or ""
    
    if isinstance(prompt, str):
        obj = ("user_text", prompt)
    elif isinstance(prompt, tuple):
        objs = []
        for p in prompt:
            if isinstance(p, str):
                objs.append(("text", p))
            elif is_image(p):
                # base64-encode image deterministically for hashing
                objs.append(("image_b64", image_to_base64(p, "PNG")))
            else:
                raise ValueError(f"Invalid prompt type: {type(p)}")
        obj = ("user_tuple", tuple(objs))
    else:
        raise ValueError(f"Invalid prompt type: {type(prompt)}")
    
    return hashlib.sha256(pickle.dumps((model_name, sys_part, obj))).hexdigest()