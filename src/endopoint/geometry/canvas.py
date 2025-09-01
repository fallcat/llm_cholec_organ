"""Canvas and coordinate transformation utilities."""

from typing import Dict, Optional, Tuple, Union

from PIL import Image


def letterbox_to_canvas(
    pil_img: Image.Image,
    canvas_size: Tuple[int, int] = (768, 768),
    fill: Union[int, Tuple[int, int, int]] = 0,
) -> Tuple[Image.Image, Dict[str, Union[int, float]]]:
    """Apply letterbox transformation to fit image on canvas.
    
    Resizes image to fit within canvas while maintaining aspect ratio,
    then centers it with padding.
    
    Args:
        pil_img: Input PIL Image
        canvas_size: Target canvas size (width, height)
        fill: Fill color for padding (grayscale value or RGB tuple)
        
    Returns:
        canvas: Letterboxed image on canvas
        meta: Transformation metadata dict containing:
            - orig_w, orig_h: Original image dimensions
            - canvas_w, canvas_h: Canvas dimensions
            - scale: Scaling factor applied
            - pad_x, pad_y: Padding offsets
            - resized_w, resized_h: Dimensions after scaling
    """
    W0, H0 = pil_img.width, pil_img.height
    CW, CH = canvas_size
    
    # Calculate scale to fit image within canvas
    scale = min(CW / W0, CH / H0)
    new_w = int(round(W0 * scale))
    new_h = int(round(H0 * scale))
    
    # Resize image
    resized = pil_img.resize((new_w, new_h), Image.BILINEAR)
    
    # Calculate padding to center image
    pad_x = (CW - new_w) // 2
    pad_y = (CH - new_h) // 2
    
    # Create canvas and paste resized image
    if isinstance(fill, int):
        fill_color = (fill, fill, fill)
    else:
        fill_color = fill
    
    canvas = Image.new("RGB", (CW, CH), color=fill_color)
    canvas.paste(resized, (pad_x, pad_y))
    
    meta = {
        "orig_w": W0,
        "orig_h": H0,
        "canvas_w": CW,
        "canvas_h": CH,
        "scale": scale,
        "pad_x": pad_x,
        "pad_y": pad_y,
        "resized_w": new_w,
        "resized_h": new_h,
    }
    
    return canvas, meta


def canvas_to_original(
    xc: int, yc: int, meta: Dict[str, Union[int, float]]
) -> Optional[Tuple[int, int]]:
    """Convert canvas coordinates to original image coordinates.
    
    Args:
        xc: X coordinate on canvas
        yc: Y coordinate on canvas
        meta: Transformation metadata from letterbox_to_canvas
        
    Returns:
        (x, y) coordinates in original image or None if outside bounds
    """
    # Reverse the padding offset
    sx = (xc - meta["pad_x"]) / meta["scale"]
    sy = (yc - meta["pad_y"]) / meta["scale"]
    
    # Check if point is outside original image bounds
    if sx < 0 or sy < 0 or sx >= meta["orig_w"] or sy >= meta["orig_h"]:
        return None
    
    # Round to nearest integer and clamp to valid range
    x = int(round(sx))
    y = int(round(sy))
    x = max(0, min(meta["orig_w"] - 1, x))
    y = max(0, min(meta["orig_h"] - 1, y))
    
    return (x, y)


def original_to_canvas(
    x: int, y: int, meta: Dict[str, Union[int, float]]
) -> Tuple[int, int]:
    """Convert original image coordinates to canvas coordinates.
    
    Args:
        x: X coordinate in original image
        y: Y coordinate in original image
        meta: Transformation metadata from letterbox_to_canvas
        
    Returns:
        (xc, yc) coordinates on canvas
    """
    # Apply scale and padding offset
    xc = int(round(x * meta["scale"] + meta["pad_x"]))
    yc = int(round(y * meta["scale"] + meta["pad_y"]))
    
    # Clamp to canvas bounds
    xc = max(0, min(meta["canvas_w"] - 1, xc))
    yc = max(0, min(meta["canvas_h"] - 1, yc))
    
    return (xc, yc)