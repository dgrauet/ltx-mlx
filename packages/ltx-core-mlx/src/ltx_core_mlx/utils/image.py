"""Image preparation utilities for VAE encoding."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
from PIL import Image


def prepare_image_for_encoding(
    image: Image.Image | str,
    height: int,
    width: int,
) -> mx.array:
    """Load and prepare an image for VAE encoding.

    Resizes to (height, width), normalizes to [-1, 1], returns as (1, 3, H, W).

    Args:
        image: PIL Image or path to image file.
        height: Target height.
        width: Target width.

    Returns:
        mx.array of shape (1, 3, H, W) in [-1, 1] range, bfloat16.
    """
    if isinstance(image, str):
        image = Image.open(image)

    image = image.convert("RGB")
    image = image.resize((width, height), Image.LANCZOS)

    # HWC uint8 -> float32 -> [-1, 1]
    arr = np.array(image, dtype=np.float32) / 255.0
    arr = arr * 2.0 - 1.0

    # HWC -> CHW -> BCHW
    tensor = mx.array(arr).transpose(2, 0, 1)[None, ...]
    return tensor.astype(mx.bfloat16)
