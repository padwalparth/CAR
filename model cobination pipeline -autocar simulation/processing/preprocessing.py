"""
Generic Frame Preprocessing and Validation Utilities.

Ensures incoming camera and video frames meet dimensionality constraints
before model dispatch. Model-specific transformations (letterbox, normalization)
remain isolated inside their respective model runners.
"""

from typing import Any, Tuple
import numpy as np


def validate_frame(frame: Any) -> Tuple[int, int, int]:
    """
    Validate that an incoming frame is a valid 3-channel image array.
    Returns (height, width, channels).
    Raises ValueError if invalid.
    """
    if frame is None:
        raise ValueError("Frame cannot be None.")

    if not hasattr(frame, "shape") or len(frame.shape) < 2:
        raise ValueError("Frame must be an array with at least 2 dimensions.")

    h = frame.shape[0]
    w = frame.shape[1]
    c = frame.shape[2] if len(frame.shape) > 2 else 1

    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid frame dimensions: height={h}, width={w}")

    return h, w, c
