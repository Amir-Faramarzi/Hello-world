"""Image processing algorithms for redness tracker.

Each algorithm is implemented as a small stateless or stateful class with
parameter injection via the constructor.  The algorithms operate on NumPy
arrays representing BGR images (as returned by OpenCV).

All functions are pure and side effect free which makes them simple to test.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import cast
from typing import Deque, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

ArrayU8 = NDArray[np.uint8]
ArrayBool = NDArray[np.bool_]


@dataclass
class RedMask:
    """Create a binary mask selecting red pixels.

    The mask is `True` where the red channel is dominant and above the given
    thresholds.

    Args:
        r_min: Minimum value for the red channel.
        max_g: Maximum allowed value for the green channel.
        max_b: Maximum allowed value for the blue channel.
    """

    r_min: int = 150
    max_g: int = 100
    max_b: int = 100

    def __call__(self, frame: ArrayU8) -> ArrayBool:
        """Return a boolean mask of pixels considered red."""
        b, g, r = cv2.split(frame)
        mask = (r >= self.r_min) & (g <= self.max_g) & (b <= self.max_b) & (r > g) & (r > b)
        return mask


@dataclass
class FlatFieldCorrector:
    """Perform simple flat-field correction.

    The algorithm divides the image by a blurred version of itself which
    compensates for non-uniform illumination.  The result is normalised to the
    original mean intensity.

    Args:
        blur_k: Kernel size for the Gaussian blur used as the flat field.
        epsilon: Small constant to avoid division by zero.
    """

    blur_k: Tuple[int, int] = (31, 31)
    epsilon: float = 1e-3

    def __call__(self, frame: ArrayU8) -> ArrayU8:
        """Return the flat-field corrected frame."""
        flat = cv2.GaussianBlur(frame, self.blur_k, 0)
        corrected = frame.astype(np.float32) / (flat.astype(np.float32) + self.epsilon)
        mean = frame.mean()
        corrected *= mean
        return np.clip(corrected, 0, 255).astype(np.uint8)


class RunningMedian:
    """Maintain a running median over the last *n* frames.

    The class stores the last ``size`` frames and computes the median along the
    time axis when called.  It is primarily used to estimate the background of a
    video sequence.
    """

    def __init__(self, size: int = 5) -> None:
        self.size = size
        self._buffer: Deque[ArrayU8] = deque(maxlen=size)

    def update(self, frame: ArrayU8) -> ArrayU8:
        """Add a new frame and return the current median."""
        self._buffer.append(frame.copy())
        stack = np.stack(list(self._buffer), axis=0)
        median = np.median(stack, axis=0).astype(frame.dtype)
        return cast(ArrayU8, median)


@dataclass
class DeltaProcessor:
    """Compute the absolute difference to a reference frame.

    Args:
        threshold: Optional threshold; values below are set to zero.
    """

    threshold: int | None = None

    def __call__(self, frame: ArrayU8, reference: ArrayU8) -> ArrayU8:
        """Return the absolute difference between ``frame`` and ``reference``."""
        delta = cv2.absdiff(frame, reference)
        if self.threshold is not None:
            _, delta = cv2.threshold(delta, self.threshold, 255, cv2.THRESH_TOZERO)
        return cast(ArrayU8, delta)


@dataclass
class MorphCleaner:
    """Apply morphological opening and closing to a binary mask.

    Args:
        ksize: Kernel size for morphological operations.
        iterations: Number of iterations for each operation.
    """

    ksize: int = 3
    iterations: int = 1

    def __call__(self, mask: ArrayBool) -> ArrayBool:
        """Return the morphologically cleaned mask."""
        kernel = np.ones((self.ksize, self.ksize), np.uint8)
        opened = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=self.iterations)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=self.iterations)
        return closed.astype(bool)
