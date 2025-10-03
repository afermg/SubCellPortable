from typing import Union
import numpy as np
import tifffile
from PIL import Image
from pathlib import Path
import os
import logging


def convert_bitdepth(image: np.ndarray, bitdepth: int) -> np.ndarray:
    """Convert image to specified bit depth.

    Args:
        image: Input image array
        bitdepth: Target bit depth (8, 16, or 32)

    Returns:
        Image converted to target bit depth
    """
    if bitdepth == 8:
        if image.dtype != np.uint8:
            return (image / np.iinfo(image.dtype).max * 255).astype(np.uint8)
        else:
            return np.uint8(image)
    elif bitdepth == 16:
        if image.dtype != np.uint16:
            return (image / np.iinfo(image.dtype).max * 65535).astype(np.uint16)
        else:
            return np.uint16(image)
    elif bitdepth == 32:
        if image.dtype != np.uint32:
            return (image / np.iinfo(image.dtype).max * 4294967295).astype(np.uint32)
        else:
            return np.uint32(image)
    return image


def read_grayscale_image(
    input_image: Union[str, Path],
    force_channel: int = -1,
    force_bit_depth: int = 0,
    minmax_norm: bool = False
) -> np.ndarray:
    """Read and preprocess a grayscale image from file or URL.

    Args:
        input_image: Path to image file or URL
        force_channel: Extract specific channel index (default: -1 for auto)
        force_bit_depth: Convert to specific bit depth (default: 0 for no conversion)
        minmax_norm: Apply min-max normalization to [0, 1] range

    Returns:
        Preprocessed grayscale image as numpy array

    Raises:
        FileNotFoundError: If image file doesn't exist
    """
    logger = logging.getLogger(__name__)

    try:
        # Check file extension to choose optimal loading method
        file_ext = Path(input_image).suffix.lower()

        if file_ext in ['.tiff', '.tif']:
            # Use tifffile for TIFF images
            np_img = tifffile.imread(input_image)
        else:
            # Use PIL for everything else (PNG, JPG, etc.)
            pil_img = Image.open(input_image)
            np_img = np.array(pil_img)
    except FileNotFoundError:
        logger.error(f"Image file not found: {input_image}")
        raise
    except Exception as e:
        logger.error(f"Error loading image {input_image}: {e}")
        raise

    # Rest of the processing remains the same
    if force_channel != -1:
        np_img = np_img[:, :, force_channel]
    elif np_img.ndim > 2:
        np_img = np.max(np_img, axis=2)
    if force_bit_depth != 0:
        np_img = convert_bitdepth(np_img, force_bit_depth)
    elif minmax_norm:
        np_img = (np_img - np.amin(np_img)) / (np.amax(np_img) - np.amin(np_img))

    return np_img