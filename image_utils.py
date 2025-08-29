import numpy as np
import tifffile
from PIL import Image
from pathlib import Path
import os


def convert_bitdepth(image, bitdepth):
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


def read_grayscale_image(input_image, force_channel = -1, force_bit_depth = 0, minmax_norm = False):
     
    # Check file extension to choose optimal loading method
    file_ext = Path(input_image).suffix.lower()
    
    if file_ext in ['.tiff', '.tif']:
        # Use tifffile for TIFF images
        np_img = tifffile.imread(input_image)
    else:
        # Use PIL for everything else (PNG, JPG, etc.)
        pil_img = Image.open(input_image)
        np_img = np.array(pil_img)
    
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