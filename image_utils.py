import cv2
import numpy as np


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
    np_img = cv2.imread(input_image, -1)
    if force_channel != -1:
        np_img = np_img[:, :, force_channel]
    elif np_img.ndim > 2:
        np_img = np.max(np_img, axis=2)
    if force_bit_depth != 0:
        np_img = convert_bitdepth(input_image, force_bit_depth)
    elif minmax_norm:
        np_img = (np_img - np.amin(np_img)) / (np.amax(np_img) - np.amin(np_img))

    return np_img