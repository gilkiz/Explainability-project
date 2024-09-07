import numpy as np
import cv2
import torch
from torch import Tensor


def remove_shadows(img, dilation_kernel=2, gamma=3.0):
    """
    Removes shadows from an image using dilation and gamma correction.

    Args:
        img: A numpy array of shape [H, W, C], where W is the width, H is the height and C is the number of channels.
        dilation_kernel: The size of the kernel to use for dilation.
        gamma: The gamma value to apply to the image.

    Returns: A numpy array of shape [H, W, C], where W is the width, H is the height and C is the number of channels.

    """
    rgb_planes = cv2.split(img)

    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((dilation_kernel, dilation_kernel), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        norm_img = 255 - cv2.absdiff(plane, bg_img) * 3
        result_norm_planes.append(norm_img)

    return (255 * (cv2.merge(result_norm_planes) / 255) ** gamma).astype(np.uint8)


def alpha_normalization(img):
    """
    Normalizes an image to the range [0, 255] by alpha scaling.

    Args:
        img: The image to normalize.

    Returns: The normalized image.

    """
    return ((255/img.max())*img).astype(np.uint8)


def mask_overlay(img, mask, alpha=0.7):
    """
    Overlays a mask on an image using alpha blending.

    Args:
        img: The image to overlay the mask on.
        mask: The mask to overlay on the image.
        alpha: The alpha value to use for blending.

    Returns: The image with the mask overlayed on it.

    """
    return (img*(1-alpha) + mask*alpha).astype(np.uint8)


def min_max_normalization(x: Tensor):
    """
    Applies per color channel min-max normalization to a batch of images.

    For a tensor X of shape [B, W, H], an image batch  single channel:

    X_normalized = (X-min(x))/(max(X) - min(x))

    Args:
        x: A batch of images of shape [B, C, W, H], there B is batch size, C is the number of
        channels and W and H are the width and height of the image.

    Returns: A batch of images of shape [B, C, W, H] normalized

    """
    if len(x.shape) != 4:
        raise RuntimeError(
            "A tensor of shape [B, C, W, H], there B is batch size, C "
            "is the number of channels and W and H are the width "
            "and height of the image. Is expected"
        )

    num_channels = x.shape[1]

    with torch.no_grad():
        for i in range(num_channels):
            x[:, i, :, :] = (x[:, i, :, :] - x[:, i, :, :].min()) / (
                x[:, i, :, :].max() - x[:, i, :, :].min()
            )

    return x
