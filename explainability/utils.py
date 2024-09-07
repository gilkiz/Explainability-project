import os
from pathlib import Path
import re

import cv2
import torch
from torch import Tensor

from explainability.image_processing import min_max_normalization

IMAGE_CUT_LEFT = 270
IMAGE_CUT_RIGHT = 1150


def npy_to_tensor(npy: str):
    """
    Converts a numpy array to a PyTorch tensor.

    Args:
        npy: A numpy array of shape [H, W, C], where W is the width, H is the height and C is the number of channels.

    Returns: A PyTorch tensor of shape [1, C, H, W].

    """
    return torch.from_numpy(npy).permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32)


def tensor_to_npy(tensor: Tensor):
    """
    Converts a PyTorch tensor to a numpy array.

    Args:
        tensor: A PyTorch tensor of shape [1, C, H, W].

    Returns: A numpy array of shape [H, W, C], where W is the width, H is the height and C is the number of channels.

    """
    return tensor.squeeze().permute(1, 2, 0).cpu().numpy()


def tensor_to_image(tensor: Tensor, gamma: float = 1.0):
    """
    Converts a PyTorch tensor to an image.

    Args:
        tensor: A PyTorch tensor of shape [1, C, H, W].
        gamma: The gamma value to apply to the image.

    Returns: A numpy array of shape [H, W, C], where W is the width, H is the height and C is the number of channels.

    """
    normalizes_tensor = min_max_normalization(tensor) ** gamma * 255
    return tensor_to_npy(normalizes_tensor).astype("uint8")


def load_image(path: str, width: int, height: int, cut_image: bool = False):
    """
    Loads an image from a file and resizes it to a given width and height.

    Args:
        path: A path to an image file.
        width: The width to resize the image to.
        height: The height to resize the image to.
        cut_image: Should the image be cut to a specific size before resizing.

    Returns: A numpy array of shape [H, W, C], where W is the width, H is the height and C is the number of channels.

    """
    img = cv2.imread(path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if cut_image:
        img = img[IMAGE_CUT_LEFT:IMAGE_CUT_RIGHT, :, :]
    img = cv2.resize(img, (width, height))

    return img


def load_model(path: Path, model_name: str, device: str):
    """
    Loads a model from a file.

    Args:
        path: A path to the directory containing the model file.
        model_name: The name of the model file.
        device: The device to load the model to.

    Returns: A PyTorch model.

    """
    model_path = os.path.join(path, model_name)
    saved_state = torch.load(model_path, map_location=torch.device(device))

    model = saved_state["full_model"]
    model.load_state_dict(saved_state["model_state"])
    model.name = re.findall(r"(?<=GPU\d_).+(?=\.pt)", model_name)[0]
    model.to(device)

    return model
