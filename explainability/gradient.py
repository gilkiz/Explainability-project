from torch import Tensor, concatenate, randn_like
from torch.nn import Module
import cv2

from explainability.utils import tensor_to_image


def evaluate_jacobian_matrix(x: Tensor, model: Module):
    """

    Args:
        x: A batch of a single image of shape [1, C, W, H], there B is batch size, C is the number of
        channels and W and H are the width and height of the image.
        model: A model to calculate the Jacobian matrix of.

    Returns: A tensor of shape [1, C, W, H] representing the Jacobian matrix of the
    model with regard to the input tensor.

    Note:
        Assumes a model using a single device for all its parameters (One GPU/CPU).
    """
    if len(x.shape) != 4:
        raise RuntimeError(
            "A tensor of shape [1, C, W, H], there B is batch size, C "
            "is the number of channels and W and H are the width "
            "and height of the image. Is expected"
        )

    if x.shape[0] != 1:
        raise RuntimeError("A single image is expected")

    model_mode = model.training
    x = x.to(next(model.parameters()).device).requires_grad_(True)
    model.zero_grad()

    model.eval()
    output_autoencoder, output_class, output_domain = model(x)
    output_class.backward()
    model.training = model_mode

    return x.grad


def smooth_grad(image_tensor, model, noise_ratio=0.2, num_noise_images=10):
    """
    Implements the SmoothGrad (https://arxiv.org/pdf/1706.03825.pdf) method
     for estimating the Jacobian matrix of a model.

    Args:
        image_tensor: A batch of a single image of shape [1, C, W, H], there B is batch size, C is the number of
        model: A model to calculate the Jacobian matrix of.
        noise_ratio: The ratio of the standard deviation of the noise to the standard deviation of the image.
        num_noise_images: The number of noise images to average the Jacobian matrix over.

    Returns: A tensor of shape [1, C, W, H] representing the noise smoothed
    Jacobian matrix of the model with regard to the input tensor.

    """
    sigma = noise_ratio * image_tensor.std()

    J = concatenate([evaluate_jacobian_matrix(image_tensor + sigma * randn_like(image_tensor), model)
                           for _ in range(num_noise_images)], dim=0)

    return J.mean(dim=0, keepdim=True)


def jacobian_to_mask(J, gamma=0.5, blur_kernel=(15, 15)):
    """
    Converts a Jacobian matrix to a mask by taking the absolute value of the Jacobian matrix,
    applying a gamma correction and blurring the result.

    Args:
        J: A tensor of shape [1, C, W, H] representing the Jacobian matrix.
        gamma: The gamma value to apply to the mask.
        blur_kernel: The height and width of the kernel to use for blurring the mask.

    Returns:

    """
    return cv2.blur(tensor_to_image(J.abs(), gamma=gamma), blur_kernel)
