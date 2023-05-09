"""
Neural Style Transfer - Utility Functions

This module provides utility functions for image loading, preprocessing,
saving, and loss calculations for neural style transfer.
"""

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.efficientnet import (
    preprocess_input as efficientnet_preprocess,
)
from tensorflow.keras.applications.inception_v3 import (
    preprocess_input as inception_v3_preprocess,
)
from tensorflow.keras.applications.resnet50 import (
    preprocess_input as resnet50_preprocess,
)
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess

# Maximum image dimension (standard VGG input size)
# Reduced from 224 due to memory constraints during experimentation
MAX_DIM = 224


def load_image(path: str, model_name: str) -> tf.Tensor:
    """
    Load and preprocess an image for the specified model.

    Args:
        path: Path to the image file
        model_name: Name of the model ('VGG16', 'VGG19', 'InceptionV3',
                    'ResNet50', or 'EfficientNetB0')

    Returns:
        Preprocessed image tensor with shape (1, H, W, 3)

    Raises:
        ValueError: If model_name is not recognized
    """
    preprocess_functions = {
        "VGG16": vgg16_preprocess,
        "VGG19": vgg19_preprocess,
        "InceptionV3": inception_v3_preprocess,
        "ResNet50": resnet50_preprocess,
        "EfficientNetB0": efficientnet_preprocess,
    }

    if model_name not in preprocess_functions:
        raise ValueError(f"Unknown model: {model_name}")

    preprocess_fn = preprocess_functions[model_name]

    # Load image
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, dtype=tf.float32)

    # Resize while maintaining aspect ratio
    # Maximum dimension is MAX_DIM pixels
    original_shape = tf.shape(image)[:2]
    ratio = tf.cast(MAX_DIM, tf.float32) / tf.cast(
        tf.reduce_max(original_shape), tf.float32
    )
    new_shape = tf.cast(original_shape, tf.float32) * ratio
    image = tf.image.resize(
        image, tf.cast(new_shape, tf.int32), method=tf.image.ResizeMethod.BILINEAR
    )

    # Apply model-specific preprocessing and add batch dimension
    image = preprocess_fn(image * 255)
    image = tf.expand_dims(image, axis=0)

    return image


def deprocess_image(image: tf.Tensor) -> tf.Tensor:
    """
    Reverse VGG preprocessing to convert back to displayable image.

    VGG preprocessing subtracts ImageNet mean values and converts RGB to BGR.
    This function reverses those operations.

    Args:
        image: Preprocessed image tensor

    Returns:
        Deprocessed image tensor in RGB format
    """
    # ImageNet mean values (BGR order, as used in VGG preprocessing)
    imagenet_mean = tf.constant(
        [103.939, 116.779, 123.68], shape=(1, 1, 1, 3), dtype=tf.float32
    )

    # Add back the mean values
    image = tf.add(image, imagenet_mean)

    # Convert BGR back to RGB
    image = tf.reverse(image, axis=[-1])

    # Remove batch dimension
    image = tf.squeeze(image, axis=0)

    return image


def save_image(path: str, image: np.ndarray) -> None:
    """
    Save a generated image to disk.

    Args:
        path: Output file path
        image: Image array (H, W, C) or (1, H, W, C)
    """
    image = image.squeeze()
    image = deprocess_image(image)
    image = np.clip(image, 0, 255).astype("uint8")
    image = Image.fromarray(image)
    image.save(path)


def calculate_content_loss(
    content_features: list[tf.Tensor], generated_features: list[tf.Tensor]
) -> tf.Tensor:
    """
    Compute the content loss between content and generated images.

    Content loss is the mean squared error between feature representations
    of the content image and the generated image at specified layers.

    Args:
        content_features: Feature maps from the content image
        generated_features: Feature maps from the generated image

    Returns:
        Scalar content loss value
    """
    content_losses = [
        tf.reduce_sum(tf.square(content_features[i] - generated_features[i]))
        for i in range(len(content_features))
    ]
    return tf.reduce_mean(content_losses)


def gram_matrix(features: tf.Tensor) -> tf.Tensor:
    """
    Compute the Gram matrix of a feature map.

    The Gram matrix captures style information by computing correlations
    between different feature channels.

    Args:
        features: Feature map tensor of shape (batch, height, width, channels)

    Returns:
        Gram matrix of shape (channels, channels)
    """
    num_channels = features.get_shape()[3]
    flattened = tf.reshape(features, (-1, num_channels))
    return tf.matmul(tf.transpose(flattened), flattened)


def calculate_style_loss(
    style_features: list[tf.Tensor], generated_features: list[tf.Tensor]
) -> tf.Tensor:
    """
    Compute the style loss between style and generated images.

    Style loss compares Gram matrices of feature maps at multiple layers,
    capturing texture and pattern information.

    Args:
        style_features: Feature maps from the style image
        generated_features: Feature maps from the generated image

    Returns:
        Scalar style loss value
    """
    style_losses = []
    for i in range(len(style_features)):
        style_gram = gram_matrix(style_features[i])
        generated_gram = gram_matrix(generated_features[i])

        # Normalization factors
        num_channels = style_features[i].shape[3]
        num_pixels = style_features[i].shape[1] * style_features[i].shape[2]

        # Normalized style loss for this layer
        layer_loss = tf.reduce_sum(tf.square(style_gram - generated_gram)) / (
            4.0 * (num_channels**2) * (num_pixels**2)
        )
        style_losses.append(layer_loss)

    return tf.reduce_mean(style_losses)


def calculate_variation_loss(generated_image: tf.Tensor) -> tf.Tensor:
    """
    Compute the total variation loss to encourage spatial smoothness.

    This loss penalizes high-frequency noise in the generated image,
    resulting in smoother and more visually appealing outputs.

    Args:
        generated_image: Generated image tensor

    Returns:
        Scalar variation loss value
    """
    return tf.reduce_sum(tf.image.total_variation(generated_image))


def calculate_loss(
    content_loss: tf.Tensor,
    style_loss: tf.Tensor,
    variation_loss: tf.Tensor,
    alpha: float,
    beta: float,
    variation_weight: float,
) -> tf.Tensor:
    """
    Compute the total weighted loss for optimization.

    Args:
        content_loss: Content loss value
        style_loss: Style loss value
        variation_loss: Total variation loss value
        alpha: Weight for content loss
        beta: Weight for style loss
        variation_weight: Weight for variation loss

    Returns:
        Total weighted loss value
    """
    return (
        (alpha * content_loss)
        + (beta * style_loss)
        + (variation_weight * variation_loss)
    )
