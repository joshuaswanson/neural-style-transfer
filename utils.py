"""Image I/O and loss functions for neural style transfer."""

import numpy as np
import tensorflow as tf
from PIL import Image

MAX_DIM = 224


def load_image(path: str) -> tf.Tensor:
    """Load an RGB image in [0, 255] with shape (1, H, W, 3), resized so max(H, W) == MAX_DIM."""
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, dtype=tf.float32, expand_animations=False)
    shape = tf.cast(tf.shape(image)[:2], tf.float32)
    new_shape = tf.cast(shape * MAX_DIM / tf.reduce_max(shape), tf.int32)
    image = tf.image.resize(image, new_shape)
    return image[tf.newaxis] * 255.0


def save_image(path: str, image: tf.Tensor) -> None:
    """Save an RGB image in [0, 255] with shape (1, H, W, 3)."""
    pixels = np.clip(np.asarray(image)[0], 0, 255).astype(np.uint8)
    Image.fromarray(pixels).save(path)


def calculate_content_loss(
    content_features: list[tf.Tensor], generated_features: list[tf.Tensor]
) -> tf.Tensor:
    return tf.reduce_mean(
        [
            tf.reduce_sum(tf.square(target - generated))
            for target, generated in zip(content_features, generated_features)
        ]
    )


def gram_matrix(features: tf.Tensor) -> tf.Tensor:
    """Channel correlation matrix, averaged over spatial positions."""
    num_channels = features.shape[-1]
    flattened = tf.reshape(features, (-1, num_channels))
    num_positions = tf.cast(tf.shape(flattened)[0], tf.float32)
    return tf.matmul(flattened, flattened, transpose_a=True) / num_positions


def calculate_style_loss(
    style_features: list[tf.Tensor], generated_features: list[tf.Tensor]
) -> tf.Tensor:
    """Gatys et al. style loss, with Gram matrices normalized per position.

    Normalizing by the number of positions lets the style image and the generated
    image have different sizes.
    """
    layer_losses = []
    for target, generated in zip(style_features, generated_features):
        num_channels = target.shape[-1]
        difference = gram_matrix(target) - gram_matrix(generated)
        layer_losses.append(
            tf.reduce_sum(tf.square(difference)) / (4.0 * num_channels**2)
        )
    return tf.reduce_mean(layer_losses)


def calculate_variation_loss(image: tf.Tensor) -> tf.Tensor:
    return tf.reduce_sum(tf.image.total_variation(image))
