"""Optimization loop for neural style transfer."""

import tensorflow as tf

from models import FeatureExtractor
from utils import calculate_content_loss, calculate_style_loss, calculate_variation_loss


def perform_style_transfer(
    content_image: tf.Tensor,
    style_image: tf.Tensor,
    extract_features: FeatureExtractor,
    alpha: float,
    beta: float,
    variation_weight: float,
    num_iterations: int = 1000,
    learning_rate: float = 5.0,
) -> tf.Tensor:
    """Optimize an image, initialized from the content image, toward the style image.

    Images are RGB in [0, 255] with shape (1, H, W, 3).
    """
    content_targets, _ = extract_features(content_image)
    _, style_targets = extract_features(style_image)

    generated_image = tf.Variable(content_image)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    optimizer.build([generated_image])

    @tf.function
    def step():
        with tf.GradientTape() as tape:
            content_outputs, style_outputs = extract_features(generated_image)
            content_loss = calculate_content_loss(content_targets, content_outputs)
            style_loss = calculate_style_loss(style_targets, style_outputs)
            variation_loss = calculate_variation_loss(generated_image)
            loss = (
                alpha * content_loss
                + beta * style_loss
                + variation_weight * variation_loss
            )
        gradient = tape.gradient(loss, generated_image)
        optimizer.apply_gradients([(gradient, generated_image)])
        generated_image.assign(tf.clip_by_value(generated_image, 0.0, 255.0))
        return loss, content_loss, style_loss, variation_loss

    for i in range(num_iterations):
        loss, content_loss, style_loss, variation_loss = step()
        if i % 100 == 0:
            print(
                f"Iteration: {i:4d}, "
                f"Total: {loss:.4e}, "
                f"Style: {style_loss:.4e}, "
                f"Content: {content_loss:.4e}, "
                f"Variation: {variation_loss:.4e}"
            )

    return tf.convert_to_tensor(generated_image)
