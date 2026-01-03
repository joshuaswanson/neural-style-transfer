"""
Neural Style Transfer - Core Algorithm

This module implements the neural style transfer optimization loop.
It iteratively updates the generated image to minimize a combination
of content loss, style loss, and total variation loss.
"""

import tensorflow as tf

from utils import (
    calculate_content_loss,
    calculate_loss,
    calculate_style_loss,
    calculate_variation_loss,
)


def perform_style_transfer(
    content_image: tf.Tensor,
    style_image: tf.Tensor,
    pretrained_cnn: tf.keras.Model,
    content_layers: list[str],
    style_layers: list[str],
    alpha: float,
    beta: float,
    variation_weight: float,
    num_iterations: int = 1000,
) -> tf.Tensor:
    """
    Perform neural style transfer to generate a stylized image.

    Args:
        content_image: Preprocessed content image tensor
        style_image: Preprocessed style image tensor
        pretrained_cnn: Pre-trained CNN model for feature extraction
        content_layers: List of layer names for content feature extraction
        style_layers: List of layer names for style feature extraction
        alpha: Weight for content loss
        beta: Weight for style loss
        variation_weight: Weight for total variation loss
        num_iterations: Number of optimization iterations

    Returns:
        Generated image as a numpy array (H, W, C)
    """
    # Build feature extraction model
    outputs = [pretrained_cnn.get_layer(layer).output for layer in content_layers]
    outputs.extend(pretrained_cnn.get_layer(layer).output for layer in style_layers)
    model = tf.keras.Model(inputs=pretrained_cnn.input, outputs=outputs)

    # Extract features from content and style images
    num_content_layers = len(content_layers)
    content_features = model(content_image)[:num_content_layers]
    style_features = model(style_image)[num_content_layers:]

    # Initialize generated image with content image
    generated_image = tf.Variable(content_image, dtype=tf.float32)

    # Set up optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=5.0)

    # Optimization loop
    for i in range(num_iterations):
        with tf.GradientTape() as tape:
            model_outputs = model(generated_image)
            content_output = model_outputs[:num_content_layers]
            style_outputs = model_outputs[num_content_layers:]

            content_loss = calculate_content_loss(content_features, content_output)
            style_loss = calculate_style_loss(style_features, style_outputs)
            variation_loss = calculate_variation_loss(generated_image)
            loss = calculate_loss(
                content_loss, style_loss, variation_loss, alpha, beta, variation_weight
            )

        gradients = tape.gradient(loss, generated_image)
        optimizer.apply_gradients([(gradients, generated_image)])

        # Clip pixel values to valid range
        clipped_image = tf.clip_by_value(
            generated_image, clip_value_min=0.0, clip_value_max=255.0
        )
        generated_image.assign(clipped_image)

        # Log progress
        if i % 100 == 0:
            print(
                f"Iteration: {i:4d}, "
                f"Total: {loss:.4e}, "
                f"Style: {style_loss:.4e}, "
                f"Content: {content_loss:.4e}, "
                f"Variation: {variation_loss:.4e}"
            )

    return generated_image.numpy()[0]
