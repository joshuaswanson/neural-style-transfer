"""
Neural Style Transfer - Main Entry Point

This module loads pre-trained CNN models and orchestrates the style transfer process.
Based on "A Neural Algorithm of Artistic Style" by Gatys et al. (2015).

Course: CSE 455 - Computer Vision
University of Washington
"""

import os

import tensorflow as tf
from tensorflow.keras.applications import (
    efficientnet,
    inception_v3,
    resnet50,
    vgg16,
    vgg19,
)
from tensorflow.keras.layers import Input

from style_transfer import perform_style_transfer
from utils import load_image, save_image

# GPU Configuration
# Attempt to use GPU if available, otherwise fall back to CPU
physical_devices = tf.config.list_physical_devices("GPU")
if physical_devices:
    try:
        tf.config.set_visible_devices(physical_devices[0], "GPU")
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print(f"Using GPU: {physical_devices[0]}")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("No GPU found. Using CPU.")


def get_model_and_layers(model_name: str):
    """
    Load a pre-trained CNN and return the model with appropriate layer names.

    Args:
        model_name: Name of the model ('VGG16', 'VGG19', 'InceptionV3',
                    'ResNet50', or 'EfficientNetB0')

    Returns:
        Tuple of (pretrained_model, content_layer_names, style_layer_names)

    Raises:
        ValueError: If model_name is not recognized
    """
    input_shape = (None, None, 3)  # Variable input size, 3 channels (RGB)
    input_tensor = Input(shape=input_shape)

    if model_name == "VGG16":
        pretrained_cnn = vgg16.VGG16(
            weights="imagenet", include_top=False, input_tensor=input_tensor
        )
        content_layers = ["block5_conv2"]
        style_layers = [
            "block1_conv1",
            "block2_conv1",
            "block3_conv1",
            "block4_conv1",
            "block5_conv1",
        ]
    elif model_name == "VGG19":
        pretrained_cnn = vgg19.VGG19(
            weights="imagenet", include_top=False, input_tensor=input_tensor
        )
        content_layers = ["block5_conv2"]
        style_layers = [
            "block1_conv1",
            "block2_conv1",
            "block3_conv1",
            "block4_conv1",
            "block5_conv1",
        ]
    elif model_name == "InceptionV3":
        pretrained_cnn = inception_v3.InceptionV3(
            weights="imagenet", include_top=False, input_tensor=input_tensor
        )
        content_layers = ["mixed7"]
        style_layers = ["mixed2", "mixed3", "mixed4", "mixed5", "mixed6"]
    elif model_name == "ResNet50":
        pretrained_cnn = resnet50.ResNet50(
            weights="imagenet", include_top=False, input_tensor=input_tensor
        )
        content_layers = ["conv5_block3_out"]
        style_layers = [
            "conv1_relu",
            "conv2_block3_out",
            "conv3_block4_out",
            "conv4_block6_out",
            "conv5_block3_out",
        ]
    elif model_name == "EfficientNetB0":
        pretrained_cnn = efficientnet.EfficientNetB0(
            weights="imagenet", include_top=False, input_tensor=input_tensor
        )
        content_layers = ["block7b_project_conv"]
        style_layers = [
            "block2a_project_conv",
            "block3a_project_conv",
            "block4a_project_conv",
            "block5a_project_conv",
            "block6a_project_conv",
            "block7a_project_conv",
        ]
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return pretrained_cnn, content_layers, style_layers


def main():
    """Run neural style transfer on all content/style image combinations."""

    # Models to use for style transfer
    # Note: InceptionV3, ResNet50, and EfficientNetB0 are commented out due to
    # memory constraints. Uncomment to experiment if sufficient memory is available.
    models = ["VGG16", "VGG19"]
    # models = ["VGG19", "InceptionV3", "ResNet50", "EfficientNetB0"]

    # Loss function weights
    alpha = 5              # Content weight
    beta = 1               # Style weight
    variation_weight = 30  # Total variation weight (encourages smoothness)

    # Directory paths
    contents_directory = "contents"
    styles_directory = "styles"
    output_directory = "output"

    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Perform style transfer for each model
    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Processing with {model_name}")
        print(f"{'='*60}")

        pretrained_cnn, content_layers, style_layers = get_model_and_layers(model_name)

        content_files = sorted(
            [c for c in os.listdir(contents_directory) if c.endswith(".jpg")]
        )
        style_files = sorted(
            [s for s in os.listdir(styles_directory) if s.endswith(".jpg")]
        )

        for content_file in content_files:
            content_image = load_image(
                f"{contents_directory}/{content_file}", model_name
            )

            for style_file in style_files:
                print(f"\nContent: {content_file}, Style: {style_file}")

                style_image = load_image(
                    f"{styles_directory}/{style_file}", model_name
                )

                output_image = perform_style_transfer(
                    content_image,
                    style_image,
                    pretrained_cnn,
                    content_layers,
                    style_layers,
                    alpha,
                    beta,
                    variation_weight,
                )

                # Generate output filename
                content_name = content_file.replace(".jpg", "")
                style_name = style_file.replace(".jpg", "")
                output_path = (
                    f"{output_directory}/output_{content_name}_{style_name}_{model_name}.jpg"
                )

                save_image(output_path, output_image)
                print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
