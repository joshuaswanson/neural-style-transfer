"""Pre-trained CNN backbones and the layers used for content and style features."""

from dataclasses import dataclass
from typing import Callable

import tensorflow as tf
from tensorflow.keras.applications import (
    efficientnet,
    inception_v3,
    resnet50,
    vgg16,
    vgg19,
)

VGG_STYLE_LAYERS = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1",
]


@dataclass(frozen=True)
class ModelSpec:
    build: Callable[..., tf.keras.Model]
    preprocess: Callable[[tf.Tensor], tf.Tensor]
    content_layers: list[str]
    style_layers: list[str]


MODELS = {
    "VGG16": ModelSpec(
        vgg16.VGG16, vgg16.preprocess_input, ["block5_conv2"], VGG_STYLE_LAYERS
    ),
    "VGG19": ModelSpec(
        vgg19.VGG19, vgg19.preprocess_input, ["block5_conv2"], VGG_STYLE_LAYERS
    ),
    "InceptionV3": ModelSpec(
        inception_v3.InceptionV3,
        inception_v3.preprocess_input,
        ["mixed7"],
        ["mixed2", "mixed3", "mixed4", "mixed5", "mixed6"],
    ),
    "ResNet50": ModelSpec(
        resnet50.ResNet50,
        resnet50.preprocess_input,
        ["conv5_block3_out"],
        [
            "conv1_relu",
            "conv2_block3_out",
            "conv3_block4_out",
            "conv4_block6_out",
            "conv5_block3_out",
        ],
    ),
    "EfficientNetB0": ModelSpec(
        efficientnet.EfficientNetB0,
        efficientnet.preprocess_input,
        ["block7b_project_conv"],
        [
            "block2a_project_conv",
            "block3a_project_conv",
            "block4a_project_conv",
            "block5a_project_conv",
            "block6a_project_conv",
            "block7a_project_conv",
        ],
    ),
}

FeatureExtractor = Callable[[tf.Tensor], tuple[list[tf.Tensor], list[tf.Tensor]]]


def build_feature_extractor(model_name: str) -> FeatureExtractor:
    """Return a function mapping an RGB image in [0, 255] to (content, style) features."""
    spec = MODELS[model_name]
    cnn = spec.build(weights="imagenet", include_top=False, input_shape=(None, None, 3))
    cnn.trainable = False
    layer_names = spec.content_layers + spec.style_layers
    extractor = tf.keras.Model(
        cnn.input, [cnn.get_layer(name).output for name in layer_names]
    )
    num_content = len(spec.content_layers)

    def extract(image: tf.Tensor) -> tuple[list[tf.Tensor], list[tf.Tensor]]:
        features = extractor(spec.preprocess(image))
        return features[:num_content], features[num_content:]

    return extract
