import tensorflow as tf
from keras.layers import Input
from keras.applications import vgg16, vgg19, inception_v3, resnet50, efficientnet
from style_transfer import perform_style_transfer
from utils import load_image, save_image
import os

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.set_visible_devices(physical_devices[0], "GPU")


def get_cnn_and_layers(model_name):
    input_shape = (None, None, 3)  # Variable input size, 3 channels (RGB)

    # Create an input tensor with the defined shape
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
        raise ValueError("Unknown model")
    return pretrained_cnn, content_layers, style_layers


models = ["VGG16", "VGG19"]
# models = ['VGG19', 'InceptionV3', 'ResNet50', 'EfficientNetB0']
# Commented out due to issues with system memory.

alpha = 5
beta = 1
variation_weight = 30

contents_directory = "contents"
styles_directory = "styles"

# Perform style transfer using each model
for model_name in models:
    pretrained_cnn, content_layers, style_layers = get_cnn_and_layers(model_name)

    for content in [c for c in os.listdir(contents_directory) if c.endswith(".jpg")]:
        content_image = load_image(f"{contents_directory}/{content}", model_name)

        for style in [s for s in os.listdir(styles_directory) if s.endswith('.jpg')]:
            style_image = load_image(f"{styles_directory}/{style}", model_name)

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

            save_image(
                f"output_{content.replace('.jpg','')}_{style.replace('.jpg','')}_{model_name}.jpg",
                output_image,
            )
