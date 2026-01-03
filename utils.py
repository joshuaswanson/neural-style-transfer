import numpy as np
import tensorflow as tf
from keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from keras.applications.vgg19 import preprocess_input as vgg19_preprocess
from keras.applications.inception_v3 import preprocess_input as inception_v3_preprocess
from keras.applications.resnet50 import preprocess_input as resnet50_preprocess
from keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from PIL import Image


def load_image(path, model_name):
    """
    Load and preprocess an image from the given path using the appropriate preprocessing function.
    """

    if model_name == "VGG16":
        preprocess_fn = vgg16_preprocess
    elif model_name == "VGG19":
        preprocess_fn = vgg19_preprocess
    elif model_name == "InceptionV3":
        preprocess_fn = inception_v3_preprocess
    elif model_name == "ResNet50":
        preprocess_fn = resnet50_preprocess
    elif model_name == "EfficientNetB0":
        preprocess_fn = efficientnet_preprocess
    else:
        raise ValueError("Unknown model")

    max_dim = 244

    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, dtype=tf.float32)

    # Resize the images while maintaining the aspect ratio.
    # Maximum dimension is 244px. This is the default size of input images to VGG.
    # Heavily inspired by the TensorFlow guide to Neural Style Transfer
    original_shape = tf.shape(image)[:2]  # Get the original shape of the image
    ratio = tf.cast(max_dim, tf.float32) / tf.cast(
        tf.reduce_max(original_shape), tf.float32
    )
    new_shape = tf.cast(original_shape, tf.float32) * ratio  # Calculate the new shape
    image = tf.image.resize(
        image, tf.cast(new_shape, tf.int32), method=tf.image.ResizeMethod.BILINEAR
    )

    image = preprocess_fn(image * 255)
    image = tf.expand_dims(image, axis=0)
    return image


def deprocess_input(image):
    image = tf.Variable(image)
    constant_tensor = tf.constant(
        [103.939, 116.779, 123.68], shape=(1, 1, 1, 3), dtype=tf.float32
    )
    image = tf.add(image, constant_tensor)
    image = tf.reverse(image, axis=[-1])
    image = tf.squeeze(image, axis=0)
    return image


# Save an image
def save_image(path, image):
    image = image.squeeze()
    image = deprocess_input(image)
    image = np.clip(image, 0, 255).astype("uint8")
    image = Image.fromarray(image)
    image.save(path)


def calculate_content_loss(content_features, generated):
    """
    Compute the content loss between the content image and the generated image.
    """
    content_losses = [
        tf.reduce_sum(tf.square(content_features[i] - generated[i]))
        for i in range(len(content_features))
    ]
    content_losses = tf.reduce_mean(content_losses)
    return content_losses


def gram_matrix(x):
    """
    Compute the Gram matrix of the given feature map.
    """
    num_channels = x.get_shape()[3]
    features = tf.reshape(x, (-1, num_channels))
    return tf.matmul(tf.transpose(features), features)


def calculate_style_loss(style_features, generated):
    """
    Compute the style loss between the style image and the generated image.
    """
    style_losses = []
    for i in range(len(style_features)):
        s = gram_matrix(style_features[i])
        g = gram_matrix(generated[i])
        n = style_features[i].shape[3]
        m = style_features[i].shape[1] * style_features[i].shape[2]
        style_losses.append(
            tf.reduce_sum(tf.square(s - g)) / (4.0 * (n**2) * (m**2))
        )
    style_loss = tf.reduce_mean(style_losses)
    return style_loss


def calculate_variation_loss(generated):
    """
    Compute the total variation loss to encourage spatial smoothness in the generated image.
    """
    return tf.reduce_sum(tf.image.total_variation(generated))


def calculate_loss(
    content_loss, style_loss, variation_loss, alpha, beta, variation_weight
):
    """
    Compute the overall loss for the optimization process.
    """
    return (
        (alpha * content_loss)
        + (beta * style_loss)
        + (variation_weight * variation_loss)
    )
