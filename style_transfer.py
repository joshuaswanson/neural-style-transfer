from utils import *


# Perform style transfer using the specified model
def perform_style_transfer(
    content_image,
    style_image,
    pretrained_cnn,
    content_layers,
    style_layers,
    alpha,
    beta,
    variation_weight,
    num_iterations=1000,
):
    outputs = [pretrained_cnn.get_layer(layer).output for layer in content_layers]
    outputs.extend(pretrained_cnn.get_layer(layer).output for layer in style_layers)

    model = tf.keras.Model(inputs=pretrained_cnn.input, outputs=outputs)

    content_features = model(content_image)[: len(content_layers)]
    style_features = model(style_image)[len(content_layers) :]

    generated_image = tf.Variable(content_image, dtype=tf.float32)

    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=5.0)

    tape = tf.GradientTape(persistent=True)
    for i in range(num_iterations):
        with tape:
            model_outputs = model(generated_image)
            content_output = model_outputs[: len(content_layers)]
            style_outputs = model_outputs[len(content_layers) :]

            content_loss = calculate_content_loss(content_features, content_output)
            style_loss = calculate_style_loss(style_features, style_outputs)
            variation_loss = calculate_variation_loss(generated_image)
            loss = calculate_loss(
                content_loss, style_loss, variation_loss, alpha, beta, variation_weight
            )

        gradients = tape.gradient(loss, generated_image)
        optimizer.apply_gradients([(gradients, generated_image)])
        clipped_image = tf.clip_by_value(
            generated_image, clip_value_min=0.0, clip_value_max=255.0
        )
        generated_image.assign(clipped_image)

        if i % 100 == 0:
            print(
                "Iteration: {}, Total loss: {:.4e}, Style loss: {:.4e}, Content loss: {:.4e}, Variation loss: {:.4e}".format(
                    i, loss, style_loss, content_loss, variation_loss
                )
            )

    return generated_image.numpy()[0]
