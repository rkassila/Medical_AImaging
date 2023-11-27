import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from fastapi.responses import Response
from tensorflow.keras.applications import ResNet50, imagenet_utils
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

def plot_gradcam(model, img_array, class_labels):
    def generate_gradcam(img_array):
        layer_names = ['block1_conv2', 'block2_conv1']

        grad_model = tf.keras.Model([model.inputs], [model.get_layer(layer_name).output for layer_name in layer_names] + [model.output])

        with tf.GradientTape() as tape:
            outputs = grad_model(img_array)
            conv_outputs, predictions = outputs[:-1], outputs[-1]
            grads = [tape.gradient(predictions[:, 1], conv_output) for conv_output in conv_outputs]

        guided_grads_list = [(tf.cast(conv_output > 0, "float32") * tf.cast(grad > 0, "float32") * grad) for conv_output, grad in zip(conv_outputs, grads)]
        weights_list = [tf.reduce_mean(guided_grad, axis=(0, 1, 2)) for guided_grad in guided_grads_list]
        cams_list = [np.dot(conv_output[0], weights) for conv_output, weights in zip(conv_outputs, weights_list)]

        return cams_list, predictions

    cams_list, predictions = generate_gradcam(np.expand_dims(img_array, axis=0))

    layer_names = ['block1_conv2', 'block2_conv1']

    for layer_name, cam in zip(layer_names, cams_list):
        if cam is not None:
            cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
            plt.imshow(img_array[0], alpha=0.8)
            plt.imshow(cam, cmap='jet', alpha=0.5)
            plt.title(f'Original Image: {class_labels[np.argmax(model.predict(img_array)[0])]} | Confidence: {np.max(predictions[0]):.2%}')
            plt.show()


def generate_gradcam(img_array, model, layer_name, class_labels):
    grad_model = tf.keras.Model([model.inputs], [model.get_layer(layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        grads = tape.gradient(predictions[:, 1], conv_output)

    guided_grads = tf.cast(conv_output > 0, "float32") * tf.cast(grads > 0, "float32") * grads
    weights = tf.reduce_mean(guided_grads, axis=(0, 1, 2))
    cam = np.dot(conv_output[0], weights)

    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
    plt.figure()
    plt.imshow(img_array[0], alpha=0.8)
    plt.imshow(cam, cmap='jet', alpha=0.5)
    plt.title(f'Original Image: {class_labels[np.argmax(model.predict(img_array)[0])]} | Confidence: {np.max(predictions[0]):.2%}')
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    return Response(buf.read(), media_type="image/png")
#class_labels = ["Normal", "Nodule", "Airspaces", "Bronch", "Parenchyma"]
#generate_and_plot_gradcam(fine_tuned_model, preprocess_input(np.expand_dims(test_image_array, axis=0)), layer_name, class_labels)



layer_names = ['block1_conv2', 'block2_conv1']

def image_gradcam(model, img_array, layer_name):
    grad_model = tf.keras.Model([model.inputs], [model.get_layer(layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_output, _ = grad_model(img_array)
        loss = model(img_array)

    grads = tape.gradient(loss, conv_output)

    if grads is None:
        print("Gradients are None. Skipping GradCAM computation.")
        return

    guided_grads = (tf.cast(conv_output > 0, "float32") * tf.cast(grads > 0, "float32") * grads)

    weights = tf.reduce_mean(guided_grads, axis=(0, 1, 2))
    cam = np.dot(conv_output[0], weights)

    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))

    plt.figure()
    plt.imshow(img_array[0], alpha=0.8)
    plt.imshow(cam, cmap='jet', alpha=0.5)
    plt.title(f'GradCAM ({layer_name})')

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    return buf.read()
