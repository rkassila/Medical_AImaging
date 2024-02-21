import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

def plot_gradcam(model, img_array, layer_name):
    def generate_gradcam(model, img_array, layer_name):
        grad_model = tf.keras.Model([model.inputs], [model.get_layer(layer_name).output, model.output])

        with tf.GradientTape() as tape:
            conv_output, predictions = grad_model(img_array)
            loss = predictions[:, 1]

        grads = tape.gradient(loss, conv_output)
        guided_grads = (tf.cast(conv_output > 0, "float32") * tf.cast(grads > 0, "float32") * grads)

        weights = tf.reduce_mean(guided_grads, axis=(0, 1, 2))
        cam = np.dot(conv_output[0], weights)

        return cam, conv_output, predictions

    cam, conv_output, predictions = generate_gradcam(model, img_array, layer_name)

    cam, _, _ = generate_gradcam(model, img_array, layer_name)

    epsilon = 1e-10  # Small epsilon value to prevent division by zero
    cam_range = np.max(cam) - np.min(cam)
    cam = (cam - np.min(cam)) / (cam_range + epsilon)

    plt.clf()

    plt.figure(facecolor='black')
    #plt.imshow(img_array[0], alpha=0.4)
    #plt.imshow(cam, cmap='jet', alpha=0.8)
    plt.imshow(cam, cmap='jet_r', alpha=0.8)
    plt.title(f'GradCAM ({layer_name})', color='black')

    buf = BytesIO()
    plt.savefig(buf, format='png', facecolor='white')  # Save with white background
    buf.seek(0)


    return buf.read()
