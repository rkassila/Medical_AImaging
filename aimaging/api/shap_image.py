import shap
import shap.maskers
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

def generate_shap_image(image, model):

    image = np.squeeze(image, axis=0)

    class_names = ['knee', 'brain', 'shoulder', 'spine', 'lung']

    masker = shap.maskers.Image("blur(128,128)", shape=image.shape)

    explainer = shap.Explainer(model, masker, output_names=class_names)

    shap_values = explainer(np.array([image]), max_evals=200, batch_size=100,
                             outputs=shap.Explanation.argsort.flip[:5],
                             silent=True)

    plt.clf()

    plt.figure(facecolor='black')  # Set background color to black
    shap.image_plot(shap_values, pixel_values=np.array([image]), show=False)

    # Convert the plot to bytes
    buf = BytesIO()
    plt.savefig(buf, format='png', facecolor='white')  # Save with black background
    buf.seek(0)

    return buf.read()
