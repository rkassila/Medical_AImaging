import shap
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from fastapi.responses import Response


def generateshap(image, model):
    #img = np.asarray(image)
    class_names = ['lung','brain','knee','shoulder','spine']

    masker = shap.maskers.Image("blur(128,128)", shape=image.shape)

    explainer = shap.Explainer(model, masker, output_names=class_names)

    shap_values = explainer(np.array([image]), max_evals=500, batch_size=50,
                                    outputs=shap.Explanation.argsort.flip[:5],
                                    silent=True)

    return shap.image_plot(shap_values, pixel_values=np.array([image]))


def generate_shap_image(image, model):

    image = np.squeeze(image, axis=0)

    class_names = ['lung', 'brain', 'knee', 'shoulder', 'spine']

    masker = shap.maskers.Image("blur(128,128)", shape=image.shape)

    explainer = shap.Explainer(model, masker, output_names=class_names)

    shap_values = explainer(np.array([image]), max_evals=500, batch_size=50,
                             outputs=shap.Explanation.argsort.flip[:5],
                             silent=True)

    # Create a custom plot using Matplotlib
    plt.figure()
    shap.image_plot(shap_values, pixel_values=np.array([image]), show=False)

    # Convert the plot to bytes
    buf = BytesIO()
    plt.savefig(buf, format='png')
    #buf.seek(0)

    # Return the image as a StreamingResponse
    return Response(buf, media_type="image/png")
