import shap
import numpy as np

def generateshap(image, model, class_names):
    img = np.asarray(image)
    #class_names = ['lung','brain','knee','shoulder','spine']

    masker = shap.maskers.Image("blur(128,128)", shape=img.shape)

    explainer = shap.Explainer(model, masker, output_names=class_names)

    shap_values = explainer(np.array([img]), max_evals=500, batch_size=50,
                                    outputs=shap.Explanation.argsort.flip[:5],
                                    silent=True)

    return shap.image_plot(shap_values, pixel_values=np.array([img]))
