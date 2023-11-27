from fastapi import FastAPI, UploadFile, File
from PIL import Image
from io import BytesIO
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import tensorflow as tf
from grad_cam import plot_gradcam
from aimaging.api.shap import generate_shap_image
from fastapi.responses import FileResponse, Response


app = FastAPI()

model_path = os.path.join(os.getcwd(), 'models', 'organ_detection_model.h5')
organ_detection_model = load_model(model_path)

@app.get("/")
def root():
    return {'greeting': 'Hello'}


@app.post("/organ_detection_model")
async def predict_organ(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(BytesIO(contents))
    img_array = np.array(img)  # Convert PIL Image to NumPy array
    img_array = tf.expand_dims(img_array, axis=0)

    if img_array is not None:
        predictions = organ_detection_model.predict(img_array)
        predicted_organ_index = np.argmax(predictions, axis=1)[0]

        organ_names = ["knee", "brain", "shoulder", "spine", "lung"]
        organ = organ_names[predicted_organ_index]

        model_path = os.path.join(os.getcwd(), 'models', f'../models/{organ}_bin_model.h5')
        disease_model = load_model(model_path)

        disease_prediction = disease_model.predict(img_array)[0][0]
        shap_image = generate_shap_image(model = organ_detection_model, image=img_array)
        if disease_prediction >= 0.5:
            class_model_path = os.path.join(os.getcwd(), 'models', f'../models/{organ}_class_model.h5')
            class_model = load_model(class_model_path)
            class_prediction = class_model.predict(img_array).tolist()
            disease_status = 'diseased'

            if organ == 'knee':
                class_labels = [ 'soft_fluid', 'acl', 'bone_inf', 'chondral',
                                'fracture', 'intra', 'meniscal', 'patella', 'pcl']
            elif organ == 'brain':
                class_labels = ['acute_infarct', 'chronic_infarct', 'extra',
                                'focal_flair_hyper', 'intra_brain', 'white_matter_changes']
            elif organ == 'shoulder':
                class_labels= ['acj_oa', 'biceps_pathology', 'ghj_oa', 'labral_pathology',
                            'marrow_inflammation', 'osseous_lesion', 'post_op',
                            'soft_tissue_edema', 'soft_tissue_fluid_shoulder', 'supraspinatus_pathology']
            elif organ == 'spine':
                class_labels=['cord_pathology', 'cystic_lesions', 'disc_pathology', 'osseous_abn']

            elif organ == 'lung':
                class_labels = ['airspace_opacity', 'bronchiectasis', 'nodule',
                                'parenchyma_destruction', 'interstitial_lung_disease']

            # {'organ': organ,
               # 'disease_status': 'diseased',
                #'class_prediction': class_prediction.tolist()}
        else:
            disease_status = 'healthy'
            class_prediction = None

        app.state.shap_image = shap_image
        app.state.grad_image = grad_image
        return {
            'organ': organ,
            'disease_status': disease_status,
            'class_prediction': class_prediction,
            #"shap_image": shap_image,
        }
            #{'organ': organ, 'disease_status': 'healthy'}

@app.get("/shap-image")
async def shap_image():
    return Response(app.state.shap_image, media_type="image/png")

@app.get("/grad-image")
async def grad_image():
    return Response(app.state.grad_image, media_type="image/png")
