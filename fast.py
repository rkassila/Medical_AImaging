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

        if disease_prediction >= 0.5:
            class_model_path = os.path.join(os.getcwd(), 'models', f'../models/{organ}_class_model.h5')
            class_model = load_model(class_model_path)
            class_prediction = class_model.predict(img_array)

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

            return {
                'organ': organ,
                'disease_status': 'diseased',
                'class_prediction': class_prediction.tolist(),
                'grad_cam_image': plot_gradcam(model=class_model, img_array=img_array, class_labels=class_labels)
            }
        else:
            return {'organ': organ, 'disease_status': 'healthy'}



#  layer_names = 'block1_conv2', 'block2_conv1'
