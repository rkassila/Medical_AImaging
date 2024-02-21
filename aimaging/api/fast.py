from fastapi import FastAPI, UploadFile, File
from PIL import Image
from io import BytesIO
import numpy as np
import os
import gc
from tensorflow.keras.models import load_model
import tensorflow as tf
from aimaging.api.grad_cam import plot_gradcam
from aimaging.api.shap_image import generate_shap_image
from fastapi.responses import Response


app = FastAPI()

model_path = os.path.join(os.getcwd(), 'models', 'organ_detection_model.h5')
organ_detection_model = load_model(model_path)

@app.get("/")
def root():
    return {'greeting': 'Hello from here'}


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

        if organ == 'knee':
            class_labels = ['normal', 'soft_fluid', 'acl', 'bone_inf', 'chondral', 'fracture', 'intra', 'meniscal', 'patella', 'pcl']

        elif organ == 'brain':
            class_labels = ['normal','acute_infarct', 'chronic_infarct', 'extra',
                         'focal_flair_hyper', 'intra_brain', 'white_matter_changes']

        elif organ == 'shoulder':
            class_labels= ['normal','acj_oa', 'biceps_pathology', 'ghj_oa', 'labral_pathology',
                             'marrow_inflammation', 'osseous_lesion', 'post_op',
                             'soft_tissue_edema', 'soft_tissue_fluid_shoulder', 'supraspinatus_pathology']

        elif organ == 'spine':
            class_labels=['normal','cord_pathology', 'cystic_lesions', 'disc_pathology', 'osseous_abn']

        elif organ == 'lung':
            class_labels = ['normal','airspace_opacity', 'bronchiectasis', 'nodule',
                             'parenchyma_destruction', 'interstitial_lung_disease']


        model_class_path = os.path.join(os.getcwd(), 'models', f'../models/{organ}_class_model_with_normal.h5')
        disease_model = load_model(model_class_path)

        # Convert RGB image to grayscale
        gray_img_array = tf.image.rgb_to_grayscale(img_array)

        # Cast grayscale image to float32
        gray_img_array_float = tf.cast(gray_img_array, tf.float32)

        # Normalize pixel values by dividing by 255
        gray_img_array = tf.divide(gray_img_array_float, 255.0)

        disease_dict = disease_model.predict(gray_img_array).tolist()
        disease_prediction = class_labels[np.argmax(disease_dict, axis=1)[0]]

        is_healthy = 0


        if disease_prediction != 'normal':
            is_healthy = 1

        #model_path = os.path.join(os.getcwd(), 'models', f'../models/{organ}_bin_model.h5')
        #disease_model = load_model(model_path)

        #disease_prediction = disease_model.predict(img_array)[0][0]
        shap_image = generate_shap_image(model = organ_detection_model, image=img_array)

        if is_healthy >= 0.5:
            disease_status = 'diseased'

            grad_image = plot_gradcam(disease_model, gray_img_array, layer_name='conv2')
            app.state.grad_image = grad_image
            grad_image2 = plot_gradcam(disease_model, gray_img_array, layer_name='conv3')
            app.state.grad_image2 = grad_image2

            #del class_model
            #del disease_model

            gc.collect()

        else:
            disease_status = 'healthy'


        app.state.shap_image = shap_image

        return {
            'organ': organ,
            'disease_status': disease_status,
            'class_prediction':disease_dict,
            'best_prediction':disease_prediction}
            #{'organ': organ, 'disease_status': 'healthy'}

@app.get("/shap-image")
async def shap_image():
    return Response(app.state.shap_image, media_type="image/png")

@app.get("/grad-image")
async def grad_image():
    return Response(app.state.grad_image, media_type="image/png")

@app.get("/grad-image2")
async def grad_image2():
    return Response(app.state.grad_image2, media_type="image/png")


##Setting things to work locally
def main():
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

if __name__ == "__main__":
    main()
