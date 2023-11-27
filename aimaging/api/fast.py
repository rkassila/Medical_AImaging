from fastapi import FastAPI, UploadFile, File
from PIL import Image
from io import BytesIO
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import tensorflow as tf

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

            return {
                'organ': organ,
                'disease_status': 'diseased',
                'class_prediction': class_prediction.tolist()
            }
        else:
            return {'organ': organ, 'disease_status': 'healthy'}
