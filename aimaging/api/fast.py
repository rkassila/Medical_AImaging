from fastapi import FastAPI, UploadFile, File
from PIL import Image
from io import BytesIO
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

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
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = organ_detection_model.predict(img_array)
    predicted_organ_index = np.argmax(predictions, axis=1)[0]

    # organ list
    organ_names = ["knee", "brain", "shoulder", "spine", "lung"]

    # get organ name
    predicted_organ = organ_names[predicted_organ_index]

    return {"predicted_organ": predicted_organ}
