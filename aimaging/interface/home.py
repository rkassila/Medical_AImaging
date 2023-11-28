import streamlit as st
from PIL import Image
import requests
import tempfile
import io
import time

def app():
    # Set title alignment and size
    st.title("Organ Disease Detector 🔍")

    # Read the image
    image = Image.open('aimaging/interface/streamlit_bg.png')

    # Display the image with wide layout
    st.image(image, use_column_width=True)

    uploaded_image = st.file_uploader("Upload an image of your organ", type=["jpg", "jpeg", "png"])


    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        image = image.resize((224, 224))
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Button to trigger the scanning process
        if st.button("Scan"):
            # Perform the scanning process here
            result = scan_image(image)
            with st.spinner('Analyzing the picture...'):
                time.sleep(25)
                st.success('Done!')

            # Display the result
            if result is not None:
                # Display the prediction details
                st.write("### Analysis Result:")
                st.write(f"- Organ: {result.get('Organ', 'N/A')}")
                st.write(f"- Disease Status: {result.get('Disease Status', 'N/A')}")
                st.write(f"- Class Prediction: {result.get('Class Prediction', 'N/A')}")

                # Display the SHAP image
                shap_url = "https://aimaging11-uz7skuvrea-ew.a.run.app/shap-image"
                response = requests.get(shap_url)

                if response.status_code == 200:
                    try:
                        # Convert binary image data to BytesIO
                        image_bytes = io.BytesIO(response.content)
                        # Attempt to open the image
                        shap_image = Image.open(image_bytes)
                        st.image(shap_image, caption="SHAP Image", use_column_width=True)
                    except Exception as e:
                        st.error(f"Error opening SHAP image: {e}")
                else:
                    st.error(f"Error retrieving SHAP image. Status code: {response.status_code}")

                # Display the Grad image
                grad_url = "https://aimaging11-uz7skuvrea-ew.a.run.app/grad-image"
                response = requests.get(grad_url)

                if response.status_code == 200:
                    try:
                        # Convert binary image data to BytesIO
                        image_bytes = io.BytesIO(response.content)
                        # Attempt to open the image
                        grad_image = Image.open(image_bytes)
                        st.image(grad_image, caption="Grad Image", use_column_width=True)
                    except Exception as e:
                        st.error(f"Error opening Grad image: {e}")
                else:
                    st.error(f"Error retrieving Grad image. Status code: {response.status_code}")


# Function to scan the image
def scan_image(image):

    # Save the image to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".png") as temp_file:
        image.save(temp_file.name)

        # Read the temporary file contents as bytes
        image_bytes = temp_file.read()

    # Prepare the file for sending it to FastAPI
    files = {"file": ("image.png", image_bytes, "image/png")}

    # Make a POST request to FastAPI
    fastapi_url = "https://aimaging11-uz7skuvrea-ew.a.run.app/organ_detection_model"
    response = requests.post(fastapi_url, files=files)

    # Display the prediction
    if response.status_code == 200:
        result = response.json()
        organ = result.get('organ')
        disease_status = result.get('disease_status')
        class_prediction = result.get('class_prediction')

        # Return structured information
        return {
            'Organ': organ,
            'Disease Status': disease_status,
            'Class Prediction': class_prediction
        }
    else:
        st.error("Error making prediction. Please try again.")
