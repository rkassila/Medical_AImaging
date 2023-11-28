import streamlit as st
from PIL import Image
import requests
import tempfile
from io import BytesIO

def app():
    # Set title alignment and size
    st.title("Organ Disease Detector 🔍")

    # Read the image
    image = Image.open('wais_directory/streamlit_bg.png')

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

            # Display the result
            st.write("### Prediction Result:")
            if result is not None:
                st.write(f"- Organ: {result.get('Organ', 'N/A')}")
                st.write(f"- Disease Status: {result.get('Disease Status', 'N/A')}")

                # Display Class Predictions with Percentages
                class_prediction = result.get('Class Prediction')
                if class_prediction is not None:
                    st.write("#### Class Predictions:")
                    organ_names = ["knee", "brain", "shoulder", "spine", "lung"]
                    for organ, percentage in zip(organ_names, class_prediction):
                        st.write(f"- {organ.capitalize()}: {percentage * 100:.2f}%")


                # Display SHAP image
                st.write("### Display SHAP Image")
                shap_url = "http://localhost:8000/shap-image"
                shap_response = requests.get(shap_url)
                if shap_response.status_code == 200:
                    shap_image = Image.open(BytesIO(shap_response.content))
                    st.image(shap_image, caption="SHAP Image", use_column_width=True)
                else:
                    st.error("Error retrieving SHAP image.")

                # Display Grad-CAM image
                st.write("### Display Grad-CAM Image")
                grad_url = "http://localhost:8000/grad-image"
                grad_response = requests.get(grad_url)
                if grad_response.status_code == 200:
                    grad_image = Image.open(BytesIO(grad_response.content))
                    st.image(grad_image, caption="Grad-CAM Image", use_column_width=True)
                else:
                    st.error("Error retrieving Grad-CAM image.")

            else:
                st.error("Error making prediction. Please try again.")





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
    fastapi_url = "http://localhost:8000/organ_detection_model"
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
