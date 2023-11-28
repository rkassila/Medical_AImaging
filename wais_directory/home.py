import streamlit as st
from PIL import Image
import requests

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

            # Store the result in session_state
            st.session_state.result = result

            # Redirect to the Result page
            st.experimental_rerun()

# Function to scan the image
def scan_image(image):
    url = "http://127.0.0.1:8000/organ_detection_model"

    response = requests.get(url).json()

    return response
