import streamlit as st
from PIL import Image
import requests
import tempfile
import io

st.set_page_config(
    page_title="Organ Disease Detector 🔍"
)

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
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            scan_button = st.button("Scan", key="scan_button", help="Click to initiate the scan.")

        # Apply custom CSS to center the button
        scan_button_style = (
            "<style>"
            ".stButton {display: flex; justify-content: center; align-items: center; font-size: 20px; padding: 15px;}"
            "</style>"
        )
        st.markdown(scan_button_style, unsafe_allow_html=True)

        if scan_button:
            # Create a loading spinner
            with st.spinner("Scanning..."):
                # Perform the scanning process here
                result = scan_image(image)

                # Display the result
                if result is not None:
                    # Display the prediction details
                    st.write("### Analysis Result:")
                    st.write(f"- Organ: {result.get('Organ', 'N/A')}")
                    st.write(f"- Disease Status: {result.get('Disease Status', 'N/A')}")

                    # Display the Class Prediction
                    class_prediction = result.get('Class Prediction', [])
                    if class_prediction:
                        st.write("### Class Predictions:")

                        # Sort class predictions by percentage in descending order
                        sorted_predictions = sorted(zip(get_class_names(result['Organ']), class_prediction[0]), key=lambda x: x[1], reverse=True)

                        # Display only the top 3 classes
                        for i, (class_name, percentage) in enumerate(sorted_predictions[:3]):
                            if i == 0:
                                # Increase font size for the first class
                                st.write(f"<p style='font-size:24px;'>{class_name}: {percentage * 100:.2f}%</p>", unsafe_allow_html=True, key=f"class_{i}")
                            else:
                                st.write(f"{class_name}: {percentage * 100:.2f}%", key=f"class_{i}")

                    # Display the SHAP image
                    shap_url = "http://127.0.0.1:8000/shap-image"
                    response = requests.get(shap_url)

                    if response.status_code == 200:
                        try:
                            # Convert binary image data to BytesIO
                            image_bytes = io.BytesIO(response.content)
                            # Attempt to open the image
                            shap_image = Image.open(image_bytes)
                            # Crop the image to a specific size (adjust these values as needed)
                            shap_image_cropped = shap_image.crop((200, 100, 1648, 430))
                            st.image(shap_image_cropped, caption="SHAP Image", use_column_width=True)
                        except Exception as e:
                            st.error(f"Error opening SHAP image: {e}")


                    # Display the Grad images

                    col1, col2 = st.columns(2)

                    with col1:
                        grad_url = "http://127.0.0.1:8000/grad-image"
                        response = requests.get(grad_url)

                        if response.status_code == 200:
                            try:
                                # Convert binary image data to BytesIO
                                image_bytes = io.BytesIO(response.content)
                                # Attempt to open the image
                                grad_image = Image.open(image_bytes)
                                # Crop the image to a specific size (adjust these values as needed)
                                grad_image_cropped = grad_image.crop((150, 60, 500, 427))
                                st.image(grad_image_cropped, caption="Grad Image 1", use_column_width=True)
                            except Exception as e:
                                st.error(f"Error opening Grad image: {e}")

                    with col2:
                        grad_url = "http://127.0.0.1:8000/grad-image2"
                        response = requests.get(grad_url)

                        if response.status_code == 200:
                            try:
                                # Convert binary image data to BytesIO
                                image_bytes = io.BytesIO(response.content)
                                # Attempt to open the image
                                grad_image = Image.open(image_bytes)
                                # Crop the image to a specific size (adjust these values as needed)
                                grad_image_cropped = grad_image.crop((150, 60, 500, 427))
                                st.image(grad_image_cropped, caption="Grad Image 2", use_column_width=True)
                            except Exception as e:
                                st.error(f"Error opening Grad image: {e}")

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
    fastapi_url = "http://127.0.0.1:8000/organ_detection_model"
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


# Function to get class names based on the organ
def get_class_names(organ):
    class_names = {
        "knee": ['Soft Fluid', 'Anterior Cruciate Ligament Injury', 'Bone Inflammation', 'Chondral Injury', 'Fracture',
                 'Intra-articular Pathology', 'Meniscal Injury',  'Patellar Injury', 'Posterior Cruciate Ligament Injury'],
        "brain": ['Acute Infarction', 'Chronic Infarction', 'Extra-axial Pathology', 'Focal Flair Hyperintensity',
                  'Intra-brain Pathology', 'White Matter Changes'],
        "shoulder": ['Acromioclavicular Joint Osteoarthritis', 'Biceps Pathology', 'Glenohumeral Joint Osteoarthritis',
                     'Labral Pathology', 'Marrow Inflammation', 'Osseous Lesion','Post-operative Changes', 'Soft Tissue Edema',
                     'Soft Tissue Fluid in Shoulder', 'Supraspinatus Pathology'],
        "spine": ['Cord Pathology', 'Cystic Lesions', 'Disc Pathology', 'Osseous Abnormalities'],
        "lung": [ 'Airspace Opacity', 'Bronchiectasis', 'Nodule', 'Parenchyma Destruction', 'Interstitial Lung Disease']
    }
    return class_names.get(organ, [])  # Return an empty list if organ is not found

if __name__ == "__main__":
    app()
