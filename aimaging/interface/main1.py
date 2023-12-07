import streamlit as st
from PIL import Image
import requests
import tempfile
import io
import time
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Organ Disease Detector",
    layout='wide'
)

URL = "http://127.0.0.1:8000"

def app():
    start_time = time.time()

    # Title image
    title_col1, title_col2, title_col3 = st.columns([1, 1, 1])
    with title_col2:
        title_image = Image.open('aimaging/interface/images/title_image.png')
        title_image = title_image.resize((1696, 541))
        st.image(title_image, use_column_width=True, width=550)

    # Read the image
    image = Image.open('aimaging/interface/images/streamlit_bg.png')
    st.image(image, use_column_width=True)

    uploaded_image = st.file_uploader("Upload an image of your organ", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Display the uploaded image
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            image = Image.open(uploaded_image)
            image = image.resize((224, 224))
            st.image(image, caption="Uploaded Image", width=400)

        # Button to trigger the scanning process
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            scan_button = st.button("SCAN", key="scan_button", help="Click to initiate the scan.", type="primary",
                                    use_container_width=True)

        if scan_button:
            # Create a loading bar
            progress_bar = st.progress(0, text="Processing Images...")

            for percent_complete in range(100):
                time.sleep(0.001)
                progress_bar.progress(percent_complete + 1, "Processing Images...")
                time.sleep(0.001)

            # Perform the scanning process here
            with st.expander("Analysis Result"):
                result = scan_image(image)

                if result is not None:
                    # Display the prediction details
                    st.write("# Analysis Result")
                    display_prediction(result)

                    st.write("## Organ Detection")
                    display_shap_image()

            # Check if 'Disease Status' key exists in result
            if result and 'Disease Status' in result:
                disease_status = result['Disease Status'].lower()

                if disease_status == 'diseased':
                    progress_bar = st.progress(0, text="Testing for Diseases...")

                    for percent_complete in range(100):
                        time.sleep(0.05)
                        progress_bar.progress(percent_complete + 1, "Testing for Diseases...")
                        time.sleep(0.05)

                    with st.expander("Disease Detection"):
                        display_disease_estimation(result)

                         # Display two grad images side by side
                        col1, col2 = st.columns(2)
                        with col1:
                            display_grad_image(1)
                        with col2:
                            display_grad_image(2)

            else:
                st.warning("Unable to retrieve 'Disease Status' from the result.")

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Elapsed Time: {elapsed_time} seconds")


def scan_image(image):
    with tempfile.NamedTemporaryFile(suffix=".png") as temp_file:
        image.save(temp_file.name)

        image_bytes = temp_file.read()

    files = {"file": ("image.png", image_bytes, "image/png")}
    fastapi_url = URL + "/organ_detection_model"
    response = requests.post(fastapi_url, files=files)

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


def display_prediction(result):
    organ = result.get('Organ', 'N/A')
    disease_status = result.get('Disease Status', 'N/A')
    organ_emojis = {'brain': '🧠', 'spine': '🩻', 'knee': '🦵', 'lung': '🫁', 'shoulder': '💪'}
    disease_emoji = {'healthy': '✅', 'diseased': '⚠️'}

    new_col1, new_col2 = st.columns(2)

    with new_col1:
        st.write(f"<p style='font-size: 40px;font-weight: bold; text-align: center;'>{organ.title()}</p>",
                 unsafe_allow_html=True)
        st.write(f"<p style='font-size: 120px; text-align: center;'>{organ_emojis.get(organ, 'N/A')}</p>",
                 unsafe_allow_html=True)

    with new_col2:
        color = 'red' if disease_status.lower() != 'healthy' else '#18db1b'

        st.write(
            f"<p style='font-size: 40px; color: {color}; font-weight: bold; text-align: center;'>{disease_status.title()} </p>",
            unsafe_allow_html=True)
        st.write(f"<p style='font-size: 120px; text-align: center;'>{disease_emoji.get(disease_status, 'N/A')}</p>",
                 unsafe_allow_html=True)


def display_shap_image():
    shap_url = URL + "/shap-image"
    response = requests.get(shap_url)

    if response.status_code == 200:
        try:
            image_bytes = io.BytesIO(response.content)
            shap_image = Image.open(image_bytes)
            shap_image_cropped = shap_image.crop((200, 100, 1648, 430))
            st.image(shap_image_cropped, use_column_width=True)
        except Exception as e:
            st.error(f"Error opening SHAP image: {e}")


def display_disease_estimation(result):
    class_prediction = result.get('Class Prediction', [])

    if class_prediction:
        st.write("## Disease Estimation")

        sorted_predictions = sorted(zip(get_class_names(result['Organ']), class_prediction[0]),
                                    key=lambda x: x[1], reverse=True)

        labels = [class_name for class_name, _ in sorted_predictions[:3]]
        percentages = [percentage * 100 for _, percentage in sorted_predictions[:3]]

        colors = ['#fc9b9b', '#e65555', '#e20000']

        fig, ax = plt.subplots(figsize=(13, 6))
        fig.patch.set_alpha(0.0)

        bars = ax.barh(labels[::-1], percentages[::-1], color=colors)

        for bar, percentage in zip(bars, percentages[::-1]):
            ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f'  {percentage:.2f}%',
                    va='center', ha='left', color='white', fontsize=24)

        ax.xaxis.label.set_color('white')
        ax.xaxis.label.set_fontsize(24)
        ax.yaxis.label.set_color('white')
        ax.yaxis.label.set_fontsize(24)

        ax.tick_params(axis='x', colors='white', labelsize=20)
        ax.tick_params(axis='y', colors='white', labelsize=20)

        ax.set_facecolor('black')

        for label in ax.get_yticklabels():
            label.set_color('white')
            label.set_fontsize(24)

        st.pyplot(fig)


def display_grad_image(grad_number):
    grad_url = URL + f"/grad-image{grad_number}"
    response = requests.get(grad_url)

    if response.status_code == 200:
        try:
            image_bytes = io.BytesIO(response.content)
            grad_image = Image.open(image_bytes)
            grad_image_cropped = grad_image.crop((150, 60, 500, 427))
            st.image(grad_image_cropped, use_column_width=True)
        except Exception as e:
            st.error(f"Error opening Grad image {grad_number}: {e}")

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
