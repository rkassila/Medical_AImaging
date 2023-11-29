import streamlit as st
from PIL import Image
import requests
import tempfile
import io
import time

st.set_page_config(
    page_title="Organ Disease Detector"
)

URL = "https://aimaging18-uz7skuvrea-ew.a.run.app/"

def app():
    # Set title alignment and size
    st.title("Organ Disease Detector")

    # Read the image
    image = Image.open('aimaging/interface/streamlit_bg.png')

    # Display the image with wide layout
    st.image(image, use_column_width=True)

    uploaded_image = st.file_uploader("Upload an image of your organ", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        image = image.resize((224,224))
        st.image(image, caption="Uploaded Image")

        # Button to trigger the scanning process
        col1, col2, col3 = st.columns([2,1,2])
        with col2:
            scan_button = st.button("SCAN", key="scan_button", help="Click to initiate the scan.", type = "primary", use_container_width=True)


        if scan_button:
            # Create a loading spinner
            with st.status("Loading Image...", expanded=True) as status:
                st.write("Classifying Organ...")
                time.sleep(5)
                st.write("Testing for Diseases...")
                time.sleep(5)
                st.write("Generating Images...")
                time.sleep(5)

                # Perform the scanning process here
                result = scan_image(image)
                status.update(label="Evaluation  Complete!", state="complete", expanded=True)
                # Display the result
                if result is not None:
                    # Display the prediction details
                    st.write("# Analysis Result")
                    organ = result.get('Organ', 'N/A')
                    disease_status = result.get('Disease Status', 'N/A')
                    organ_emojis = {
                        'brain':'🧠',
                        'spine':'🩻',
                        'knee':'🦵',
                        'lung':'🫁',
                        'shoulder':'🙋'
                    }
                    disease_emoji = {
                        'healthy':'✅',
                        'diseased':'⚠️'
                    }
                    new_col1, new_col2 = st.columns(2)

                    with new_col1:
                        st.write(f"<p style='font-size: 40px;font-weight: bold; text-align: center;'>{organ.title()}</p>", unsafe_allow_html=True)
                        st.write(f"<p style='font-size: 120px; text-align: center;'>{organ_emojis.get(organ, 'N/A')}</p>", unsafe_allow_html=True)


                    with new_col2:
                        color = 'red' if disease_status.lower() != 'healthy' else '#18db1b'

                        st.write(f"<p style='font-size: 40px; color: {color}; font-weight: bold; text-align: center;'>{disease_status.title()} </p>", unsafe_allow_html=True)
                        st.write(f"<p style='font-size: 120px; text-align: center;'>{disease_emoji.get(disease_status, 'N/A')}</p>", unsafe_allow_html=True)


                    st.write("## SHAP")
                    shap_url = URL + "/shap-image"
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



                    st.divider()
                    class_prediction = result.get('Class Prediction', [])
                    if class_prediction:
                        st.write("## Class Predictions")

                        # Sort class predictions by percentage in descending order
                        sorted_predictions = sorted(zip(get_class_names(result['Organ']), class_prediction[0]), key=lambda x: x[1], reverse=True)

                        # Display a horizontal bar chart with the top 3 classes
                        import matplotlib.pyplot as plt

                        labels = [class_name for class_name, _ in sorted_predictions[:3]]
                        percentages = [percentage * 100 for _, percentage in sorted_predictions[:3]]

                        # Choose custom colors for each bar
                        colors = ['#fc9b9b', '#e65555', '#e20000']

                        # Set a transparent background for the entire figure
                        fig, ax = plt.subplots(figsize=(13, 6))
                        fig.patch.set_alpha(0.0)  # Set transparency

                        bars = ax.barh(labels[::-1], percentages[::-1], color=colors)

                        # Display the percentage values inside the bars
                        for bar, percentage in zip(bars, percentages[::-1]):
                            ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f'  {percentage:.2f}%',
                                    va='center', ha='left', color='white', fontsize=24)  # Adjusted fontsize

                        # Set axis label text color and fontsize to white
                        ax.xaxis.label.set_color('white')
                        ax.xaxis.label.set_fontsize(24)  # Adjusted fontsize
                        ax.yaxis.label.set_color('white')
                        ax.yaxis.label.set_fontsize(24)  # Adjusted fontsize

                        # Set tick color and fontsize to white
                        ax.tick_params(axis='x', colors='white', labelsize=20)  # Adjusted fontsize
                        ax.tick_params(axis='y', colors='white', labelsize=20)  # Adjusted fontsize

                        # Set individual background color for the axis
                        ax.set_facecolor('black')

                        # Ensure that the class names are visible
                        for label in ax.get_yticklabels():
                            label.set_color('white')
                            label.set_fontsize(24)  # Adjusted fontsize

                        # Display the bar chart using streamlit
                        st.pyplot(fig)


                    # Display the Grad images
                    # st.divider()
                    if disease_status.lower() == 'diseased':
                        st.write("## GradCAM")

                        col1, col2 = st.columns(2)

                        with col1:
                            grad_url = URL + "/grad-image"
                            response = requests.get(grad_url)

                            if response.status_code == 200:
                                try:
                                    # Convert binary image data to BytesIO
                                    image_bytes = io.BytesIO(response.content)
                                    # Attempt to open the image
                                    grad_image = Image.open(image_bytes)
                                    # Crop the image to a specific size (adjust these values as needed)
                                    grad_image_cropped = grad_image.crop((150, 60, 500, 427))
                                    st.image(grad_image_cropped, use_column_width=True)
                                except Exception as e:
                                    st.error(f"Error opening Grad image: {e}")

                        with col2:
                            grad_url2 = URL + "/grad-image2"
                            response = requests.get(grad_url2)

                            if response.status_code == 200:
                                try:
                                    # Convert binary image data to BytesIO
                                    image_bytes = io.BytesIO(response.content)
                                    # Attempt to open the image
                                    grad_image = Image.open(image_bytes)
                                    # Crop the image to a specific size (adjust these values as needed)
                                    grad_image_cropped = grad_image.crop((150, 60, 500, 427))
                                    st.image(grad_image_cropped, use_column_width=True)
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
    fastapi_url = URL + "/organ_detection_model"
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
