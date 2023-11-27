import streamlit as st
from PIL import Image
import time

def app():
    st.title("Analysis result")

    # Retrieve the result from session_state
    result = st.session_state.result


    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)

    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(1)
    my_bar.empty()

    st.button("Rerun")

    # Display the result
    if result is not None:
        st.success(f"Analysis result: {result}")
    else:
        st.write("Please upload and scan an image first.")

    st.text("SHAP example")
    shap_image = Image.open('aimaging/interface/shape_example.png')
    st.image(shap_image)
    st.divider()

    st.text("grad cam example 1")
    grad_image_1 = Image.open('aimaging/interface/grad_cam_1_example.png')
    st.image(grad_image_1)
    st.divider()

    st.text("grad cam example 2")
    grad_image_2 = Image.open('aimaging/interface/grad_cam_2_example.png')
    st.image(grad_image_2)
