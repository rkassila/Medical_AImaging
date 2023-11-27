import streamlit as st
from PIL import Image

def app():
    st.title("Analysis result")

    # Retrieve the result from session_state
    result = st.session_state.result

    # Display the result
    if result is not None:
        st.success(f"Analysis result: {result}")
    else:
        st.write("Please upload and scan an image first.")
