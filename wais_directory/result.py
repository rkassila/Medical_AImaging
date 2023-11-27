import streamlit as st
from PIL import Image

def app():
    st.title("Analysis result")

    # Display the result stored in session_state
    if hasattr(st.session_state, "scan_result"):
        st.write("Scan Result:")
        st.write(st.session_state.scan_result)
    else:
        st.write("Please upload and scan an image first.")
