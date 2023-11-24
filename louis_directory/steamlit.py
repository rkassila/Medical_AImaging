import streamlit as st
fo
st.title("Image Uploader Test")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    # Check if an image is uploaded
if uploaded_file is not None:
    # Process the image (you can add your own image processing logic here)
    image = Image.open(uploaded_file)
    image = image.resize((224, 224))
    st.write("Image Size:", image.size)
    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)
