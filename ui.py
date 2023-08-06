import requests
import numpy as np
import streamlit as st
import os
from PIL import Image
from io import BytesIO
import base64

# Determine the API URL based on the execution environment
if "DOCKER_MODE" in os.environ:
    api_url = "http://api:5000"
else:
    api_url = "http://localhost:5000"

st.title("TikTok dancing segmentation")
st.write('Upload your file!')
# Select the input type: video, image
input_type = st.selectbox("Select the input type", ("Image", "Video"))

if input_type == "Image":
    # Upload the input image file
    input_image = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])

    if input_image is not None:
        # Process the image and get the output bytes
        image = Image.open(input_image)
        bytes_image = BytesIO()
        image.save(bytes_image, format="JPEG")
        bytes_image = bytes_image.getvalue()
        im = st.image(image, use_column_width=True)
        if st.button('Run'):
            im.empty()
            t = st.empty()
            t.markdown('Running...')
            predicted = requests.post(f"{api_url}/process_image", files={'file': bytes_image})
            predicted = predicted.json()
            # Decode the base64 string back to bytes
            result_bytes = base64.b64decode(predicted['prediction'])
            # Open the image from bytes
            result_image = Image.open(BytesIO(result_bytes))
            t.empty()
            t.markdown('Your prediction:')
            st.image(result_image, use_column_width=True)
