# Import necessary libraries
import requests
import streamlit as st
import os
from PIL import Image
from io import BytesIO
import base64

# Check if the script is being run directly
if __name__ == '__main__':
    # Determine the API URL based on environment variable for Docker or local development
    if "DOCKER_MODE" in os.environ:
        api_url = "http://api:5000" # API URL for Docker mode
    else:
        api_url = "http://localhost:5000" # API URL for local development

    # Set up Streamlit app title and initial UI elements
    st.title("TikTok dancing segmentation")
    st.write('Upload your file!')

    # Allow user to choose input type (Image or Video)
    input_type = st.selectbox("Select the input type", ("Image", "Video"))

    # Handle image input
    if input_type == "Image":
        # Allow user to upload an image file
        input_image = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])
        if input_image is not None:
            # Open the uploaded image and prepare it for processing
            image = Image.open(input_image)
            bytes_image = BytesIO()
            image.save(bytes_image, format="JPEG")
            bytes_image = bytes_image.getvalue()
            im = st.image(image, use_column_width=True)

            # Execute processing when "Run" button is clicked
            if st.button('Run'):
                im.empty()
                t = st.empty()
                t.markdown('Running...')

                # Send a POST request to the image processing API and receive the prediction
                predicted = requests.post(f"{api_url}/process_image", files={'file': bytes_image})
                predicted = predicted.json()

                # Decode and display the processed image result
                result_bytes = base64.b64decode(predicted['prediction'])
                result_image = Image.open(BytesIO(result_bytes))
                t.empty()
                t.markdown('Your prediction:')
                st.image(result_image, use_column_width=True)

    # Handle video input
    elif input_type == "Video":
        # Allow user to upload a video file
        input_video = st.file_uploader("Upload a video file", type=["mp4", "rb", "webm", 'avi'])
        if input_video is not None:
            video = input_video.getvalue()
            v = st.video(video)

            # Execute processing when "Run" button is clicked
            if st.button('Run'):
                v.empty()
                t = st.empty()
                t.markdown('Running...')

                # Send a POST request to the video processing API and receive the prediction
                predicted = requests.post(f"{api_url}/process_video", files={'file': input_video})
                predicted = predicted.content

                # Display the processed video result
                st.video(predicted)
