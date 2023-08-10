import shutil
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import cv2
import segmentation_models_pytorch as smp
import torch
from config import PATH_TO_MODEL, DEVICE, TRANSFORM_VAL_TEST
import io
import base64
import os


app = Flask(__name__)

# Load the pre-trained UNet model for image segmentation
model = smp.Unet(
    encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=1,  # model output channels (number of classes in your dataset)
    activation=None
)
model.load_state_dict(torch.load(PATH_TO_MODEL, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Define a function to apply a segmentation mask to an image
def apply_mask(image):
    # Convert image to NumPy array and preprocess for model input
    image_np = np.array(image) / 255.0
    with torch.no_grad():
        image_tensor = TRANSFORM_VAL_TEST(image=image_np)['image'].to(DEVICE).float()
        image_tensor = image_tensor.unsqueeze(0)
        prediction = model(image_tensor)
        prediction = torch.sigmoid(prediction).cpu().squeeze().numpy()

    # Apply threshold to prediction for mask creation
    mask = (prediction > 0.5).astype(np.uint8)
    mask = cv2.resize(mask, (image_np.shape[1], image_np.shape[0]))

    # Create an RGBA image with the segmentation mask
    rgba_image = np.zeros((image_np.shape[0], image_np.shape[1], 4), dtype=np.uint8)
    rgba_image[:, :, 0] = 255
    rgba_image[:, :, 1] = 0
    rgba_image[:, :, 2] = 0
    rgba_image[:, :, 3] = mask * 100
    rgba_image = Image.fromarray(rgba_image)

    # Combine original image with the segmentation mask
    image = image.convert('RGBA')
    image_with_mask = Image.alpha_composite(image, rgba_image)

    # Convert the result image to bytes
    output_bytes = io.BytesIO()
    image_with_mask.save(output_bytes, format="PNG")
    output_bytes = output_bytes.getvalue()
    return output_bytes

# Define a route to process uploaded images
@app.route('/process_image', methods=['POST'])
def process_image():
    image = request.files['file']
    image = Image.open(image)
    result_bytes = apply_mask(image)
    result_base64 = base64.b64encode(result_bytes).decode('utf-8')
    return jsonify({'prediction': result_base64})

# Define a route to process uploaded video
@app.route('/process_video', methods=['POST'])
def process_video():
    video = request.files['file']
    video.save(video.filename)
    cap = cv2.VideoCapture(video.filename)
    fps = cap.get(cv2.CAP_PROP_FPS)
    output_folder = 'processed_frames'
    os.makedirs(output_folder, exist_ok=True)
    frame_count = 0

    # Process each frame in the video
    while True:
        ret, image = cap.read()
        if not ret:
            break
        image = Image.fromarray(image)
        result = apply_mask(image)
        result_image = Image.open(io.BytesIO(result))
        result_image.save(os.path.join(output_folder, f'frame_{frame_count:04d}.png'))
        frame_count += 1
        if frame_count == 3:
            break
    cap.release()

    # Convert processed frames to a video
    image_files = sorted(os.listdir('./processed_frames'))
    if not image_files:
        return None
    image_path = os.path.join('./processed_frames', image_files[0])
    frame = cv2.imread(image_path)
    height, width, layers = frame.shape
    video_writer = cv2.VideoWriter('./output.webm', cv2.VideoWriter_fourcc(*'VP80'), fps, (width, height))

    for image_file in image_files:
        image_path = os.path.join('./processed_frames', image_file)
        frame = cv2.imread(image_path)
        video_writer.write(frame)

    video_writer.release()
    shutil.rmtree(output_folder)

    # Read the final video and return it as the result
    with open('./output.webm', 'rb') as f:
        result = f.read()
    return result

# Start the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)