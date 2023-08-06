from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import cv2
import segmentation_models_pytorch as smp
import torch
from config import PATH_TO_MODEL, DEVICE, TRANSFORM_VAL_TEST
import io
import base64

app = Flask(__name__)

# Load U-Net model and other necessary initializations here
model = smp.Unet(
    encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=1,  # model output channels (number of classes in your dataset)
    activation=None
    )
model.load_state_dict(torch.load(PATH_TO_MODEL))
model.to(DEVICE)
model.eval()


def apply_mask(image):
    # Convert the image to numpy array
    image_np = np.array(image) / 255.0

    # Apply the model to the image
    with torch.no_grad():
        # Convert the image to tensor and ensure it has the correct data type (float32)
        image_tensor = TRANSFORM_VAL_TEST(image=image_np)['image'].to(DEVICE).float()

        # Ensure the image tensor has 4 dimensions (batch size, channels, height, width)
        image_tensor = image_tensor.unsqueeze(0)

        # Apply the model to get the prediction
        prediction = model(image_tensor)
        prediction = torch.sigmoid(prediction).cpu().squeeze().numpy()

    # Threshold the prediction to get a binary mask
    mask = (prediction > 0.5).astype(np.uint8)

    # Resize the mask to match the shape of the resized image
    mask = cv2.resize(mask, (image_np.shape[1], image_np.shape[0]))

    # Create a new RGBA image with the same size as the original image
    rgba_image = np.zeros((image_np.shape[0], image_np.shape[1], 4), dtype=np.uint8)

    # Set the RGB channels to yellow (R=255, G=255, B=0)
    rgba_image[:, :, 0] = 255  # R channel
    rgba_image[:, :, 1] = 0  # G channel
    rgba_image[:, :, 2] = 0  # B channel

    # Set the alpha channel based on the resized predicted mask
    rgba_image[:, :, 3] = mask * 100

    # Convert the numpy array to PIL Image
    rgba_image = Image.fromarray(rgba_image)

    # Convert the image to RGBA mode to ensure it has an alpha channel
    image = image.convert('RGBA')

    # Superimpose the RGBA mask on the original image
    image_with_mask = Image.alpha_composite(image, rgba_image)

    # Convert the image to bytes and return as bytes object
    output_bytes = io.BytesIO()
    image_with_mask.save(output_bytes, format="PNG")
    output_bytes = output_bytes.getvalue()

    return output_bytes


@app.route('/process_image', methods=['POST'])
def process_image():
    # Get the uploaded image from the request
    image = request.files['file']
    image = Image.open(image)

    # Apply the model and get the result as bytes
    result_bytes = apply_mask(image)

    # Convert bytes to base64-encoded string
    result_base64 = base64.b64encode(result_bytes).decode('utf-8')

    return jsonify({'prediction': result_base64})



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)