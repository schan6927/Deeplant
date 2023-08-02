from flask import Flask, request, jsonify
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image
import torch
from cam import visualize_all_vit_model
import boto3
import io
import numpy as np
import requests
from urllib.parse import urlparse

# Initialize the Flask app
app = Flask(__name__)

# Define the available CAM types
CAM_TYPES = ["GradCAM", "GradCAMPlusPlus", "XGradCAM", "ScoreCAM", "EigenCAM"]


def grade_decoder(x):
    if x == 0:
        return "1++"
    elif x == 1:
        return "1+"
    elif x == 2:
        return "1"
    elif x == 3:
        return "2"
    elif x == 4:
        return "3"


class MLModel:
    def __init__(self):
        self.grade_model_pth = "./model/grade_model.pth"
        self.sensory_model_pth = "./model/sensory_model.pth"
        self.transformation = Compose([ToTensor()])

    def save_image_to_s3(self, rgb_list, bucket_name, file_name):
        s3 = boto3.client("s3")
        try:
            array = np.array(rgb_list, dtype=np.uint8)
            image = Image.fromarray(array)
            image_byte_stream = io.BytesIO()
            image.save(image_byte_stream, format="png")
            image_byte_stream.seek(0)
            s3.upload_fileobj(image_byte_stream, bucket_name, file_name)

            return True
        except Exception as e:
            print("Error:", str(e))
            return False

    def load_grade_model(self):
        model_pth = self.grade_model_pth
        return torch.load(model_pth).eval()

    def load_sensory_model(self):
        model_pth = self.sensory_model_pth
        return torch.load(model_pth).eval()

    def process_image(self, image_path):
        try:
            # Check if the image_path is a URL (from the internet)
            parsed_url = urlparse(image_path)
            if parsed_url.scheme in ("http", "https"):
                res = requests.get(image_path)
                res.raise_for_status()  # Raise an exception for any HTTP errors
                image = Image.open(io.BytesIO(res.content))
            else:
                # Assume the image_path is a local file path
                image = Image.open(image_path)

            image = image.resize((448, 448))
            image_tensor = self.transformation(image).unsqueeze(0)
            image_tensor = image_tensor.to("cuda")
            return image_tensor, image
        except requests.exceptions.RequestException as req_ex:
            return (
                jsonify(
                    {
                        "error": "Error fetching the image from the URL",
                        "description": str(req_ex),
                    }
                ),
                404,
            )
        except Exception as e:
            return (
                jsonify({"error": "Error processing the image", "description": str(e)}),
                404,
            )

    def predict(
        self, image_path, cam_type="GradCAM", bucket_name="xai-deep-plant-image"
    ):
        grade_model = self.load_grade_model()
        sensory_model = self.load_sensory_model()
        image_tensor, image = self.process_image(image_path)

        with torch.no_grad():
            grade_output = grade_model(image_tensor)
            grade_output = grade_output.tolist()
            grade_output = grade_decoder(grade_output[0].index(max(grade_output[0])))

            sensory_output = sensory_model(image_tensor)
            sensory_output = sensory_output.tolist()

        # Get the GradCAM visualization
        if cam_type not in CAM_TYPES:
            return jsonify({"error": "Invalid CAM type XAI"}), 404

        # XAI image
        img_size = 448
        target_classes = None
        grade_visualizations = visualize_all_vit_model(
            grade_model, image, cam_type, img_size, target_classes
        )
        sensory_visualizations = visualize_all_vit_model(
            sensory_model, image, cam_type, img_size, target_classes
        )

        xai_gradeNum_imageName = "grade_xai.png"
        xai_imageName = "sensory_xai.png"

        self.save_image_to_s3(grade_visualizations, bucket_name, xai_gradeNum_imageName)
        self.save_image_to_s3(sensory_visualizations, bucket_name, xai_imageName)
        bucketPath = "https://xai-deep-plant-image.s3.ap-northeast-2.amazonaws.com/"
        xai_imagePath = bucketPath + xai_imageName

        xai_gradeNum_imagePath = bucketPath + xai_gradeNum_imageName

        try:
            return (
                jsonify(
                    {
                        "xai_gradeNum_imagePath": xai_gradeNum_imagePath,
                        "gradeNum": grade_output,
                        "xai_imagePath": xai_imagePath,
                        "marbling": sensory_output[0][0],
                        "color": sensory_output[0][1],
                        "texture": sensory_output[0][2],
                        "surfaceMoisture": sensory_output[0][3],
                        "overall": sensory_output[0][4],
                    }
                ),
                200,
            )
        except Exception as e:
            # Handle the exception and return a 404 error with a description
            return (
                jsonify({"error": "Error occurred", "description": str(e)}),
                404,
            )


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    if "image_path" not in data:
        return jsonify({"error": "Invalid request data image_path"}), 404

    image_path = data["image_path"]

    model_api = MLModel()
    return model_api.predict(image_path)


if __name__ == "__main__":
    app.run("0.0.0.0", debug=True)
