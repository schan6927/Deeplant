import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import timm
import cv2
import matplotlib.ticker as ticker
import torch.nn.functional as F

from torchvision.transforms import ToTensor
from pathlib import Path
from PIL import Image
from torchvision import datasets, transforms

from pytorch_grad_cam import (
    GradCAM,
    HiResCAM,
    ScoreCAM,
    GradCAMPlusPlus,
    AblationCAM,
    XGradCAM,
    EigenCAM,
    FullGrad,
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import (
    show_cam_on_image,
    preprocess_image,
    deprocess_image,
)
from pytorch_grad_cam import GuidedBackpropReLUModel


class CAMVisualizer:
    def __init__(self, model, img_size=448, use_cuda=False):
        self.model = model
        self.img_size = img_size
        self.use_cuda = use_cuda

    def reshape_vit_transform(self, tensor, height=14, width=14):
        result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
        result = result.transpose(2, 3).transpose(1, 2)
        return result

    def visualize_vit_model(self, image, cam_type, target_classes=None):
        transform = transforms.Compose([transforms.ToTensor()])
        input_tensor = transform(image).unsqueeze(0)

        if cam_type == "GradCAM":
            cam = GradCAM(
                self.model,
                target_layers=[self.model.blocks[-1].norm1],
                use_cuda=self.use_cuda,
                reshape_transform=self.reshape_vit_transform,
            )
        elif cam_type == "ScoreCAM":
            cam = ScoreCAM(
                self.model,
                target_layers=[self.model.blocks[-1].norm1],
                use_cuda=self.use_cuda,
                reshape_transform=self.reshape_vit_transform,
            )
        # Add more CAM types as needed.

        if target_classes is None:
            targets = None
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :]
            img_normalized = np.float32(image) / 255.0
            visualization = show_cam_on_image(
                img_normalized, grayscale_cam, use_rgb=True
            )
            return visualization.tolist()
        else:
            visualizations = []
            for target_class in target_classes:
                targets = [ClassifierOutputTarget(target_class)]
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
                grayscale_cam = grayscale_cam[0, :]
                img_normalized = np.float32(image) / 255.0
                visualization = show_cam_on_image(
                    img_normalized, grayscale_cam, use_rgb=True
                )
                visualizations.append(visualization.tolist())
            return visualizations

    def visualize_cnn_model(self, img_path, cam_type, target_classes=None):
        img = Image.open(img_path)
        transform = transforms.Compose(
            [transforms.Resize((self.img_size, self.img_size)), transforms.ToTensor()]
        )
        input_tensor = transform(img).unsqueeze(0)
        img = img.resize((self.img_size, self.img_size))

        if cam_type == "GradCAM":
            cam = GradCAM(
                self.model,
                target_layers=[self.model.layer4[-1]],
                use_cuda=self.use_cuda,
            )
        elif cam_type == "ScoreCAM":
            cam = ScoreCAM(
                self.model,
                target_layers=[self.model.layer4[-1]],
                use_cuda=self.use_cuda,
            )
        # Add more CAM types as needed.

        if target_classes is None:
            targets = None
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :]
            img_normalized = np.float32(img) / 255.0
            visualization = show_cam_on_image(
                img_normalized, grayscale_cam, use_rgb=True
            )
            return visualization.tolist()
        else:
            visualizations = []
            for target_class in target_classes:
                targets = [ClassifierOutputTarget(target_class)]
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
                grayscale_cam = grayscale_cam[0, :]
                img_normalized = np.float32(img) / 255.0
                visualization = show_cam_on_image(
                    img_normalized, grayscale_cam, use_rgb=True
                )
                visualizations.append(visualization.tolist())
            return visualizations


# %%
def visualize_cnn_model(model, img_path, cam_type, img_size, target_classes):
    # Load the image
    img = Image.open(img_path)

    # Preprocess the image
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]
    )
    input_tensor = transform(img).unsqueeze(0)
    img = img.resize((448, 448))

    # Specify the target layers
    target_layers = [model.layer4[-1]]

    # GradCAM instances for each target layer
    if cam_type == "GradCAM":
        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    elif cam_type == "ScoreCAM":
        cam = ScoreCAM(model=model, target_layers=target_layers, use_cuda=False)
    elif cam_type == "GradCAMPlusPlus":
        cam = GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=False)
    elif cam_type == "XGradCAM":
        cam = XGradCAM(model=model, target_layers=target_layers, use_cuda=False)
    elif cam_type == "EigenCAM":
        cam = EigenCAM(model=model, target_layers=target_layers, use_cuda=False)

    if target_classes == None:
        # Target classes and titles
        targets = None
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        img_normalized = np.float32(img) / 255.0
        visualization = show_cam_on_image(img_normalized, grayscale_cam, use_rgb=True)

        visualization = visualization.tolist()
        return visualization

    else:
        visualizations = []

        for target_class in target_classes:
            targets = [ClassifierOutputTarget(target_class)]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :]
            img_normalized = np.float32(img) / 255.0
            visualization = show_cam_on_image(
                img_normalized, grayscale_cam, use_rgb=True
            )
            visualizations.append(visualization)

        return visualizations


# How to use
visualizer = CAMVisualizer(model=vit_model, img_size=448, use_cuda=False)
# Load an image you want to visualize
image = Image.open("your_image.jpg")

# Choose the CAM technique (e.g., "GradCAM") and optional target classes (None for top predicted class)
# GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
cam_type = "GradCAM"
target_classes = (
    None  # Replace with a list of class indices if you want to target specific classes
)
visualization = visualizer.visualize_vit_model(image, cam_type, target_classes)
for vis in visualization:
    plt.figure()
    plt.axis("off")
    plt.imshow(vis)
    plt.show()
